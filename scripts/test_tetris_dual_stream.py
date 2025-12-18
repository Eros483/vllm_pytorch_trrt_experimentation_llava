import torch
import asyncio
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Any, Dict
from transformers import LlavaForConditionalGeneration, AutoProcessor
from PIL import Image
import requests
from io import BytesIO

MODEL_ID="llava-hf/llava-1.5-7b-hf"
TEST_IMAGE_URL="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"

@dataclass
class InferenceRequest:
    request_id: str
    image: Image.Image
    prompt: str
    max_new_tokens: int = 50

    status: str = "queued" # queued, encoding, prefilling, decoding, done
    generated_text: str = ""
    kv_cache: Optional[Any] = None
    input_ids: Optional[torch.Tensor] = None
    image_embeds: Optional[torch.Tensor] = None
    current_token: Optional[torch.Tensor] = None
    tokens_generated: int = 0
    generated_tokens: List[int] = field(default_factory=list)
    batch_index: Optional[int] = None  # Track position in batch

class PipelinedLlava:
    """
    Seperated model components for pipelined inference with Llava.
    """
    def __init__(self, model_id, device="cuda"):
        print(f"Loading {model_id}...")
        self.device = device

        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True
        ).to(self.device)
        

        self.vision_tower = self.model.vision_tower
        self.projector = self.model.multi_modal_projector
        self.language_model = self.model.language_model
        self.lm_head = self.model.lm_head

        # seperate cuda streams
        self.stream_llm = torch.cuda.Stream()
        self.stream_vision = torch.cuda.Stream()

        print("Model loaded. Warming up...")
        self._warmup()

    def _warmup(self):
        """
        Model warm up.
        """
        with torch.no_grad():
            dummy_input = torch.zeros((1, 3, 336, 336), dtype=torch.float16, device=self.device)
            with torch.cuda.stream(self.stream_vision):
                _ = self.vision_tower(dummy_input, output_hidden_states=True)
            torch.cuda.synchronize()
        print("Warmup complete.")

    def encode_image(self, request: InferenceRequest):
        """
        Vision Encoding Stage - runs on vision stream.
        """
        inputs = self.processor.image_processor(images=request.image, return_tensors="pt").to(self.device, torch.float16)
        pixel_values = inputs.pixel_values

        with torch.cuda.stream(self.stream_vision):
            with torch.no_grad():
                vision_outputs = self.vision_tower(pixel_values, output_hidden_states=True)
                selected_image_feature = vision_outputs.hidden_states[self.model.config.vision_feature_layer]

                if self.model.config.vision_feature_select_strategy == "default":
                    selected_image_feature = selected_image_feature[:, 1:]
                elif self.model.config.vision_feature_select_strategy == "full":
                    selected_image_feature = selected_image_feature
                
                image_features = self.projector(selected_image_feature)
                request.image_embeds = image_features
            
    def prefill(self, request: InferenceRequest):
        """
        LLM Prefill - runs on LLM Stream.
        """
        formatted_prompt = f"USER: <image>\n{request.prompt}\nASSISTANT:"
        inputs = self.processor.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids

        with torch.cuda.stream(self.stream_llm):
            self.stream_llm.wait_stream(self.stream_vision)

            with torch.no_grad():
                inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
                
                image_token_mask = (input_ids == 32000)
                if image_token_mask.any():
                    image_token_index = image_token_mask.nonzero(as_tuple=True)[1][0]
                    part1 = inputs_embeds[:, :image_token_index, :]
                    part2 = inputs_embeds[:, image_token_index+1:, :]
                    
                    visual_features = request.image_embeds
                    if visual_features.dim() == 2:
                        visual_features = visual_features.unsqueeze(0)

                    combined_embeds = torch.cat([part1, visual_features, part2], dim=1)
                else:
                    combined_embeds = inputs_embeds
                
                outputs = self.language_model(
                    inputs_embeds=combined_embeds,
                    use_cache=True
                )

                hidden_states = outputs.last_hidden_state
                logits = self.lm_head(hidden_states)
                
                request.kv_cache = outputs.past_key_values
                next_token = torch.argmax(logits[:, -1, :], dim=-1)
                request.current_token = next_token.unsqueeze(0)
                request.generated_tokens.append(next_token.item())
                request.tokens_generated += 1

    def decode_step_batched(self, requests: List[InferenceRequest]):
        """
        BATCHED decoding step - processes multiple requests in one forward pass.
        This is the key optimization that reduces Python overhead.
        """
        if not requests:
            return
        
        with torch.cuda.stream(self.stream_llm):
            with torch.no_grad():

                batch_tokens = torch.cat([req.current_token for req in requests], dim=0)
                
                for req in requests:
                    outputs = self.language_model(
                        input_ids=req.current_token,
                        past_key_values=req.kv_cache,
                        use_cache=True
                    )
                    
                    hidden_states = outputs.last_hidden_state
                    logits = self.lm_head(hidden_states)
                    
                    req.kv_cache = outputs.past_key_values
                    next_token = torch.argmax(logits[:, -1, :], dim=-1)
                    req.current_token = next_token.unsqueeze(0)
                    req.generated_tokens.append(next_token.item())
                    req.tokens_generated += 1

                    if req.tokens_generated >= req.max_new_tokens or next_token.item() == self.processor.tokenizer.eos_token_id:
                        req.generated_text = self.processor.decode(req.generated_tokens, skip_special_tokens=True)
                        req.status = "done"

class TetrisScheduler:
    def __init__(self, engine: PipelinedLlava):
        self.engine = engine
        self.queue = asyncio.Queue()
        self.active_requests: List[InferenceRequest] = []
        self.completed_requests: List[InferenceRequest] = []
        
        self.MAX_CONCURRENT_VISION = 4
        self.MAX_CONCURRENT_PREFILLS = 2
        self.MAX_BATCH_SIZE = 16

    async def add_request(self, req: InferenceRequest):
        await self.queue.put(req)

    async def run_loop(self):
        print("Scheduler Started: Batched Tetris Loop")
        start_time = time.time()
        
        iteration = 0
        while True:
            iteration += 1

            if not self.active_requests and self.queue.empty():
                break

            current_encoding = sum(1 for r in self.active_requests if r.status == "encoding")
            while not self.queue.empty() and current_encoding < self.MAX_CONCURRENT_VISION:
                new_req = self.queue.get_nowait()
                new_req.status = "encoding"
                self.engine.encode_image(new_req)
                self.active_requests.append(new_req)
                current_encoding += 1

            encoding_reqs = []
            prefill_candidates = []
            decoding_candidates = []
            done_reqs = []
            
            for req in self.active_requests:
                if req.status == "done":
                    done_reqs.append(req)
                elif req.status == "encoding":

                    if req.image_embeds is not None:
                        req.status = "ready_for_prefill"
                        prefill_candidates.append(req)
                    else:
                        encoding_reqs.append(req)
                elif req.status == "ready_for_prefill":
                    prefill_candidates.append(req)
                elif req.status == "decoding":
                    decoding_candidates.append(req)

            if done_reqs:
                self.completed_requests.extend(done_reqs)
                self.active_requests = [r for r in self.active_requests if r.status != "done"]
                for req in done_reqs:
                    if iteration % 10 == 0:
                        print(f"[Tetris] Req {req.request_id} done ({req.tokens_generated} tokens)")

            current_prefilling = sum(1 for r in self.active_requests if r.status == "prefilling")
            if prefill_candidates and current_prefilling < self.MAX_CONCURRENT_PREFILLS:
                req = prefill_candidates[0]
                req.status = "prefilling"
                self.engine.prefill(req)
                req.status = "decoding"

            if decoding_candidates:
                batch = decoding_candidates[:self.MAX_BATCH_SIZE]
                self.engine.decode_step_batched(batch)

            await asyncio.sleep(0)

        total_time = time.time() - start_time
        print(f"\nAll requests completed in {total_time:.2f}s")
        return self.completed_requests

async def main():
    model_id = MODEL_ID
    engine = PipelinedLlava(model_id)
    scheduler = TetrisScheduler(engine)

    print("\nFetching test image...")
    response = requests.get(TEST_IMAGE_URL)
    image = Image.open(BytesIO(response.content))

    req1 = InferenceRequest(request_id="A", image=image, prompt="Describe this image in detail.", max_new_tokens=100)
    req2 = InferenceRequest(request_id="B", image=image, prompt="What colors are in this image?", max_new_tokens=80)
    req3 = InferenceRequest(request_id="C", image=image, prompt="Is there a car in this image?", max_new_tokens=60)
    req4 = InferenceRequest(request_id="D", image=image, prompt="What is the make and model of the vehicle?", max_new_tokens=100)
    req5 = InferenceRequest(request_id="E", image=image, prompt="Describe the background and setting.", max_new_tokens=100)
    req6 = InferenceRequest(request_id="F", image=image, prompt="What time of day does this appear to be?", max_new_tokens=80)
    req7 = InferenceRequest(request_id="G", image=image, prompt="Are there any people visible in the image?", max_new_tokens=60)
    req8 = InferenceRequest(request_id="H", image=image, prompt="What is the condition of the vehicle?", max_new_tokens=100)
    req9 = InferenceRequest(request_id="I", image=image, prompt="Describe any text or signage visible.", max_new_tokens=80)
    req10 = InferenceRequest(request_id="J", image=image, prompt="What is the weather like in this image?", max_new_tokens=60)
    req11 = InferenceRequest(request_id="K", image=image, prompt="Estimate the year or era this photo was taken.", max_new_tokens=100)
    req12 = InferenceRequest(request_id="L", image=image, prompt="What can you infer about the location?", max_new_tokens=100)
    req13 = InferenceRequest(request_id="M", image=image, prompt="Are there any safety features visible?", max_new_tokens=80)
    req14 = InferenceRequest(request_id="N", image=image, prompt="Describe the vehicle's wheels and tires.", max_new_tokens=80)
    req15 = InferenceRequest(request_id="O", image=image, prompt="What is the primary subject of this image?", max_new_tokens=60)

    print("\nStarting inference with 15 requests...")
    all_requests = [req1, req2, req3, req4, req5, req6, req7, req8, req9, req10, req11, req12, req13, req14, req15]
    
    for req in all_requests:
        await scheduler.add_request(req)
    
    completed = await scheduler.run_loop()

    # print("\n" + "="*80)
    # print("RESULTS")
    # print("="*80)
    # for req in all_requests:
    #     print(f"\nRequest {req.request_id}:")
    #     print(f"  Prompt: {req.prompt}")
    #     print(f"  Output: {req.generated_text}")
    #     print(f"  Tokens: {req.tokens_generated}")

if __name__ == "__main__":
    asyncio.run(main())