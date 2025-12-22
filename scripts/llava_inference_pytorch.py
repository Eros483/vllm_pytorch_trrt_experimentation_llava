import torch
import time
from transformers import LlavaForConditionalGeneration, AutoProcessor
from PIL import Image
import requests
from io import BytesIO

MODEL_ID = "llava-hf/llava-1.5-7b-hf"
TEST_IMAGE_URL = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"

class BaselineLlava:
    """
    Standard sequential inference with Llava.
    Processes one request at a time from start to finish.
    """
    def __init__(self, model_id, device="cuda"):
        print(f"Loading {model_id}...")
        self.device = device

        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True
        ).to(self.device)
        
        self.processor = AutoProcessor.from_pretrained(model_id)
        print("Model loaded.")
    
    def generate(self, image: Image.Image, prompt: str, max_new_tokens: int = 50):
        """
        Standard synchronous generation - does everything sequentially.
        """
        formatted_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"

        inputs = self.processor(
            text=formatted_prompt,
            images=image,
            return_tensors="pt"
        ).to(self.device, torch.float16)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True
            )

        generated_text = self.processor.decode(
            output_ids[0][inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        
        return generated_text

def main():
    engine = BaselineLlava(MODEL_ID)

    print("\nFetching test image...")
    response = requests.get(TEST_IMAGE_URL)
    image = Image.open(BytesIO(response.content))

    requests_list = [
        {"id": "A", "prompt": "Describe this image in detail.", "max_tokens": 100},
        {"id": "B", "prompt": "What colors are in this image?", "max_tokens": 80},
        {"id": "C", "prompt": "Is there a car in this image?", "max_tokens": 60},
        {"id": "D", "prompt": "What is the make and model of the vehicle?", "max_tokens": 100},
        {"id": "E", "prompt": "Describe the background and setting.", "max_tokens": 100},
        {"id": "F", "prompt": "What time of day does this appear to be?", "max_tokens": 80},
        {"id": "G", "prompt": "Are there any people visible in the image?", "max_tokens": 60},
        {"id": "H", "prompt": "What is the condition of the vehicle?", "max_tokens": 100},
        {"id": "I", "prompt": "Describe any text or signage visible.", "max_tokens": 80},
        {"id": "J", "prompt": "What is the weather like in this image?", "max_tokens": 60},
        {"id": "K", "prompt": "Estimate the year or era this photo was taken.", "max_tokens": 100},
        {"id": "L", "prompt": "What can you infer about the location?", "max_tokens": 100},
        {"id": "M", "prompt": "Are there any safety features visible?", "max_tokens": 80},
        {"id": "N", "prompt": "Describe the vehicle's wheels and tires.", "max_tokens": 80},
        {"id": "O", "prompt": "What is the primary subject of this image?", "max_tokens": 60},
    ]
    
    print("\n" + "="*80)
    print("BASELINE INFERENCE - Sequential Processing")
    print("="*80)
    
    results = []
    total_start = time.time()
    
    for req in requests_list:
        print(f"\n[Baseline] Processing Request {req['id']}...")
        start_time = time.time()
        
        generated_text = engine.generate(
            image=image,
            prompt=req['prompt'],
            max_new_tokens=req['max_tokens']
        )
        
        elapsed = time.time() - start_time
        
        results.append({
            'id': req['id'],
            'prompt': req['prompt'],
            'output': generated_text,
            'time': elapsed
        })
        
        print(f"[Baseline] Request {req['id']} completed in {elapsed:.2f}s")
    
    total_time = time.time() - total_start
    
    # # Print results
    # print("\n" + "="*80)
    # print("BASELINE RESULTS")
    # print("="*80)
    print(f"\nTotal Time: {total_time:.2f}s")
    # print(f"Average Time per Request: {total_time/len(requests_list):.2f}s\n")
    
    # for result in results:
    #     print(f"\nRequest {result['id']}:")
    #     print(f"  Prompt: {result['prompt']}")
    #     print(f"  Time: {result['time']:.2f}s")
    #     print(f"  Output: {result['output']}")
    #     print("-" * 80)

if __name__ == "__main__":
    main()