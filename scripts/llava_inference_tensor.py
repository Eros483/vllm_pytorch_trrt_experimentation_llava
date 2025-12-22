import os
import sys
import subprocess
import argparse
import time
import json
import GPUtil
from datetime import datetime
from utils import add_common_args
from tensorrt_llm.runtime import MultimodalModelRunner

BATCH_SIZE = 1
MAX_INPUT_LEN = 2048
MAX_OUTPUT_LEN = 512
MAX_NEW_TOKENS = 50

IMAGES = [
    'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png'
]

PROMPTS = [
    'Describe this image in detail.'
]

MODEL_PATH = "/llava-1.5-7b-hf"
ENGINE_DIR = "/engines/llava1.5"
TENSORRTLLM_BACKEND = "/tensorrtllm_backend"


sys.path.insert(0, f'{TENSORRTLLM_BACKEND}/tensorrt_llm/examples/models/core/multimodal')

class LlavaEngineBuilder:
    def __init__(self, model_path, engine_dir, backend_path, batch_size, max_input_len, max_output_len):
        self.model_path = model_path
        self.engine_dir = engine_dir
        self.backend_path = backend_path
        self.batch_size = batch_size
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        
        self.llama_dir = f"{backend_path}/tensorrt_llm/examples/models/core/llama"
        self.checkpoint_dir = "/tmp/trt_models/llava/fp16/1-gpu"
        self.engine_llm_dir = f"{engine_dir}/llm"
        self.engine_vision_dir = f"{engine_dir}/vision"
        
    def needs_rebuild(self):
        """
        Checks JSON config to see if rebuild is needed.
        """
        config_file = f"{self.engine_dir}/build_config.json"
        
        if not os.path.exists(f"{self.engine_llm_dir}/config.json"):
            print("LLM engine not found, build required\n")
            return True
        if not os.path.exists(f"{self.engine_vision_dir}"):
            print("Vision engine not found, build required\n")
            return True

        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                old_config = json.load(f)
            
            current_config = {
                "batch_size": self.batch_size,
                "max_input_len": self.max_input_len,
                "max_output_len": self.max_output_len,
                "model_path": self.model_path
            }
            
            if old_config != current_config:
                print("→ Configuration changed, rebuild required")
                print(f"   Old: {old_config}")
                print(f"   New: {current_config}")
                return True
        else:
            print("No build config found, build required\n")
            return True
        
        print("Existing engines match current configuration, inferring from pre-built engines.\n")
        return False
    
    def save_config(self):
        """
        Save build configuration
        """
        config = {
            "batch_size": self.batch_size,
            "max_input_len": self.max_input_len,
            "max_output_len": self.max_output_len,
            "model_path": self.model_path,
            "build_time": datetime.now().isoformat()
        }
        
        os.makedirs(self.engine_dir, exist_ok=True)
        with open(f"{self.engine_dir}/build_config.json", 'w') as f:
            json.dump(config, f, indent=2)
    
    def validate_setup(self):
        """
        Helper function to validate build setup.
        """
        print("VALIDATING BUILD SETUP\n")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        if not os.path.exists(self.llama_dir):
            raise FileNotFoundError(f"Llama dir not found: {self.llama_dir}")
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.engine_llm_dir, exist_ok=True)
        os.makedirs(self.engine_vision_dir, exist_ok=True)
        print("SETUP VALIDATED\n")
    
    def convert_checkpoint(self):
        """
        Convert HF checkpoint to TRT-LLM format
        """
        print("Converting HuggingFace Checkpoint\n")
        
        convert_cmd = [
            "python3",
            f"{self.llama_dir}/convert_checkpoint.py",
            "--model_dir", self.model_path,
            "--output_dir", self.checkpoint_dir,
            "--dtype", "float16"
        ]

        subprocess.run(convert_cmd, check=True)
        print("Checkpoint conversion complete\n")
    
    def build_llm_engine(self):
        """
        Build TRT-LLM engine for the LLM part
        """
        print("Building LLM Engine\n")
        
        build_cmd = [
            "trtllm-build",
            "--checkpoint_dir", self.checkpoint_dir,
            "--output_dir", self.engine_llm_dir,
            "--gemm_plugin", "float16",
            "--use_fused_mlp", "enable",
            "--max_batch_size", str(self.batch_size),
            "--max_input_len", str(self.max_input_len),
            "--max_seq_len", str(self.max_input_len + self.max_output_len),
            "--max_multimodal_len", "576"
        ]
        
        subprocess.run(build_cmd, check=True)
        print("LLM engine build complete\n")
    
    def build_visual_engine(self):
        """
        Build TensorRT engine for visual components
        """
        print("Building Visual Engine\n")
        
        build_script = f"""
import argparse
from tensorrt_llm.tools.multimodal_builder import build_llava_engine, add_multimodal_arguments

parser = argparse.ArgumentParser()
parser = add_multimodal_arguments(parser)

args = parser.parse_args([
    '--model_path', '{self.model_path}',
    '--output_dir', '{self.engine_vision_dir}',
    '--model_type', 'llava'
])

args.device = 'cuda'

print("Building LLaVA visual engine...")
build_llava_engine(args)
print("Visual engine build complete!")
"""
        subprocess.run(["python3", "-c", build_script], check=True, text=True)
        print("Visual engine build complete\n")
    
    def build(self):
        """
        Execute build pipeline
        """
        try:
            self.validate_setup()
            self.convert_checkpoint()
            self.build_llm_engine()
            self.build_visual_engine()
            self.save_config()

            print("BUILD SUCCESSFUL")
            print(f"Engines ready at: {self.engine_dir}")
            
        except Exception as e:
            print(f"\nBuild failed: {e}\n")
            raise


def log_vram(stage, log_file="../metrics/tensor_rt_metrics.jsonl"):
    """helper to Log VRAM usage"""
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            metric = {
                "timestamp": datetime.now().isoformat(),
                "stage": stage,
                "vram_used_mb": gpu.memoryUsed,
                "vram_total_mb": gpu.memoryTotal,
                "gpu_load": round(gpu.load * 100, 1),
                "gpu_temp": gpu.temperature
            }
            
            with open(log_file, "a") as f:
                f.write(json.dumps(metric) + "\n")
            
            print(f"[{stage}] VRAM: {gpu.memoryUsed:.0f}MB / {gpu.memoryTotal:.0f}MB ({gpu.memoryUtil*100:.1f}%)")
            return metric
    except:
        pass
    return None


def run_inference(images, prompts, max_new_tokens):
    """Run inference on images with prompts"""

    print("Running Inference\n")
    
    # Ensure prompts match images
    if len(prompts) == 1 and len(images) > 1:
        prompts = prompts * len(images)
    
    parser = argparse.ArgumentParser()
    parser = add_common_args(parser)
    
    args = parser.parse_args([
        '--max_new_tokens', str(max_new_tokens),
        '--hf_model_dir', MODEL_PATH,
        '--engine_dir', ENGINE_DIR,
        '--image_path', images[0],
        '--input_text', prompts[0]
    ])

    if isinstance(args.input_text, list):
        args.input_text = ' '.join(args.input_text)
    if isinstance(args.image_path, list):
        args.image_path = args.image_path[0]
    
    args.visual_engine_dir = os.path.join(args.engine_dir, 'vision')
    args.llm_engine_dir = os.path.join(args.engine_dir, 'llm')
    args.use_py_session = (args.session == 'python')
    args.use_cpp_session = (args.session == 'cpp')
    
    log_vram("baseline")

    print("LOADING MODEL\n")
    start_load = time.time()
    model = MultimodalModelRunner(args)
    load_time = time.time() - start_load
    print(f"✓ Model loaded in {load_time:.2f}s")
    log_vram("model_loaded")
    
    results = []
    for idx, (image_path, prompt) in enumerate(zip(images, prompts)):
        print(f"\nImage: {image_path}")
        print(f"Prompt: {prompt}\n")

        args.image_path = image_path
        args.input_text = prompt

        visual_data = model.load_test_image()
        log_vram(f"image_{idx}_loaded")

        start_infer = time.time()
        input_text, output_text = model.run(
            args.input_text,
            visual_data,
            args.max_new_tokens
        )
        infer_time = (time.time() - start_infer) * 1000
        
        log_vram(f"image_{idx}_inferred")

        if isinstance(output_text, list):
            response = output_text[0][0] if isinstance(output_text[0], (list, tuple)) else output_text[0]
        else:
            response = output_text
        
        results.append({
            "image": image_path,
            "prompt": prompt,
            "response": response,
            "latency_ms": infer_time
        })
        
        print(f"\nResponse: {response}")
        print(f"Latency: {infer_time:.0f}ms")

    for idx, result in enumerate(results):
        print(f"\n[{idx+1}] {result['image']}")
        print(f"    Prompt: {result['prompt']}")
        print(f"    Response: {result['response']}")
        print(f"    Latency: {result['latency_ms']:.0f}ms")
    
    print(f"\nProcessed {len(results)} image(s)")
    print(f"  Avg latency: {sum(r['latency_ms'] for r in results)/len(results):.0f}ms")
    print(f"  Model load time: {load_time:.2f}s")
    
    return results


def main():
    builder = LlavaEngineBuilder(
        MODEL_PATH, 
        ENGINE_DIR, 
        TENSORRTLLM_BACKEND,
        BATCH_SIZE,
        MAX_INPUT_LEN,
        MAX_OUTPUT_LEN
    )
    
    if builder.needs_rebuild():
        print("\nBuilding engines...")
        builder.build()
    else:
        print("\nUsing existing engines")

    results = run_inference(IMAGES, PROMPTS, MAX_NEW_TOKENS)

if __name__ == "__main__":
    main()