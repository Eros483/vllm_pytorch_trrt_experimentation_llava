import os
import sys
import subprocess
import argparse
import time
import json
import GPUtil
from datetime import datetime

TENSORRTLLM_BACKEND = "/tensorrtllm_backend"
sys.path.insert(0, f'{TENSORRTLLM_BACKEND}/tensorrt_llm/examples/models/core/multimodal')

from utils import add_common_args
from tensorrt_llm.runtime import MultimodalModelRunner

BUILD_CONFIG = {
    "model_path": "/llava-1.5-7b-hf",
    "engine_dir": "/engines/llava1.5",
    "backend_path": TENSORRTLLM_BACKEND,
    "max_batch_size": 4, 
    "max_input_len": 2048,
    "max_output_len": 512,
    "max_multimodal_len": 2304, #ensure multplying max length for one batch with number of batches
    "dtype": "float16"
}

INFERENCE_CONFIG = {
    "batch_size": 4,  # Process 4 images at once
    "max_new_tokens": 50,
    "images": [
        'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png',
    ],
    "prompts": [
        'Describe this image in detail.',
    ]
}

class LlavaEngineBuilder:
    def __init__(self, config):
        self.config = config
        self.model_path = config["model_path"]
        self.engine_dir = config["engine_dir"]
        self.backend_path = config["backend_path"]
        
        self.llama_dir = f"{self.backend_path}/tensorrt_llm/examples/models/core/llama"
        self.checkpoint_dir = "/tmp/trt_models/llava/fp16/1-gpu"
        self.engine_llm_dir = f"{self.engine_dir}/llm"
        self.engine_vision_dir = f"{self.engine_dir}/vision"
        
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
            
            current_params = {
                "max_batch_size": self.config["max_batch_size"],
                "max_input_len": self.config["max_input_len"],
                "max_output_len": self.config["max_output_len"],
                "max_multimodal_len": self.config["max_multimodal_len"],
                "model_path": self.config["model_path"]
            }
            
            old_params = {k: old_config.get(k) for k in current_params}
            
            if old_params != current_params:
                print("→ Configuration changed, rebuild required")
                print(f"   Old: {old_params}")
                print(f"   New: {current_params}")
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
        save_data = self.config.copy()
        save_data["build_time"] = datetime.now().isoformat()
        
        os.makedirs(self.engine_dir, exist_ok=True)
        with open(f"{self.engine_dir}/build_config.json", 'w') as f:
            json.dump(save_data, f, indent=2)
    
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
            "--dtype", self.config["dtype"]
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
            "--gemm_plugin", self.config["dtype"],
            "--use_fused_mlp", "enable",
            "--max_batch_size", str(self.config["max_batch_size"]),
            "--max_input_len", str(self.config["max_input_len"]),
            "--max_seq_len", str(self.config["max_input_len"] + self.config["max_output_len"]),
            "--max_multimodal_len", str(self.config["max_multimodal_len"])
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


def log_vram(stage, log_file="tensor_rt_metrics.jsonl"):
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


def run_inference(build_cfg, infer_cfg):
    """Run inference on images with prompts"""
    
    all_images = infer_cfg["images"]
    all_prompts = infer_cfg["prompts"]
    batch_size = infer_cfg.get("batch_size", 1)
    max_new_tokens = infer_cfg["max_new_tokens"]

    if batch_size > build_cfg["max_batch_size"]:
        raise ValueError(f"Inference batch size ({batch_size}) > engine max_batch_size ({build_cfg['max_batch_size']})")

    print("Running Inference\n")
    
    if len(all_prompts) == 1 and len(all_images) > 1:
        all_prompts = all_prompts * len(all_images)
    
    parser = argparse.ArgumentParser()
    parser = add_common_args(parser)
    
    args = parser.parse_args([
        '--max_new_tokens', str(max_new_tokens),
        '--hf_model_dir', build_cfg["model_path"],
        '--engine_dir', build_cfg["engine_dir"],
        '--image_path', "", 
        '--input_text', ""
    ])
    
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
    
    total_samples = len(all_images)
    for i in range(0, total_samples, batch_size):
        batch_images = all_images[i : i + batch_size]
        batch_prompts = all_prompts[i : i + batch_size]
        
        print(f"\nProcessing batch {i//batch_size + 1} (Images {i+1}-{min(i+batch_size, total_samples)})")

        args.image_path = batch_images if batch_size > 1 else batch_images[0]
        args.input_text = batch_prompts if batch_size > 1 else batch_prompts[0]

        visual_data = model.load_test_image()
        log_vram(f"batch_{i//batch_size}_loaded")

        start_infer = time.time()
        _, output_text = model.run(
            args.input_text,
            visual_data,
            args.max_new_tokens
        )
        infer_time = (time.time() - start_infer) * 1000
        
        log_vram(f"batch_{i//batch_size}_inferred")

        for j, output in enumerate(output_text):
            response = output[0] if isinstance(output, list) else output
            
            results.append({
                "image": batch_images[j],
                "prompt": batch_prompts[j],
                "response": response,
                "latency_ms": infer_time 
            })
            print(f"  Response [{j}]: {response}")

    print("-" * 50)
    for idx, result in enumerate(results):
        print(f"\n[{idx+1}] {result['image']}")
        print(f"    Prompt: {result['prompt']}")
        print(f"    Response: {result['response']}")
        print(f"    Batch Latency: {result['latency_ms']:.0f}ms")
    
    print(f"\nProcessed {len(results)} image(s)")
    print(f"  Avg batch latency: {sum(r['latency_ms'] for r in results)/len(results):.0f}ms")
    print(f"  Model load time: {load_time:.2f}s")
    
    return results

def run_inference_batched(build_cfg, infer_cfg):
    """Run inference on images with prompts - processes items sequentially"""
    
    all_images = infer_cfg["images"]
    all_prompts = infer_cfg["prompts"]
    batch_size = infer_cfg.get("batch_size", 1)
    max_new_tokens = infer_cfg["max_new_tokens"]

    if batch_size > build_cfg["max_batch_size"]:
        raise ValueError(f"Inference batch size ({batch_size}) > engine max_batch_size ({build_cfg['max_batch_size']})")

    print("Running Inference\n")
    
    if len(all_prompts) == 1 and len(all_images) > 1:
        all_prompts = all_prompts * len(all_images)
    
    parser = argparse.ArgumentParser()
    parser = add_common_args(parser)
    
    args = parser.parse_args([
        '--max_new_tokens', str(max_new_tokens),
        '--hf_model_dir', build_cfg["model_path"],
        '--engine_dir', build_cfg["engine_dir"],
        '--image_path', "", 
        '--input_text', ""
    ])
    
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
    
    total_samples = len(all_images)
    
    # Process each image individually
    for i, (image, prompt) in enumerate(zip(all_images, all_prompts)):
        print(f"\nProcessing image {i+1}/{total_samples}")

        # Pass single values as strings, not lists
        args.image_path = image
        args.input_text = prompt

        visual_data = model.load_test_image()
        log_vram(f"image_{i}_loaded")

        start_infer = time.time()
        _, output_text = model.run(
            args.input_text,
            visual_data,
            args.max_new_tokens
        )
        infer_time = (time.time() - start_infer) * 1000
        
        log_vram(f"image_{i}_inferred")

        # Extract response
        response = output_text[0] if isinstance(output_text, list) else output_text
        
        results.append({
            "image": image,
            "prompt": prompt,
            "response": response,
            "latency_ms": infer_time 
        })
        print(f"  Response: {response}")

    print("-" * 50)
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
    builder = LlavaEngineBuilder(BUILD_CONFIG)
    
    if builder.needs_rebuild():
        print("\nBuilding engines...")
        builder.build()
    else:
        print("\nUsing existing engines")

    results = run_inference_batched(BUILD_CONFIG, INFERENCE_CONFIG)

if __name__ == "__main__":
    main()