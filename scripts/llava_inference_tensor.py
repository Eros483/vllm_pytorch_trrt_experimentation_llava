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
    "max_batch_size":50, 
    "max_input_len": 2048,
    "max_output_len": 512,
    "max_multimodal_len": 28800, #ensure multplying max length for one batch with number of batches i.e 576 * batch size
    "dtype": "float16"
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

def main():
    builder = LlavaEngineBuilder(BUILD_CONFIG)
    
    if builder.needs_rebuild():
        print("\nBuilding engines...")
        builder.build()
    else:
        print("\nUsing existing engines")

if __name__ == "__main__":
    main()