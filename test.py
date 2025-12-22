#!/usr/bin/env python3
"""
Llava1.5-7B Triton Deployment Script with Full Observability
Monitors VRAM usage, performance metrics, and provides detailed logging
"""

import os
import sys
import subprocess
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
import signal

try:
    import psutil
    import GPUtil
except ImportError:
    print("Installing required dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil", "gputil"])
    import psutil
    import GPUtil


class ResourceMonitor:
    """Monitors GPU and system resources"""
    
    def __init__(self, log_file="resource_metrics.jsonl"):
        self.log_file = log_file
        self.start_time = time.time()
        self.baseline_vram = None
        
    def get_gpu_stats(self):
        """Get current GPU statistics"""
        try:
            gpus = GPUtil.getGPUs()
            stats = []
            for gpu in gpus:
                stats.append({
                    "id": gpu.id,
                    "name": gpu.name,
                    "load": f"{gpu.load * 100:.1f}%",
                    "memory_used_mb": gpu.memoryUsed,
                    "memory_total_mb": gpu.memoryTotal,
                    "memory_util": f"{gpu.memoryUtil * 100:.1f}%",
                    "temperature": f"{gpu.temperature}°C"
                })
            return stats
        except Exception as e:
            return [{"error": str(e)}]
    
    def get_system_stats(self):
        """Get system CPU and RAM statistics"""
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "ram_used_gb": psutil.virtual_memory().used / (1024**3),
            "ram_total_gb": psutil.virtual_memory().total / (1024**3),
            "ram_percent": psutil.virtual_memory().percent
        }
    
    def log_metrics(self, stage, extra_info=None):
        """Log metrics to file and console"""
        timestamp = datetime.now().isoformat()
        elapsed = time.time() - self.start_time
        
        metrics = {
            "timestamp": timestamp,
            "elapsed_seconds": round(elapsed, 2),
            "stage": stage,
            "gpu_stats": self.get_gpu_stats(),
            "system_stats": self.get_system_stats()
        }
        
        if extra_info:
            metrics["extra"] = extra_info
        
        # Calculate VRAM usage comparison
        if self.baseline_vram is None and stage == "baseline":
            self.baseline_vram = metrics["gpu_stats"][0]["memory_used_mb"]
        elif self.baseline_vram is not None:
            current_vram = metrics["gpu_stats"][0]["memory_used_mb"]
            metrics["vram_delta_mb"] = current_vram - self.baseline_vram
        
        # Write to log file
        with open(self.log_file, "a") as f:
            f.write(json.dumps(metrics) + "\n")
        
        # Print to console
        self._print_metrics(metrics)
        
        return metrics
    
    def _print_metrics(self, metrics):
        """Pretty print metrics to console"""
        print(f"\n{'='*70}")
        print(f"Stage: {metrics['stage']}")
        print(f"Time Elapsed: {metrics['elapsed_seconds']}s")
        print(f"{'='*70}")
        
        for gpu in metrics["gpu_stats"]:
            if "error" not in gpu:
                print(f"\nGPU {gpu['id']}: {gpu['name']}")
                print(f"  Memory: {gpu['memory_used_mb']:.0f}MB / {gpu['memory_total_mb']:.0f}MB ({gpu['memory_util']})")
                print(f"  Load: {gpu['load']}")
                print(f"  Temperature: {gpu['temperature']}")
        
        if "vram_delta_mb" in metrics:
            delta = metrics["vram_delta_mb"]
            symbol = "+" if delta > 0 else ""
            print(f"\n  VRAM Change from Baseline: {symbol}{delta:.0f}MB")
        
        sys_stats = metrics["system_stats"]
        print(f"\nSystem:")
        print(f"  CPU: {sys_stats['cpu_percent']:.1f}%")
        print(f"  RAM: {sys_stats['ram_used_gb']:.1f}GB / {sys_stats['ram_total_gb']:.1f}GB ({sys_stats['ram_percent']:.1f}%)")
        
        if metrics.get("extra"):
            print(f"\nAdditional Info: {json.dumps(metrics['extra'], indent=2)}")
        print(f"{'='*70}\n")
    
    def print_summary(self):
        """Print summary of all metrics"""
        if not os.path.exists(self.log_file):
            print("No metrics logged yet")
            return
        
        with open(self.log_file, "r") as f:
            metrics = [json.loads(line) for line in f]
        
        print(f"\n{'#'*70}")
        print(f"DEPLOYMENT SUMMARY")
        print(f"{'#'*70}\n")
        
        stages = {}
        for m in metrics:
            stage = m["stage"]
            if stage not in stages:
                stages[stage] = []
            stages[stage].append(m)
        
        print("VRAM Usage by Stage:")
        print("-" * 70)
        for stage, stage_metrics in stages.items():
            avg_vram = sum(m["gpu_stats"][0]["memory_used_mb"] for m in stage_metrics) / len(stage_metrics)
            print(f"  {stage:20s}: {avg_vram:.0f}MB")
        
        if len(metrics) > 1:
            baseline = metrics[0]["gpu_stats"][0]["memory_used_mb"]
            peak = max(m["gpu_stats"][0]["memory_used_mb"] for m in metrics)
            current = metrics[-1]["gpu_stats"][0]["memory_used_mb"]
            
            print(f"\n{'='*70}")
            print(f"Baseline VRAM: {baseline:.0f}MB")
            print(f"Peak VRAM: {peak:.0f}MB (+{peak-baseline:.0f}MB)")
            print(f"Current VRAM: {current:.0f}MB (+{current-baseline:.0f}MB)")
            print(f"{'='*70}\n")


class LlavaDeployment:
    """Manages Llava1.5 deployment with full observability"""
    
    def __init__(self, args):
        self.args = args
        self.monitor = ResourceMonitor(args.log_file)
        self.triton_process = None
        
    def validate_paths(self):
        """Validate all required paths exist"""
        print("Validating paths...")
        
        required = {
            "TensorRT-LLM Backend": self.args.tensorrtllm_backend,
            "Llava Model": self.args.model_path,
        }
        
        for name, path in required.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"{name} not found at: {path}")
        
        # Create engines directory if needed
        os.makedirs(self.args.engine_dir, exist_ok=True)
        
        print("✓ All paths validated")
    
    def build_engines(self):
        """Build TensorRT-LLM engines with monitoring"""
        if self.args.skip_build:
            print("Skipping engine build (--skip-build flag set)")
            return
        
        print("\n" + "="*70)
        print("BUILDING ENGINES")
        print("="*70)
        
        self.monitor.log_metrics("baseline", {"note": "Before engine build"})
        
        # Step 1: Convert checkpoint
        print("\nStep 1: Converting HuggingFace checkpoint...")
        convert_cmd = [
            "python3",
            f"{self.args.tensorrtllm_backend}/tensorrt_llm/examples/llama/convert_checkpoint.py",
            "--model_dir", self.args.model_path,
            "--output_dir", "/tmp/ckpt/llava/7b/",
            "--dtype", "float16"
        ]
        
        self._run_command(convert_cmd, "checkpoint_conversion")
        self.monitor.log_metrics("after_checkpoint_conversion")
        
        # Step 2: Build LLM engine
        print("\nStep 2: Building TensorRT-LLM engine...")
        build_cmd = [
            "trtllm-build",
            "--checkpoint_dir", "/tmp/ckpt/llava/7b/",
            "--output_dir", self.args.engine_dir,
            "--gemm_plugin", "float16",
            "--use_fused_mlp",
            "--max_batch_size", str(self.args.max_batch_size),
            "--max_input_len", str(self.args.max_input_len),
            "--max_output_len", str(self.args.max_output_len),
            "--max_multimodal_len", "576"
        ]
        
        self._run_command(build_cmd, "llm_engine_build")
        self.monitor.log_metrics("after_llm_engine_build")
        
        # Step 3: Build visual engine
        print("\nStep 3: Building visual engine...")
        visual_cmd = [
            "python",
            f"{self.args.tensorrtllm_backend}/tensorrt_llm/examples/multimodal/build_visual_engine.py",
            "--model_path", self.args.model_path,
            "--model_type", "llava",
            "--output_dir", self.args.engine_dir
        ]
        
        self._run_command(visual_cmd, "visual_engine_build")
        self.monitor.log_metrics("after_visual_engine_build")
        
        print("\n✓ Engine build complete!")
    
    def test_engine(self):
        """Test the built engines"""
        if self.args.skip_test:
            print("Skipping engine test (--skip-test flag set)")
            return
        
        print("\n" + "="*70)
        print("TESTING ENGINES")
        print("="*70)
        
        test_cmd = [
            "python3",
            f"{self.args.tensorrtllm_backend}/tensorrt_llm/examples/multimodal/run.py",
            "--max_new_tokens", "30",
            "--hf_model_dir", self.args.model_path,
            "--visual_engine_dir", self.args.engine_dir,
            "--llm_engine_dir", self.args.engine_dir,
            "--decoder_llm",
            "--input_text", "Question: which city is this? Answer:"
        ]
        
        self._run_command(test_cmd, "engine_test")
        self.monitor.log_metrics("after_engine_test")
    
    def setup_triton_repo(self):
        """Set up Triton model repository"""
        print("\n" + "="*70)
        print("SETTING UP TRITON MODEL REPOSITORY")
        print("="*70)
        
        fill_template_cmd = [
            "python3",
            f"{self.args.tensorrtllm_backend}/tools/fill_template.py",
            "-i", f"{self.args.model_repo}/tensorrt_llm/config.pbtxt",
            f"engine_dir:{self.args.engine_dir}"
        ]
        
        self._run_command(fill_template_cmd, "triton_setup")
        self.monitor.log_metrics("after_triton_setup")
    
    def launch_triton(self):
        """Launch Triton server with monitoring"""
        print("\n" + "="*70)
        print("LAUNCHING TRITON SERVER")
        print("="*70)
        
        # Set environment variables
        os.environ["TRT_ENGINE_LOCATION"] = f"{self.args.engine_dir}/visual_encoder.engine"
        os.environ["HF_LOCATION"] = self.args.model_path
        
        launch_cmd = [
            "python3",
            f"{self.args.tensorrtllm_backend}/scripts/launch_triton_server.py",
            "--world_size", str(self.args.world_size),
            "--model_repo", self.args.model_repo
        ]
        
        print(f"Starting Triton Server...")
        print(f"Command: {' '.join(launch_cmd)}")
        
        self.triton_process = subprocess.Popen(
            launch_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        # Wait for server to start and monitor
        print("\nWaiting for Triton to start (monitoring VRAM)...")
        start_wait = time.time()
        server_ready = False
        
        while time.time() - start_wait < 300:  # 5 minute timeout
            self.monitor.log_metrics("triton_starting")
            
            # Check if server is ready
            try:
                result = subprocess.run(
                    ["curl", "-s", "localhost:8000/v2/health/ready"],
                    capture_output=True,
                    timeout=2
                )
                if result.returncode == 0:
                    server_ready = True
                    break
            except:
                pass
            
            time.sleep(5)
        
        if server_ready:
            print("\n✓ Triton Server is ready!")
            self.monitor.log_metrics("triton_ready")
        else:
            print("\n✗ Triton Server failed to start within timeout")
            return False
        
        return True
    
    def run_inference_tests(self):
        """Run inference tests and measure performance"""
        if self.args.skip_inference:
            print("Skipping inference tests (--skip-inference flag set)")
            return
        
        print("\n" + "="*70)
        print("RUNNING INFERENCE TESTS")
        print("="*70)
        
        tests = [
            {
                "name": "Simple description",
                "prompt": "Describe the picture.",
                "image": "http://images.cocodataset.org/test2017/000000155781.jpg",
                "max_tokens": 15
            },
            {
                "name": "Detailed description",
                "prompt": "Describe this image in detail.",
                "image": "http://images.cocodataset.org/test2017/000000155781.jpg",
                "max_tokens": 50
            }
        ]
        
        for i, test in enumerate(tests, 1):
            print(f"\nTest {i}/{len(tests)}: {test['name']}")
            
            start = time.time()
            
            # Create client script inline
            client_code = f'''
import tritonclient.http as httpclient
import numpy as np
from PIL import Image
import requests
from io import BytesIO

client = httpclient.InferenceServerClient(url="localhost:8000")

# Load and prepare image
response = requests.get("{test['image']}")
img = Image.open(BytesIO(response.content))

# Prepare inputs
prompt_input = httpclient.InferInput("prompt", [1], "BYTES")
prompt_input.set_data_from_numpy(np.array([f"USER: <image>\\n{test['prompt']}"], dtype=object))

image_input = httpclient.InferInput("image", [1], "BYTES") 
image_input.set_data_from_numpy(np.array(["{test['image']}"], dtype=object))

max_tokens_input = httpclient.InferInput("max_tokens", [1], "INT32")
max_tokens_input.set_data_from_numpy(np.array([{test['max_tokens']}], dtype=np.int32))

# Run inference
result = client.infer("llava-1.5", [prompt_input, image_input, max_tokens_input])
output = result.as_numpy("text_output")
print(output[0].decode() if isinstance(output[0], bytes) else output[0])
'''
            
            try:
                result = subprocess.run(
                    ["python3", "-c", client_code],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                latency = time.time() - start
                
                self.monitor.log_metrics(
                    f"inference_test_{i}",
                    {
                        "test_name": test['name'],
                        "latency_seconds": round(latency, 3),
                        "output": result.stdout.strip() if result.returncode == 0 else "ERROR",
                        "success": result.returncode == 0
                    }
                )
                
                if result.returncode == 0:
                    print(f"✓ Success (latency: {latency:.2f}s)")
                    print(f"Output: {result.stdout.strip()}")
                else:
                    print(f"✗ Failed: {result.stderr}")
                    
            except Exception as e:
                print(f"✗ Error: {str(e)}")
    
    def _run_command(self, cmd, stage_name):
        """Run a command and handle errors"""
        print(f"Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            print("✓ Success")
            return result
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed with exit code {e.returncode}")
            print(f"Error output:\n{e.stderr}")
            raise
    
    def cleanup(self):
        """Clean up resources"""
        print("\nCleaning up...")
        
        if self.triton_process:
            print("Stopping Triton Server...")
            self.triton_process.terminate()
            try:
                self.triton_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.triton_process.kill()
        
        # Kill any remaining triton processes
        subprocess.run(["pkill", "tritonserver"], stderr=subprocess.DEVNULL)
        
        self.monitor.log_metrics("cleanup_complete")
        self.monitor.print_summary()
        
        print(f"\n✓ Metrics saved to: {self.args.log_file}")
    
    def run(self):
        """Run the full deployment pipeline"""
        try:
            self.validate_paths()
            self.build_engines()
            self.test_engine()
            self.setup_triton_repo()
            
            if self.launch_triton():
                self.run_inference_tests()
                
                if not self.args.no_interactive:
                    print("\nTriton Server is running. Press Ctrl+C to stop...")
                    try:
                        while True:
                            time.sleep(10)
                            self.monitor.log_metrics("running")
                    except KeyboardInterrupt:
                        print("\nShutdown requested...")
            
        except Exception as e:
            print(f"\n✗ Error: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()


def main():
    parser = argparse.ArgumentParser(
        description="Deploy Llava1.5-7B with Triton and full observability"
    )
    
    # Required paths
    parser.add_argument(
        "--tensorrtllm-backend",
        default="/tensorrtllm_backend",
        help="Path to tensorrtllm_backend repository"
    )
    parser.add_argument(
        "--model-path",
        default="/llava-1.5-7b-hf",
        help="Path to Llava1.5 HuggingFace model"
    )
    parser.add_argument(
        "--engine-dir",
        default="/engines/llava1.5",
        help="Directory to store/load engines"
    )
    parser.add_argument(
        "--model-repo",
        default="/tutorials/Popular_Models_Guide/Llava1.5/model_repository",
        help="Path to Triton model repository"
    )
    
    # Engine build parameters
    parser.add_argument("--max-batch-size", type=int, default=1)
    parser.add_argument("--max-input-len", type=int, default=2048)
    parser.add_argument("--max-output-len", type=int, default=512)
    parser.add_argument("--world-size", type=int, default=1)
    
    # Control flags
    parser.add_argument("--skip-build", action="store_true", help="Skip engine building")
    parser.add_argument("--skip-test", action="store_true", help="Skip engine testing")
    parser.add_argument("--skip-inference", action="store_true", help="Skip inference tests")
    parser.add_argument("--no-interactive", action="store_true", help="Exit after tests")
    
    # Logging
    parser.add_argument(
        "--log-file",
        default=f"llava_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl",
        help="Metrics log file"
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("Llava1.5-7B Triton Deployment with Observability")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Model Path: {args.model_path}")
    print(f"  Engine Dir: {args.engine_dir}")
    print(f"  Log File: {args.log_file}")
    print(f"  Max Batch Size: {args.max_batch_size}")
    print(f"  Skip Build: {args.skip_build}")
    print("="*70 + "\n")
    
    deployment = LlavaDeployment(args)
    deployment.run()


if __name__ == "__main__":\
    main()