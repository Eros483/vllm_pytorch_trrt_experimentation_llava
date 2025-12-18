import torch
import asyncio
import time
from typing import List, Dict
from dataclasses import dataclass
import json
import sys
import os
from PIL import Image
import requests
from io import BytesIO
    
from test_tetris_dual_stream import PipelinedLlava, TetrisScheduler, InferenceRequest
from llava_inference import BaselineLlava

@dataclass
class BenchmarkResult:
    method: str
    num_requests: int
    total_time: float
    total_tokens: int
    tokens_per_second: float
    avg_latency: float
    peak_vram_gb: float

class BenchmarkSuite:
    """
    Comprehensive benchmark comparing Tetris vs Baseline
    """
    def __init__(self, image, prompts_pool):
        self.image = image
        self.prompts_pool = prompts_pool
        self.results = []
    
    def generate_requests(self, num_requests, token_range=(50, 150)):
        """Generate N requests with varying token lengths"""
        import random
        requests = []
        for i in range(num_requests):
            prompt = self.prompts_pool[i % len(self.prompts_pool)]
            tokens = random.randint(token_range[0], token_range[1])
            requests.append({
                'id': f"R{i:03d}",
                'prompt': prompt,
                'max_tokens': tokens
            })
        return requests
    
    async def benchmark_tetris(self, num_requests, token_range, engine, staggered=False):
        """Benchmark Tetris implementation"""
        import sys
        import os

        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        
        try:
            scheduler = TetrisScheduler(engine)
            request_configs = self.generate_requests(num_requests, token_range)

            requests = []
            for cfg in request_configs:
                req = InferenceRequest(
                    request_id=cfg['id'],
                    image=self.image,
                    prompt=cfg['prompt'],
                    max_new_tokens=cfg['max_tokens']
                )
                requests.append(req)

            torch.cuda.reset_peak_memory_stats()

            start = time.time()
            
            if staggered:

                for i, req in enumerate(requests):
                    await scheduler.add_request(req)
                    if i < len(requests) - 1:
                        await asyncio.sleep(0.05)
            else:
                for req in requests:
                    await scheduler.add_request(req)
            
            await scheduler.run_loop()
            
            total_time = time.time() - start
        finally:
            sys.stdout.close()
            sys.stdout = old_stdout

        total_tokens = sum(req.tokens_generated for req in requests)
        tokens_per_second = total_tokens / total_time
        avg_latency = total_time / num_requests
        peak_vram = torch.cuda.max_memory_allocated() / 1e9
        
        return BenchmarkResult(
            method="Tetris" + (" (Staggered)" if staggered else ""),
            num_requests=num_requests,
            total_time=total_time,
            total_tokens=total_tokens,
            tokens_per_second=tokens_per_second,
            avg_latency=avg_latency,
            peak_vram_gb=peak_vram
        )
    
    def benchmark_baseline(self, num_requests, token_range, engine):
        """Benchmark baseline implementation"""
        import sys
        import os
        
        # Suppress all output
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        
        try:
            request_configs = self.generate_requests(num_requests, token_range)
            torch.cuda.reset_peak_memory_stats()

            start = time.time()
            
            for cfg in request_configs:
                generated_text = engine.generate(
                    image=self.image,
                    prompt=cfg['prompt'],
                    max_new_tokens=cfg['max_tokens']
                )
            
            total_time = time.time() - start
        finally:
            sys.stdout.close()
            sys.stdout = old_stdout

        total_tokens = sum(cfg['max_tokens'] for cfg in request_configs)
        tokens_per_second = total_tokens / total_time
        avg_latency = total_time / num_requests
        peak_vram = torch.cuda.max_memory_allocated() / 1e9
        
        return BenchmarkResult(
            method="Baseline",
            num_requests=num_requests,
            total_time=total_time,
            total_tokens=total_tokens,
            tokens_per_second=tokens_per_second,
            avg_latency=avg_latency,
            peak_vram_gb=peak_vram
        )
    
    def print_result(self, result: BenchmarkResult):
        """Pretty print a single result"""
        print(f"\n{'='*70}")
        print(f"Method: {result.method}")
        print(f"{'='*70}")
        print(f"Requests:          {result.num_requests}")
        print(f"Total Time:        {result.total_time:.2f}s")
        print(f"Total Tokens:      {result.total_tokens}")
        print(f"Throughput:        {result.tokens_per_second:.2f} tokens/sec")
        print(f"Avg Latency:       {result.avg_latency:.3f}s per request")
        print(f"Peak VRAM:         {result.peak_vram_gb:.2f} GB")
    
    def compare_results(self, baseline: BenchmarkResult, tetris: BenchmarkResult):
        """Compare two results"""
        speedup = baseline.total_time / tetris.total_time
        throughput_gain = (tetris.tokens_per_second - baseline.tokens_per_second) / baseline.tokens_per_second * 100
        
        print(f"\n{'='*70}")
        print(f"COMPARISON")
        print(f"{'='*70}")
        print(f"Speedup:           {speedup:.2f}x {'✓' if speedup > 1.0 else '✗'}")
        print(f"Throughput Gain:   {throughput_gain:+.1f}%")
        print(f"Time Saved:        {baseline.total_time - tetris.total_time:.2f}s")
        print(f"VRAM Overhead:     {tetris.peak_vram_gb - baseline.peak_vram_gb:+.2f} GB")


async def run_comprehensive_benchmark():
    """
    Run complete benchmark suite with multiple configurations
    """
    MODEL_ID = "llava-hf/llava-1.5-7b-hf"
    TEST_IMAGE_URL = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"

    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')

    response = requests.get(TEST_IMAGE_URL)
    image = Image.open(BytesIO(response.content))
    
    sys.stdout.close()
    sys.stdout = old_stdout
    
    prompts = [
        "Describe this image in detail.",
        "What colors are in this image?",
        "Is there a car in this image?",
        "What is the make and model of the vehicle?",
        "Describe the background and setting.",
        "What time of day does this appear to be?",
        "Are there any people visible in the image?",
        "What is the condition of the vehicle?",
        "Describe any text or signage visible.",
        "What is the weather like in this image?",
        "Estimate the year or era this photo was taken.",
        "What can you infer about the location?",
        "Are there any safety features visible?",
        "Describe the vehicle's wheels and tires.",
        "What is the primary subject of this image?",
    ]
    
    suite = BenchmarkSuite(image, prompts)
    
    print("\n" + "="*70)
    print("COMPREHENSIVE BENCHMARK SUITE")
    print("="*70)

    configs = [
        {"name": "Small Load (15 reqs, short)", "num_requests": 15, "token_range": (30, 60)},
        {"name": "Medium Load (30 reqs, medium)", "num_requests": 30, "token_range": (80, 120)},
        # {"name": "Large Load (50 reqs, long)", "num_requests": 50, "token_range": (100, 200)},
    ]
    
    for config in configs:
        print(f"\n{'#'*70}")
        print(f"TEST: {config['name']}")
        print(f"{'#'*70}")

        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        tetris_engine = PipelinedLlava(MODEL_ID)
        sys.stdout.close()
        sys.stdout = old_stdout
        
        tetris_result = await suite.benchmark_tetris(
            config['num_requests'], 
            config['token_range'], 
            tetris_engine,
            staggered=False
        )
        suite.print_result(tetris_result)
        
        tetris_staggered = await suite.benchmark_tetris(
            config['num_requests'], 
            config['token_range'], 
            tetris_engine,
            staggered=True
        )
        suite.print_result(tetris_staggered)

        del tetris_engine
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        baseline_engine = BaselineLlava(MODEL_ID)
        sys.stdout.close()
        sys.stdout = old_stdout
        
        baseline_result = suite.benchmark_baseline(
            config['num_requests'], 
            config['token_range'], 
            baseline_engine
        )
        suite.print_result(baseline_result)

        suite.compare_results(baseline_result, tetris_result)

        del baseline_engine
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    print("\n" + "="*70)
    print("BENCHMARK COMPLETE")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(run_comprehensive_benchmark())