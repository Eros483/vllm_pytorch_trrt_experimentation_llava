import subprocess
import time
import asyncio
import argparse
from typing import List, Dict

async def run_client(request_id: int, text: str, image: str, max_tokens: int) -> Dict:
    """Run a single client.py process"""
    cmd = [
        "python3", 
        "tensorrt_llm/triton_backend/tools/multimodal/client.py",
        "--text", text,
        "--image", image,
        "--request-output-len", str(max_tokens),
        "--model_type", "llava"
    ]
    
    start_time = time.time()
    print(f"Starting request {request_id}")
    
    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd="/workspace/tensorrtllm_backend"
        )
        
        stdout, stderr = await process.communicate()
        end_time = time.time()
        latency = end_time - start_time
        
        success = process.returncode == 0
        
        if success:
            print(f"Request {request_id} completed in {latency:.3f}s")
        else:
            print(f"Request {request_id} failed in {latency:.3f}s")
            if stderr:
                print(f"   Error: {stderr.decode('utf-8')[:200]}")
        
        return {
            "request_id": request_id,
            "success": success,
            "latency": latency,
            "start_time": start_time,
            "end_time": end_time,
            "stdout": stdout.decode('utf-8'),
            "stderr": stderr.decode('utf-8'),
            "prompt": text
        }
    except Exception as e:
        end_time = time.time()
        print(f"Request {request_id} error: {e}")
        return {
            "request_id": request_id,
            "success": False,
            "latency": end_time - start_time,
            "error": str(e)
        }


async def run_batch(num_requests: int, prompts: List[str], images: List[str], 
                   max_tokens: int) -> List[Dict]:
    """Run multiple requests concurrently"""
    
    print(f"\n{'='*70}")
    print(f"Starting {num_requests} concurrent requests")
    print(f"{'='*70}\n")
    
    batch_start = time.time()
    
    tasks = []
    for i in range(num_requests):
        prompt = prompts[i % len(prompts)]
        image = images[i % len(images)]
        task = run_client(i, prompt, image, max_tokens)
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    
    batch_end = time.time()
    total_time = batch_end - batch_start
    
    successful = [r for r in results if r.get("success", False)]
    failed = [r for r in results if not r.get("success", False)]
    
    print(f"\n{'='*70}")
    print(f"Batch complete")
    print(f"{'='*70}")
    print(f"Total requests: {num_requests}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Total time: {total_time:.3f}s")
    
    if successful:
        latencies = [r["latency"] for r in successful]

        print(f"  Avg: {sum(latencies)/len(latencies):.3f}s")

        end_times = [r["end_time"] for r in successful]
    
    if failed:
        print(f"\nFailed Requests:")
        for r in failed[:2]:
            print(f"\n  Request {r['request_id']}:")
            if 'stderr' in r and r['stderr']:
                print(f"  Error: {r['stderr'][:300]}")
            if 'error' in r:
                print(f"  Exception: {r['error']}")
    
    print(f"{'='*70}\n")
    
    return results


async def main():
    parser = argparse.ArgumentParser(description="Batch testing using client.py")
    parser.add_argument("--batch-size", type=int, default=4, 
                       help="Number of concurrent requests (default: 4)")
    parser.add_argument("--num-batches", type=int, default=1,
                       help="Number of batches to run (default: 1)")
    parser.add_argument("--max-tokens", type=int, default=50,
                       help="Max tokens per request (default: 50)")
    
    args = parser.parse_args()
    
    prompts = [
        "Describe this image in detail.",
        "What objects can you see?",
        "What is the main subject?",
        "Analyze the composition.",
    ]
    
    images = [
        "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png",
    ] * 10
    
    print(f"Batch Test Configuration:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Number of batches: {args.num_batches}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Config batch size: 4, Engine batch size: 8")
    
    all_results = []
    
    for batch_num in range(args.num_batches):
        if args.num_batches > 1:
            print(f"\n{'#'*70}")
            print(f"BATCH {batch_num + 1}/{args.num_batches}")
            print(f"{'#'*70}")
        
        results = await run_batch(
            args.batch_size,
            prompts,
            images,
            args.max_tokens
        )
        
        all_results.extend(results)
        
        if batch_num < args.num_batches - 1:
            print("Waiting 3s before next batch...\n")
            await asyncio.sleep(3)
    
    # # Sample outputs section (commented out)
    # print("\n" + "="*70)
    # print("SAMPLE OUTPUTS")
    # print("="*70)
    # successful = [r for r in all_results if r.get("success", False)]
    # for result in successful[:2]:
    #     print(f"\nRequest {result['request_id']}:")
    #     print(f"Prompt: {result['prompt']}")
    #     stdout = result['stdout']
    #     if "ASSISTANT:" in stdout:
    #         output = stdout.split("ASSISTANT:")[1].strip().split('\n')[0][:200]
    #         print(f"Output: {output}...")
    #     print(f"Latency: {result['latency']:.3f}s")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
