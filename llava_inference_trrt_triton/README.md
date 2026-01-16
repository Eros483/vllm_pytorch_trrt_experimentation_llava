# Llava Inference via TensorRT and Triton
Refer to `docs/setup_tensor_triton.md` for instructions on how to setup the repository for all necessary functionalities.

## Overview of the Repository
The repository handles and provides instructions alongisde scritps to build TensorRT and tensorRT-LLM engines for Llava-1.5 7B, alongside Triton Inference server.
- Shifts approach from PyTorch provided standard continuous batching, to inflight batching. 
- Reduces inference latency by 15x from 30ms per request, to 2ms per request (tested on a Nvidia L40s).
- Increases per-GPU request concurrency and throughput, potentially enabling 8x clients to be served per GPU.
- Removes PyTorch Caching allocator induced cap on batch requests of size 62 (as on a Nvidia L40s).

## Repository setup
```
git clone https://github.com/Eros483/Llava_Inference_TensorRT_Triton.git
cd Llava_Inference_TensorRT_Triton
```
- Kindly ensure you have `Docker` setup and running.
- Navigate to `docs/setup_tensor_triton` for further steps.
---
### **Caution**
- The method used in this repository utilizes the  `multimodal encoders` directory from the `tensorrtllm_backend` repository.
  - This has been depreciated and is no longer updated.
  - The provided solution utilizes a forked, and modified version of the same repository i.e `Eros483/tensorrtllm_backend.git`
- It would be advisable to instead follow the currently supported stack i.e
```
Triton PyTorch backend
+ torch.compile
+ TensorRT-LLM via Dynamo
```
