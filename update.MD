## **PyTorch LLaVa Model Profiling Documentation #2**
- ### **vRAM consumption as a function of time**
    - PyTorch time measurements during inference yield highly variable results with often ~2000ms worth of shifts for runs with the same parameters.
    - On Average
        - Let us assume inference begins at time 0 ms.
        - Processor load in takes an average of ~4900 ms.
        - Model weight load in takes ~ 2500 ms.
        - Image load in consumes ~460 ms per batch.
        - Input preparation consumes ~5 ms per batch.
        - KV Cache prefill consumes ~140 ms per batch.
        - Decode consumes ~0.35 ms per token per batch.
    - These parameters are roughly accurate to readings as taken in PyTorch.

- ### **Utilization of TensorRT ecosystem for optimizing inference**
    - Utilises seperate engines for components which make up the Llava model.
    - Utilise TensorRT engine for the vision tower.
    - Utilise TensorRT-LLM engine for optimizing decode inference.
    - Wire engines together using `triton` for easy batched inference, and miscellaneous benefits.
    - Shift entire code base to hard drive `/mnt/V_1TB_531` for further usage.

- ### **CUDA Caching Allocator**
    - Utilises two memory pools, one for blocks < 1Mb, another for blocks > 1 Mb
    - Memory pool for the small blocks uses a vector of linkedlists consisting of memory blocks.
    - Memory pool for the larger blocks uses a Balanced tree.
    - Overall, uses best fit strategy.