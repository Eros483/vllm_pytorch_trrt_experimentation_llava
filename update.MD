### **Update Dec-22 on discussed objectives**
- #### **TensorRT setup**
    - Built single run application for
        - Accessible Llava configuration setup
        - loggable metrics, primarily
            - vram consumption
            - inference time
        - Handles engine build with
            - TensorRT for vision tower
            - TensorRT-LLM for LLM Decoder
    
    - Base run observations:
        - Higher vRAM consumption.
        - Significantly lower inference time.

    - Possible reason:
        - TensorRT allocates at the beginning, amount of vram necessary in worst-case situation for engine parameter specifications.
            - Thus vRAM is preallocated for tensors that haven't even been created yet.
            - primarily, need to identify use-case parameters, and appropriately initialise engine.

        - PyTorch allocates and caches memory overtime, as tensors are brought into memory and removed.

- #### **TensorRT architectural choices for efficiency over PyTorch**
    - PyTorch executes operations line-by-line, introducing latency as GPU waits for CPU instructions on what to do, repeatedly. This forces sequential computational graph build.
        - TensorRT precompiles model as a engine, allowing entire graph to be visible, alongside pre-allocating memory.
    
    - TensorRT applies kernel Auto-Tuning, for example, to the matrix multiplication algorith, tailored to GPU being used.

    - TensorRT fuses layers together.
        - Example: Convolution + ReLu is 2 seperate calls for pyTorch, but single call for TensorRT

    - Utilises In-flight batching:
        - Example: We send batches of 4 images
            - PyTorch moves to next batch, only after all 4 images in current batch finished processing.
            - TensorRT ejects requests as soon as processing has finished, and brings in new request mid-generation.

    - TensorRT utilises Paged KV Cache.
        - PyTorch uses a single continguous block of memory for KV cache.
        - TensorRT splits it into pages, allowing usage of smaller pre-free blocks.

- Note: TensorRT and TensorRT-LLM do not utilise Nvidia RT cores, i.e ray-tracing cores.