### **Update Dec-29 on discussed objectives**
- #### **vLLM Inference pipeline**
    - Tested vLLM inference pipeline, kindly refer to script @ `notebooks/vllm_inference.ipynb`.
    - Pro's vs PyTorch
        - Constant memory usage, easily predictable.
        - PyTorch was able to handle a maximum of 62 batched requests vs vLLM having handled a batch of 1000 requests so far, without any issues, with plenty of vRAM available.
        - supports out-of-the-box InFlight batching, mimicking production-grade required form of batching.
    
    - Pro's vs TensorRT
        - Signfiicantly easier process of usage and inference.
        - Allows for much faster prototyping, testing and benchmarking inference latency and vRAM usage.
    
- #### **Future Steps**
    - Test inflight batching as provided by vLLM.
    - study techniques used by vLLM for providing efficient inference.