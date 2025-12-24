### **Update Dec-23 on discussed objectives**
- #### **TensorRT setup failure for batched inference**
    - Current setup fails to handle batched requests.
    - Seems the flaw is in the repository's code itself.
    - Sources referred to, in attempt to find solution:
        - https://docs.nvidia.com/dynamo/dev/multimodal/trtllm.html
        - https://github.com/NVIDIA/TensorRT-LLM
    - Found a single Github issue that references a similiar problem
        - https://github.com/NVIDIA/TensorRT-LLM/issues/1227
    - Current setup only works for one batch at a time, i.e. sequential responses.
        - Attempts at introducing batched inference failed.
        - next pathway  is to fuse tensors via cpu in the beginning before passing to engine and using repository's `run` function.
    
- #### **Miscellaneous updates****
    - Clarification: TensorRT does NOT have any PyTorch dependency.
        - The previous statement was caused due to a confusion with Torch-TensorRT, which allows usage of the `compile` function in torch for tensorRT engines.
    - If sufficient GPU memory is not free, TensorRT kills the process with an immediate `OOM` error.
    - Latency time reported was pure inference time, timer begins with invocation of `model.run` and ends at its execution.