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
    - For MultiModal atleast, there is a PyTorch backend being used by TensorRT and TensorRT-LLM.
    - For any generic model, conversion to tensorRT exists by finding base model in hugging face configuration.
        - Vast support for over so far all thinkable models.
    - Same docker image can be used to run multiple TensorRT engines.