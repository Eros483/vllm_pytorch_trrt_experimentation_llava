### **Update Dec-30 on discussed objectives**
- #### **vLLM support for diffusion-based and convolution-based models**
  - vLLM-omni supports diffusion-based models on a limited basis.
    - Very recent framework.
    - Reliant on hugging face's diffusers, and major manual configuration by user.
  
  - For convolution-based models:
    - no vLLM spin-off for the same.
    - `TensorRT` and `ONNX runtime` are the current methods of speeding up convolution based models.

- #### **vLLM architectural decisions for efficiency**
    - Utilizes [Paged Attention](https://medium.com/cj-express-tech-tildi/how-does-vllm-optimize-the-llm-serving-system-d3713009fb73)
    - Utilizes [Continuous Batching and Optimized kernels for hardware](https://www.redhat.com/en/blog/meet-vllm-faster-more-efficient-llm-inference-and-serving)
    - vLLM Engine is quite efficient and fast in terms of initialisation:
        - Creates engine configurations for model and cache.
        - Enables chunked prefill.
        - Profiles token size and available memory,
        - Caches compiled graph from previous runs, for quicker starts. ---> primary reason for why engine spin up was so quick.

- #### **TensorRT not scaling with batch size**
    - Similiar to vLLM, tensorRT's vram allocation is determined as 90% of available vRAM and not actually as something such as batch size, etc.
    - Can be [manually configured](https://github.com/NVIDIA/TensorRT-LLM/issues/401) utilizing fill_template.py while configuring the llm portion of the multimodal setup.
    
    ```
    # configuring tensorRT llm backend
    python3 tensorrt_llm/triton_backend/tools/fill_template.py \
      -i multimodal_ifb/tensorrt_llm/config.pbtxt \
      triton_backend:tensorrtllm,\
    triton_max_batch_size:${MAX_BATCH_SIZE},\
    decoupled_mode:False,\
    max_beam_width:1,\
    engine_dir:${ENGINE_PATH},\
    enable_kv_cache_reuse:False,\
    batching_strategy:inflight_fused_batching,\
    max_queue_delay_microseconds:${MAX_QUEUE_DELAY},\
    enable_chunked_context:False,\
    encoder_input_features_data_type:${ENCODER_INPUT_FEATURES_DTYPE},\
    logits_datatype:TYPE_FP32,\
    prompt_embedding_table_data_type:TYPE_FP16, \
    kv_cache_free_gpu_mem_fraction:0.4
    ```