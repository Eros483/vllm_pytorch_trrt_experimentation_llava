### **Update Dec-27 on discussed objectives**
- #### **Triton Server Batched Results observation**
    - Setup has configurable MAX_BATCH_SIZE for Engine Build, Inference Client server, and Inference Request
        - Assume that for below observations MAX_BATCH_SIZE for engine and client was typically kept the same.
            - Only batch size of inference request was altered.
    
    - Succesfully have tested
        - reliable engine build script @ `scripts/llava_inference_tensor.py`
        - reliable script for engine build and inference requests @ `setup_instructions/llava_inference_tensor.MD`
        - reliable script for batch query execution @ `scripts/test_triton_server.py`

    - Observations
        - With variations of max_batch_size for engine build (ex: 1, 4, 8, 12, 50), max vram consumed in engine build remained same at ~35 gigabytes, taking roughly 90 seconds for dual engine builds.

        - Inference requests for different number of batches yielded negligible increase in VRAM consumption.

        - As number of tokens consumed increased, GPU utilization simply remained at 100% for longer.

        - As observable from logs, never was any actual batching of batch size>9 actually observed.
            - Investigations simply point towards efficiency, and no other cause has been identifiable so far.

        - Investigation was also carried out with inflight batching with a max queue delay of 20ms.
            - Led to no identifiable changes, despite being able to observe the delay actually occuring.

- #### **Future Steps**
    - Requires guidance from lead, no thoughts on how to proceed ahead currently.