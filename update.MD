## **Update Dec-31 on discussed objectives**
- ### **vLLM scheduler**
  - Offers two scheduling options (source code is available):
    - First come first served.
    - Priority based queueing.
  
  - In a priority based queuing system, if requests have the same priority, then whichever request came first is received, if it can be processed.
    - Even if requests are sent at the same time in a batch, the order in which they're sequenced gives them an ordering at a  lower level, which is utilised.

- ### **ONNX Constraints  for inference speedup**
  - Primarily investigating conversion of Llava models and Stable diffusion.
  - Supports convolution based models entirely.
  - Diffusion models are a blend of python code and ONNX optimization.
    - ONNX individually requires exporting each component invidividually, providing optimized versions.
    - Post that, they still need to run via python in loops and the denoising scheduler logic stays in python as well.
      - TensorRT works similiarly as well, so component level optimizations seem to be the standard.
  
  - While ONNX does support all GPU architectures, the support for Nvidia is the best, followed closely by Apple's CoreML provider. AMD is supported via rocM but is less mature and optimizations vary greatly with GPU generation.