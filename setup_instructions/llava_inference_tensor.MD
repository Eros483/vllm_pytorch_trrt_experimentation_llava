### **Instructions to set up TensorRT inference for Llava**
This guide assumes this is the first instance of engine setup for the relevant directory. 
#### **Downloading relevant repositories into base**
```
sudo apt-get update && sudo apt-get install git-lfs -y
git lfs install

git clone https://github.com/triton-inference-server/tensorrtllm_backend.git
cd tensorrtllm_backend
git submodule update --init --recursive
cd ..

git clone https://huggingface.co/llava-hf/llava-1.5-7b-hf

mkdir -p engines
```
- It is assumed that the relevant scripts are already present in `/scripts`

#### **Launching Triton Container**
Make sure to be present in base directory.
```
docker run --rm -it --net host --shm-size=2g \
    --ulimit memlock=-1 --ulimit stack=67108864 --gpus all \
    -v $(pwd)/tensorrtllm_backend:/tensorrtllm_backend \
    -v $(pwd)/llava-1.5-7b-hf:/llava-1.5-7b-hf \
    -v $(pwd)/engines:/engines \
    -v $(pwd)/scripts:/workspace \
    nvcr.io/nvidia/tritonserver:24.12-trtllm-python-py3

# from inside the container image
pip install psutil gputil tritonclient[http] pillow requests
```

#### **Engine build and inference**
- The script handles engine setup, build (if needed) and inference.
- Modify the first few lines to control parameters such as batch size, images, prompts, tokens generated, etc.

```
python3 llava_inference_tensor.py
```
- metrics for inference will be saved as `/scripts/inference_metrics.jsonl`
