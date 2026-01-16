# **Instructions to set up TensorRT inference for Llava**
## **Repository Structure**
```
llava_inference_trrt
├── docs
│   └── setup_tensor_triton.md
├── engines   
├── llava-1.5-7b-hf   
├── scripts
│   ├── llava_engine_build.py
│   └── test_triton_server.py
└── tensorrtllm_backend
    ├── LICENSE
    ├── README.md
    ├── dockerfile
    ├── docs
    ├── images
    ├── multimodal_ifb
    ├── scripts
    └── tensorrt_llm
```
Carry out all CLI instructions from base directory.

---
## **Setup Swap file for extra RAM if system RAM <2-3x model weights in gb**
```
sudo fallocate -l 32G /mnt/data/swapfile
sudo chmod 600 /mnt/data/swapfile
sudo mkswap /mnt/data/swapfile
sudo swapon /mnt/data/swapfile
```

## **Setup for building TensorRT and TensorRT-LLM Engine**

- Downloading relevant repositories into base directory.
```
sudo apt-get update && sudo apt-get install git-lfs -y
git lfs install

git clone https://github.com/Eros483/tensorrtllm_backend.git
cd tensorrtllm_backend
git submodule update --init --recursive
cd ..

git clone https://huggingface.co/llava-hf/llava-1.5-7b-hf

mkdir -p engines
```
- It is assumed that the relevant scripts are already present in `/scripts` directory.

- Launching Triton Container and engine build
  - Make sure to be present in base directory.
  - Make sure to verify file and check configuration as per requirements.
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

cd /workspace
python3 llava_engine_build.py
```
This concludes engine build, exit container, and return to base directory.

---

## **Initializing Triton Inference server**
````
docker run --rm -it --net=host \
  --shm-size=2g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --gpus all \
  -v $(pwd)/tensorrtllm_backend:/workspace/tensorrtllm_backend \
  -v $(pwd)/engines:/engines \
  -v $(pwd)/llava-1.5-7b-hf:/llava-1.5-7b-hf \
  -v $(pwd)/scripts:/workspace/tensorrtllm_backend/scripts \
  -w /workspace/tensorrtllm_backend \
  nvcr.io/nvidia/tritonserver:24.12-trtllm-python-py3 bash
````
```
bash scripts/init_server.sh
```
Configuration setup completed. Stay inside container.

---
## **Launching Triton Inference server**
```
tritonserver \
--model-repository=multimodal_ifb/ \
--model-control-mode=explicit \
--grpc-port=8001 \
--http-port=8000 \
--metrics-port=8002 \
--disable-auto-complete-config \
--backend-config=python,shm-region-prefix-name=prefix0_ \
--backend-config=tensorrtllm,default-max-batch-size=4 \
--cuda-memory-pool-byte-size=0:2000000000
```
---
## **Sending Inference Requests**
- ### **Enter existing docker container**
    ```
    # replace with your own docker container id using docker ps
    docker exec -it e6ea3c587db5 bash
    ```
    ```
    pip3 install tritonclient[all]
    pip install tabulate
    ```

- ### **Launch Models**
  - For Multimodal, specifically Llava-1.5
  ```
  curl -X POST localhost:8000/v2/repository/models/preprocessing/load
  curl -X POST localhost:8000/v2/repository/models/postprocessing/load
  curl -X POST localhost:8000/v2/repository/models/multimodal_encoders/load
  curl -X POST localhost:8000/v2/repository/models/tensorrt_llm/load
  curl -X POST localhost:8000/v2/repository/models/tensorrt_llm_bls/load
  curl -X POST localhost:8000/v2/repository/models/ensemble/load
  ```

- ### **Sending Requests**

    - Verification Request to test if setup is working
    ```
    python3 tensorrt_llm/triton_backend/tools/multimodal/client.py \
    --text 'Describe this image in detail.' \
    --image 'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png' \
    --request-output-len 50 \
    --model_type llava
    ```

    - Batched Request
    ```
    cd scripts
    python3 test_triton_server.py --batch-size 4 #or vary accordingly
    ```

- ### **Unloading Models**
  ```
  curl -X POST localhost:8000/v2/repository/models/preprocessing/unload
  curl -X POST localhost:8000/v2/repository/models/postprocessing/unload
  curl -X POST localhost:8000/v2/repository/models/multimodal_encoders/unload
  curl -X POST localhost:8000/v2/repository/models/tensorrt_llm/unload
  curl -X POST localhost:8000/v2/repository/models/tensorrt_llm_bls/unload
  curl -X POST localhost:8000/v2/repository/models/ensemble/unload
  ```


- **Server metrics**: `curl localhost:8002/metrics`
- **Server Shutdown**: `pkill tritonserver`
