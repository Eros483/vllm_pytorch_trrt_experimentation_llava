export MODEL_NAME="llava-1.5-7b-hf"
export HF_MODEL_PATH="/llava-1.5-7b-hf"
export ENGINE_PATH="/engines/llava1.5/llm"
export MULTIMODAL_ENGINE_PATH="/engines/llava1.5/vision"
export ENCODER_INPUT_FEATURES_DTYPE="TYPE_FP16"
export MAX_BATCH_SIZE=4
export MAX_QUEUE_DELAY=20000
export GPU_FRACTION_USED=0.9
export PMIX_MCA_gds=hash

rm -rf multimodal_ifb
cp -r tensorrt_llm/triton_backend/all_models/inflight_batcher_llm/ multimodal_ifb
cp -r tensorrt_llm/triton_backend/all_models/multimodal/ensemble multimodal_ifb/
cp -r tensorrt_llm/triton_backend/all_models/multimodal/multimodal_encoders multimodal_ifb/

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
prompt_embedding_table_data_type:TYPE_FP16,\
kv_cache_free_gpu_mem_fraction:${GPU_FRACTION_USED}

# configuring preprocessing
python3 tensorrt_llm/triton_backend/tools/fill_template.py \
  -i multimodal_ifb/preprocessing/config.pbtxt \
  tokenizer_dir:${HF_MODEL_PATH},\
triton_max_batch_size:${MAX_BATCH_SIZE},\
preprocessing_instance_count:1,\
multimodal_model_path:${MULTIMODAL_ENGINE_PATH},\
engine_dir:${ENGINE_PATH},\
max_num_images:1,\
max_queue_delay_microseconds:${MAX_QUEUE_DELAY}

# configure postprocessing
python3 tensorrt_llm/triton_backend/tools/fill_template.py \
  -i multimodal_ifb/postprocessing/config.pbtxt \
  tokenizer_dir:${HF_MODEL_PATH},\
triton_max_batch_size:${MAX_BATCH_SIZE},\
postprocessing_instance_count:1

# configure ensemble
python3 tensorrt_llm/triton_backend/tools/fill_template.py \
  -i multimodal_ifb/ensemble/config.pbtxt \
  triton_max_batch_size:${MAX_BATCH_SIZE},\
logits_datatype:TYPE_FP32

# configure BLS
python3 tensorrt_llm/triton_backend/tools/fill_template.py \
  -i multimodal_ifb/tensorrt_llm_bls/config.pbtxt \
  triton_max_batch_size:${MAX_BATCH_SIZE},\
decoupled_mode:False,\
bls_instance_count:1,\
accumulate_tokens:False,\
tensorrt_llm_model_name:tensorrt_llm,\
multimodal_encoders_name:multimodal_encoders,\
logits_datatype:TYPE_FP32,\
prompt_embedding_table_data_type:TYPE_FP16

# configure multimodal encoders
python3 tensorrt_llm/triton_backend/tools/fill_template.py \
  -i multimodal_ifb/multimodal_encoders/config.pbtxt \
  triton_max_batch_size:${MAX_BATCH_SIZE},\
multimodal_model_path:${MULTIMODAL_ENGINE_PATH},\
encoder_input_features_data_type:${ENCODER_INPUT_FEATURES_DTYPE},\
hf_model_path:${HF_MODEL_PATH},\
max_queue_delay_microseconds:${MAX_QUEUE_DELAY},\
prompt_embedding_table_data_type:TYPE_FP16