#!/bin/bash

CKPT_DIR=$1
TOKENIZER_PATH=$2

export ENFLAME_ENABLE_TF32=true
export OMP_NUM_THREADS=5
export ECCL_MAX_NCHANNELS=2
export ECCL_RUNTIME_3_0_ENABLE=true

export ENFLAME_LOG_LEVEL=ERROR
export ECCL_DEBUG=WARN

export TOPS_EXE_CACHE_PATH=./graph_cache
export TOPS_EXE_CACHE_DISABLE=false

torchrun --nproc_per_node 1 ../../llama_inference_service/src/inference_service.py -c $CKPT_DIR  -t $TOKENIZER_PATH
