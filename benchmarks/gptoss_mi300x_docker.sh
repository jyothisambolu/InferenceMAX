#!/usr/bin/env bash

# ========= Required Env Vars =========
# HF_TOKEN
# HF_HUB_CACHE
# MODEL
# PORT
# TP
# CONC
# MAX_MODEL_LEN

# Reference
# https://rocm.docs.amd.com/en/docs-7.0-rc1/preview/benchmark-docker/inference-vllm-gpt-oss-120b.html#run-the-inference-benchmark

export VLLM_USE_AITER_TRITON_FUSED_SPLIT_QKV_ROPE=1
export VLLM_USE_AITER_TRITON_FUSED_ADD_RMSNORM_PAD=1
export VLLM_USE_AITER_TRITON_GEMM=1
export VLLM_ROCM_USE_AITER=1
export VLLM_USE_AITER_UNIFIED_ATTENTION=1
export VLLM_ROCM_USE_AITER_MHA=0
export TRITON_HIP_PRESHUFFLE_SCALES=1
export VLLM_DISABLE_COMPILE_CACHE=1
export HSA_NO_SCRATCH_RECLAIM=1

vllm serve $MODEL --port=$port \
--tensor-parallel-size=$TP \
--gpu-memory-utilization=0.95 \
--compilation-config='{"full_cuda_graph": true}' \
--block-size=64 \
--swap-space=16 \
--no-enable-prefix-caching \
--disable-log-requests
