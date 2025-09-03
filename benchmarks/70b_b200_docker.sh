#!/usr/bin/bash

# ========= Required Env Vars =========
# HF_TOKEN
# HF_HUB_CACHE
# MODEL
# PORT
# TP
# CONC
# MAX_MODEL_LEN

set -x
pip install "git+https://github.com/flashinfer-ai/flashinfer.git@9720182476ede910698f8d783c29b2ec91cec023#egg=flashinfer-python"
pip install --upgrade nvidia-nccl-cu12==2.26.2.post1

export VLLM_FLASHINFER_ALLREDUCE_FUSION_THRESHOLDS_MB='{"2":32,"4":32,"8":8}'

vllm serve $MODEL --host 0.0.0.0 --port $PORT \
--trust-remote-code --kv-cache-dtype fp8 --gpu-memory-utilization 0.9 \
--pipeline-parallel-size 1 --tensor-parallel-size $TP --max-num-seqs $CONC --max-num-batched-tokens 8192 --max-num-seqs 512 --max-model-len $MAX_MODEL_LEN \
--enable-chunked-prefill --async-scheduling --no-enable-prefix-caching \
--compilation-config '{"pass_config": {"enable_fi_allreduce_fusion": true}, "custom_ops": ["+rms_norm"], "level": 3}' \
--disable-log-requests
