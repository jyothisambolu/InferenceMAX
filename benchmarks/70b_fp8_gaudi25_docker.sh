#!/usr/bin/env bash

# === Required Env Vars === 
# HF_TOKEN
# HF_HUB_CACHE
# IMAGE
# MODEL
# MAX_MODEL_LEN
# TP
# CONC

pip install -q datasets pandas

mkdir -p /tmp/vllm_config
CONFIG_FILE="/tmp/vllm_config/config.yaml"

cat > $CONFIG_FILE << EOF
kv-cache-dtype: fp8_inc
async-scheduling: true
no-enable-prefix-caching: true
quantization: inc
weights-load-device: cpu
max-num-batched-tokens: 8192
max-model-len: 10240
EOF

export QUANT_CONFIG=/software/inc/llama-3.3-70b-instruct-$TP/maxabs_quant_g3.json
export PYTHONNOUSERSITE=1

vllm serve $MODEL --host=0.0.0.0 --port=$PORT \
--config=config.yaml \
--device=hpu \
--tensor-parallel-size=$TP \
--max-num-seqs=$CONC \
--disable-log-requests
