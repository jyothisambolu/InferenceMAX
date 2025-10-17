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
async-scheduling: true
no-enable-prefix-caching: true
max-num-batched-tokens: 8192
max-model-len: 10240
EOF

export QUANT_CONFIG=/software/inc/llama-3.3-70b-instruct/maxabs_quant_g3.json
export QUANT_CONFIG=/software/data/vllm-benchmarks/inc/meta-llama-3.3-70b-instruct/maxabs_quant_g3.json
export PYTHONNOUSERSITE=1

vllm serve $MODEL --host=0.0.0.0 --port=$PORT \
--config=/tmp/vllm_config/config.yaml \
--device=hpu \
--tensor-parallel-size=$TP \
--max-num-seqs=$CONC \
--dtype=bfloat16 \
--disable-log-requests \
--disable-log-stats \
--quantization=inc \
--weights-load-device=cpu \
--kv-cache-dtype=fp8_inc \