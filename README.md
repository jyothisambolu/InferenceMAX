# InferenceMAX


## Benchmark Client Configuration

| Parameter | Values |
| :-: | :- |
| (`ISL`, `OSL`) | (1024, 1024), (1024, 8192), (8192, 1024) |
| `CONC` | 4, 8, 16, 32, 64 |
| `RANDOM_RANGE_RATIO` | 0.8 |

```bash
python3 bench_serving/benchmark_serving.py \
--model $MODEL --backend vllm \
--base-url http://0.0.0.0:\$PORT \
--dataset-name random \
--random-input-len $ISL --random-output-len $OSL --random-range-ratio $RANDOM_RANGE_RATIO \
--num-prompts $(( $CONC * 10 )) --max-concurrency $CONC \
--request-rate inf --ignore-eos \
--save-result --percentile-metrics 'ttft,tpot,itl,e2el'
```


## Server Configurations

| GPU | Model | Image | Server Launch Command |
| :-: | :- | :- | :-: |
| H200 | `nvidia/Llama-3.1-70B-Instruct-FP8` | `kedarpotdar147/vllm0.1:latest` | [Link](#h200-70b) |
| H200 | `nvidia/Llama-4-Scout-17B-16E-Instruct-FP8` | `kedarpotdar147/vllm0.1:latest` | [Link](#h200-scout) |
| H200 | `deepseek-ai/DeepSeek-R1-0528` | `lmsysorg/sglang:v0.4.9.post1-cu126` | [Link](#h200-dsr1) |
| B200 | `nvidia/Llama-3.1-70B-Instruct-FP8` | `kedarpotdar147/vllm0.1:latest` | [Link](#b200-70b) |
| B200 | `nvidia/Llama-4-Scout-17B-16E-Instruct-FP8` | `kedarpotdar147/vllm0.1:latest` | [Link](#b200-scout) |
| B200 | `deepseek-ai/DeepSeek-R1-0528` | `lmsysorg/sglang:v0.4.10.post1-cu128-b200` | [Link](#b200-dsr1) |


#### H200 70B

```bash
vllm serve $MODEL --host 0.0.0.0 --port $PORT \
--trust-remote-code --quantization modelopt --gpu-memory-utilization 0.9 \
--pipeline-parallel-size 1 --tensor-parallel-size $TP --max-num-seqs $CONC --max-num-batched-tokens 8192 --max-model-len $MAX_MODEL_LEN \
--enable-chunked-prefill --async-scheduling --no-enable-prefix-caching \
--compilation-config '{"pass_config": {"enable_fi_allreduce_fusion": true}, "custom_ops": ["+rms_norm"], "level": 3}'
```

#### H200 Scout

```bash
vllm serve $MODEL --host 0.0.0.0 --port $PORT \
--trust-remote-code --quantization modelopt --gpu-memory-utilization 0.9 \
--pipeline-parallel-size 1 --tensor-parallel-size $TP --max-num-seqs $CONC --max-num-batched-tokens 8192 --max-model-len $MAX_MODEL_LEN \
--enable-chunked-prefill --async-scheduling --no-enable-prefix-caching \
--compilation-config '{"pass_config": {"enable_fi_allreduce_fusion": true}, "custom_ops": ["+rms_norm"], "level": 3}'
```

#### H200 DSR1

```bash
export SGL_ENABLE_JIT_DEEPGEMM=1
python3 -m sglang.launch_server --model-path $MODEL --host 0.0.0.0 --port \$port --trust-remote-code \
--tp $TP --cuda-graph-max-bs $CONC
```

#### B200 70B

```bash
vllm serve $MODEL --host 0.0.0.0 --port $PORT \
--trust-remote-code --quantization modelopt --kv-cache-dtype fp8 --gpu-memory-utilization 0.9 \
--pipeline-parallel-size 1 --tensor-parallel-size $TP --max-num-seqs $CONC --max-num-batched-tokens 8192 --max-model-len $MAX_MODEL_LEN \
--enable-chunked-prefill --async-scheduling --no-enable-prefix-caching \
--compilation-config '{"pass_config": {"enable_fi_allreduce_fusion": true}, "custom_ops": ["+rms_norm"], "level": 3}'
```

#### B200 Scout

```bash
vllm serve $MODEL --host 0.0.0.0 --port $PORT \
--trust-remote-code --quantization modelopt --kv-cache-dtype fp8 --gpu-memory-utilization 0.9 \
--pipeline-parallel-size 1 --tensor-parallel-size $TP \
--max-num-seqs $CONC --max-num-batched-tokens 8192 --max-model-len $MAX_MODEL_LEN \
--enable-chunked-prefill --async-scheduling --no-enable-prefix-caching \
--compilation-config '{"pass_config": {"enable_fi_allreduce_fusion": true}, "custom_ops": ["+rms_norm"], "level": 3}'
```

#### B200 DSR1

```bash
export SGL_ENABLE_JIT_DEEPGEMM=0
python3 -m sglang.launch_server --model-path $MODEL --host 0.0.0.0 --port $PORT --trust-remote-code \
--tp $TP --dp 1 \
--max-running-requests $CONC --cuda-graph-max-bs $CONC \
--disable-radix-cache --chunked-prefill-size 32768 --mem-fraction-static 0.89 --max-prefill-tokens 32768 \
--attention-backend trtllm_mla --disable-shared-experts-fusion --enable-flashinfer-trtllm-moe
```


## Sponsors

- NVIDIA
- Crusoe
- Nebius
- CoreWeave
- AMD
- TogetherAI
