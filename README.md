# InferenceMAX


## System Design

We hook GPU nodes into the GitHub Action of this repo as **runners**, and a GitHub Action **workflow** assigns work to the runners.
- Workflow scheduler: The workflow that launches the full benchmark run sweep
- LLaMA 70B template and DSR1 template: The logic for declaring GPUs and their benchmark run configs
- Benchmark template: The core logic of launching a single benchmark run

A runner loads code from this repo and executes them.
- `benchmarks/`: Scripts of benchmark configs, including the benchmark server and the benchmark client setup
- `runners/`: Scripts that launches the runners

The flow of operations is as follows:
1. GitHub Action workflow assigns a runner a config to benchmark, specified by the workflow YAML file.  
   Config includes: GPU type, Model, ISL / OSL, TP
1. The runner pulls the repo code and launchs the respective GPU node script in `runner/`,
   specifying the benchmark config: Model, ISL / OSL, TP
1. The script executes the corresponding script in `benchmarks/`, configuring it with the benchmark config


## Benchmark Client Configuration

| Parameter | Values |
| :-: | :- |
| (`ISL`, `OSL`) | (1024, 1024), (1024, 8192), (8192, 1024) |
| `CONC` | 4, 8, 16, 32, 64 |
| `RANDOM_RANGE_RATIO` | 0.8 |

```bash
git clone https://github.com/kimbochen/bench_serving.git 
python3 bench_serving/benchmark_serving.py \
--model $MODEL --backend vllm \
--base-url http://0.0.0.0:$PORT \
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
| H200 | `deepseek-ai/DeepSeek-R1-0528` | `lmsysorg/sglang:v0.4.9.post1-cu126` | [Link](#h200-dsr1) |
| B200 | `nvidia/Llama-3.1-70B-Instruct-FP8` | `kedarpotdar147/vllm0.1:latest` | [Link](#b200-70b) |
| B200 | `deepseek-ai/DeepSeek-R1-0528` | `lmsysorg/sglang:v0.4.10.post1-cu128-b200` | [Link](#b200-dsr1) |
| MI300X | `amd/Llama-3.1-70B-Instruct-FP8-KV` | `rocm/vllm-dev:nightly_official_0729_rc1_20250718` | [Link](#mi300x-70b) |
| MI300X | `deepseek-ai/DeepSeek-R1-0528` | `lmsysorg/sglang:v0.4.9.post2-rocm630-mi30x` | [Link](#mi300x-dsr1) |
| MI325X | `amd/Llama-3.1-70B-Instruct-FP8-KV` | `rocm/vllm-dev:nightly_official_0729_rc1_20250718` | [Link](#mi325x-70b) |
| MI325X | `deepseek-ai/DeepSeek-R1-0528` | `lmsysorg/sglang:v0.4.9.post2-rocm630-mi30x` | [Link](#mi325x-dsr1) |
| MI355X | `amd/Llama-3.1-70B-Instruct-FP8-KV` | `rocm/7.0-preview:rocm7.0_preview_ubuntu_22.04_vllm_0.9.1_mi35x_alpha` | [Link](#mi355x-70b) |


#### H200 70B

```bash
vllm serve $MODEL --host 0.0.0.0 --port $PORT \
--trust-remote-code --quantization modelopt --gpu-memory-utilization 0.9 \
--pipeline-parallel-size 1 --tensor-parallel-size $TP --max-num-seqs $CONC --max-num-batched-tokens 8192 --max-model-len $MAX_MODEL_LEN \
--enable-chunked-prefill --async-scheduling --no-enable-prefix-caching \
--compilation-config '{"pass_config": {"enable_fi_allreduce_fusion": true}, "custom_ops": ["+rms_norm"], "level": 3}'
```

#### H200 DSR1

```bash
python3 -m sglang.launch_server --model-path $MODEL --host 0.0.0.0 --port $PORT --trust-remote-code \
--tensor-parallel-size=$TP --data-parallel-size=1 \
--disable-radix-cache --decode-log-interval 1 --cuda-graph-bs 4 8 16 32 64 128 256 --cuda-graph-max-bs 256 --max-running-requests 512
```

#### B200 70B

```bash
vllm serve $MODEL --host 0.0.0.0 --port $PORT \
--trust-remote-code --quantization modelopt --kv-cache-dtype fp8 --gpu-memory-utilization 0.9 \
--pipeline-parallel-size 1 --tensor-parallel-size $TP --max-num-seqs $CONC --max-num-batched-tokens 8192 --max-model-len $MAX_MODEL_LEN \
--enable-chunked-prefill --async-scheduling --no-enable-prefix-caching \
--compilation-config '{"pass_config": {"enable_fi_allreduce_fusion": true}, "custom_ops": ["+rms_norm"], "level": 3}'
```

#### B200 DSR1

```bash
export SGL_ENABLE_JIT_DEEPGEMM=0
python3 -m sglang.launch_server --model-path $MODEL --host 0.0.0.0 --port $PORT --trust-remote-code \
--tensor-parallel-size=$TP --data-parallel-size=1 \
--cuda-graph-max-bs 256 --max-running-requests 512 --mem-fraction-static 0.89 \
--chunked-prefill-size 32768 --max-prefill-tokens 32768 \
--disable-radix-cache --attention-backend trtllm_mla --disable-shared-experts-fusion --enable-flashinfer-trtllm-moe
```

#### MI300X 70B

```bash
export VLLM_USE_TRITON_FLASH_ATTN=0
vllm serve $MODEL --port $PORT \
--tensor-parallel-size $TP --distributed-executor-backend mp \
--dtype bfloat16 --quantization fp8 \
--max-num-seqs $CONC --max-model-len $MAX_MODEL_LEN --max-seq-len-to-capture $MAX_MODEL_LEN
```

#### MI300X DSR1

```bash
python3 -m sglang.launch_server --model-path $MODEL --host 0.0.0.0 --port $PORT --trust-remote-code \
--tp $TP --cuda-graph-max-bs $CONC
```

#### MI325X 70B

```bash
export VLLM_USE_TRITON_FLASH_ATTN=0
vllm serve $MODEL --port $PORT \
--tensor-parallel-size $TP --distributed-executor-backend mp \
--dtype bfloat16 --quantization fp8 \
--max-num-seqs $CONC --max-model-len $MAX_MODEL_LEN --max-seq-len-to-capture $MAX_MODEL_LEN
```

#### MI325X DSR1

```bash
python3 -m sglang.launch_server --model-path $MODEL --host 0.0.0.0 --port $PORT --trust-remote-code \
--tp $TP --cuda-graph-max-bs $CONC
```

#### MI355X 70B

```bash
export VLLM_USE_TRITON_FLASH_ATTN=0
vllm serve $MODEL --port $PORT \
--tensor-parallel-size $TP --distributed-executor-backend mp \
--dtype bfloat16 --quantization fp8 \
--max-num-seqs $CONC --max-model-len $MAX_MODEL_LEN --max-seq-len-to-capture $MAX_MODEL_LEN
```

## Sponsors

- NVIDIA
- Crusoe
- Nebius
- CoreWeave
- AMD
- TogetherAI
