#!/usr/bin/bash

MODEL_CODE="${1%%_*}"
HF_HUB_CACHE_MOUNT="/mnt/vdb/gha_cache/hf_hub_cache/"
source benchmarks/${MODEL_CODE}_mi300x_docker.sh
