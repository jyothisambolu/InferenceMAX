#!/usr/bin/bash

MODEL_CODE="${1%%_*}"
HF_HUB_CACHE_MOUNT="/dev/shm/hf_hub_cache/"

source benchmarks/${MODEL_CODE}_mi300x_docker.sh
