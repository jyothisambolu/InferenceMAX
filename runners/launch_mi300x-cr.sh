#!/usr/bin/bash

MODEL="${1%%_*}"
HF_HUB_CACHE_MOUNT="/mnt/vdb/gha_cache/hf_hub_cache/"
source benchmarks/${MODEL}_mi300x_docker.sh
