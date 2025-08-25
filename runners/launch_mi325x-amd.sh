#!/usr/bin/bash

MODEL="${1%%_*}"
HF_HUB_CACHE_MOUNT="/home/kimbosemianalysis/hf_hub_cache/"
source benchmarks/${MODEL}_mi325x_docker.sh
