#!/bin/bash

export PYTHONPATH=.:fabrique

export HF_HUB_CACHE=/data/hf_cache
mkdir -p $HF_HUB_CACHE

export JAX_COMPILATION_CACHE_DIR="/data/jax_cache"