#!/bin/bash

set -ex

export CUDA_VISIBLE_DEVICES=7

# Test Majorness
/data00/qiyuhang/venvs/cutedsl/bin/python3 dense_gemm_persistent_v3.py \
  --mnkl 4096,4096,4096,8 \
  --tile_shape_mn 128,128 \
  --cluster_shape_mn 1,1 \
  --a_major k \
  --b_major k

/data00/qiyuhang/venvs/cutedsl/bin/python3 dense_gemm_persistent_v3.py \
  --mnkl 4096,4096,4096,8 \
  --tile_shape_mn 128,128 \
  --cluster_shape_mn 1,1 \
  --a_major m \
  --b_major k

/data00/qiyuhang/venvs/cutedsl/bin/python3 dense_gemm_persistent_v3.py \
  --mnkl 4096,4096,4096,8 \
  --tile_shape_mn 128,128 \
  --cluster_shape_mn 1,1 \
  --a_major k \
  --b_major n

/data00/qiyuhang/venvs/cutedsl/bin/python3 dense_gemm_persistent_v3.py \
  --mnkl 4096,4096,4096,8 \
  --tile_shape_mn 128,128 \
  --cluster_shape_mn 1,1 \
  --a_major m \
  --b_major n

# Test Multicast
/data00/qiyuhang/venvs/cutedsl/bin/python3 dense_gemm_persistent_v3.py \
  --mnkl 4096,4096,4096,8 \
  --tile_shape_mn 128,64 \
  --cluster_shape_mn 1,2 \
  --a_major k \
  --b_major k

/data00/qiyuhang/venvs/cutedsl/bin/python3 dense_gemm_persistent_v3.py \
  --mnkl 4096,4096,4096,8 \
  --tile_shape_mn 64,64 \
  --cluster_shape_mn 2,2 \
  --a_major k \
  --b_major k

# Test Small N
/data00/qiyuhang/venvs/cutedsl/bin/python3 dense_gemm_persistent_v3.py \
  --mnkl 4096,16,4096,8 \
  --tile_shape_mn 128,16 \
  --cluster_shape_mn 1,1 \
  --a_major k \
  --b_major k

/data00/qiyuhang/venvs/cutedsl/bin/python3 dense_gemm_persistent_v3.py \
  --mnkl 4096,32,4096,8 \
  --tile_shape_mn 128,16 \
  --cluster_shape_mn 1,2 \
  --a_major k \
  --b_major k