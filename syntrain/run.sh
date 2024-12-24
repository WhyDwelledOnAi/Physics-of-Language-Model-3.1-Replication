#!/bin/bash
size="tiny"
# rm -rf output/"$size"
mkdir -p output/"$size"

CUDA_VISIBLE_DEVICES=0,1  accelerate launch train.py \
--size "$size" \
--output_dir output/"$size"/ \
2>&1 | tee -a output/"$size"/train.log