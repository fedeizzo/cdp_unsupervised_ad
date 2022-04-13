#!/bin/sh

python3 -u \
  ${HOME}/cdp_fastflow/src/simple.py \
  --data ${HOME}/cdp_fastflow/datasets/1x1 \
  --epochs 5000 \
  --bs 8 \
  --lr 0.0001 \
  --tp 0.8 \
  --nl 32 \
  --seed 0
