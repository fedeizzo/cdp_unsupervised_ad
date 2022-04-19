#!/bin/sh

python3 -u \
  ${HOME}/cdp_fastflow/src/t2x.py \
  --data ${HOME}/cdp_fastflow/datasets/1x1 \
  --originals 76 \
  --epochs 300 \
  --bs 8 \
  --lr 0.01 \
  --tp 0.4 \
  --vp 0.1 \
  --seed 0
