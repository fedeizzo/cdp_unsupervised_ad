#!/bin/sh

python3 -u \
  ${HOME}/cdp_fastflow/src/main.py \
  --data /home/users/p/pulfer/cdp_fastflow/1x1 \
  --epochs 100 \
  --bs 8 \
  --lr 0.001 \
  --tp 0.3 \
  --fc 256 \
  --nl 16 \
  --pretrained \
  --seed 0 \
  --model ${HOME}/cdp_fastflow/flow_model.pt
