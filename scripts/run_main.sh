#!/bin/sh

python3 -u \
  ${HOME}/cdp_fastflow/src/main.py \
  --data /home/users/p/pulfer/cdp_fastflow/1x1 \
  --model ${HOME}/cdp_fastflow/flow_model_sd.pt \
  --epochs 500 \
  --bs 16 \
  --lr 0.0001 \
  --tp 0.3 \
  --fc 256 \
  --nl 16 \
  --seed 0 \
  --pretrained \
  --freeze_backbone
