#!/bin/sh

#ORIGINALS=55
ORIGINALS=76
mkdir ${HOME}/cdp_fastflow/results
mkdir ${HOME}/cdp_fastflow/results/${ORIGINALS}

for SEED in 0 1 2 3 4
do
  python3 -u \
  ${HOME}/cdp_fastflow/src/t2xa.py \
  --data ${HOME}/cdp_fastflow/datasets/1x1 \
  --originals $ORIGINALS \
  --epochs 150 \
  --bs 8 \
  --lr 0.01 \
  --tp 0.4 \
  --vp 0.1 \
  --seed $SEED \
  --result_dir ${HOME}/cdp_fastflow/results/${ORIGINALS}/${SEED}
done


