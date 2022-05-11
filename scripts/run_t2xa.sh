#!/bin/sh

#ORIGINALS=55
ORIGINALS=76

PROJECT_DIR=cdp_unsupervised_ad

mkdir ${HOME}/${PROJECT_DIR}/results
mkdir ${HOME}/${PROJECT_DIR}/results/${ORIGINALS}

for SEED in 0 1 2 3 4
do
  python3 -u \
  ${HOME}/${PROJECT_DIR}/src/t2xa.py \
  --data ${HOME}/${PROJECT_DIR}/datasets/1x1 \
  --originals $ORIGINALS \
  --epochs 150 \
  --bs 8 \
  --lr 0.01 \
  --tp 0.4 \
  --vp 0.1 \
  --seed $SEED \
  --result_dir ${HOME}/${PROJECT_DIR}/results/${ORIGINALS}/${SEED}
done


