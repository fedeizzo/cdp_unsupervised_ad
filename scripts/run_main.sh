#!/bin/sh

PROJECT_DIR=cdp_unsupervised_ad

mkdir "${HOME}"/${PROJECT_DIR}/results
mkdir "${HOME}"/${PROJECT_DIR}/results/55
mkdir "${HOME}"/${PROJECT_DIR}/results/76


for SEED in 0 1 2 3 4
do
  for ORIGINAL in 76
  do
    for MODE in 1 2 3 4 5 6
    do
      mkdir "${HOME}"/${PROJECT_DIR}/results/${ORIGINAL}/${MODE}/${SEED}
      python3 -u \
      "${HOME}"/cdp_unsupervised_ad/src/t2x.py \
      --data "${HOME}"/${PROJECT_DIR}/datasets/1x1 \
      --mode $MODE \
      --originals $ORIGINAL \
      --epochs 150 \
      --bs 16 \
      --lr 0.01 \
      --tp 0.4 \
      --vp 0.1 \
      --seed $SEED \
      --result_dir "${HOME}"/${PROJECT_DIR}/results/${ORIGINAL}/${MODE}/${SEED}
    done
  done
done
