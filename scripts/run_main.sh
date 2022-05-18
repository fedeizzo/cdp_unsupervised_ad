#!/bin/sh

PROJECT_DIR=cdp_unsupervised_ad

mkdir -p "${HOME}"/${PROJECT_DIR}/results
mkdir -p "${HOME}"/${PROJECT_DIR}/results/55
mkdir -p "${HOME}"/${PROJECT_DIR}/results/76


for SEED in 0 1 2 3 4
do
  for ORIGINAL in 76
  do
    mkdir -p "${HOME}"/${PROJECT_DIR}/results/${ORIGINAL}/seed_${SEED}
    for MODE in t2x t2xa x2t x2ta both both_a
    do
      mkdir -p "${HOME}"/${PROJECT_DIR}/results/${ORIGINAL}/seed_${SEED}/${MODE}
      python3 -u \
      "${HOME}"/cdp_unsupervised_ad/src/main.py \
      --data "${HOME}"/${PROJECT_DIR}/datasets/1x1 \
      --mode $MODE \
      --originals $ORIGINAL \
      --epochs 150 \
      --bs 64 \
      --lr 0.01 \
      --tp 0.4 \
      --vp 0.1 \
      --seed $SEED \
      --result_dir "${HOME}"/${PROJECT_DIR}/results/${ORIGINAL}/seed_${SEED}/${MODE}
    done
  done
done
