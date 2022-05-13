#!/bin/sh

PROJECT_DIR=cdp_unsupervised_ad

mkdir "${HOME}"/${PROJECT_DIR}/results

for SEED in 0 1 2 3 4
do
  for ORIGINALS in 76
  do
    mkdir "${HOME}"/${PROJECT_DIR}/results/${ORIGINALS}
    for MODE in 1 2 3 4 5 6
    do
      mkdir "${HOME}"/${PROJECT_DIR}/results/${ORIGINALS}/${MODE}
      echo python3 -u \
      "${HOME}"/cdp_unsupervised_ad/src/t2x.py \
      --data "${HOME}"/${PROJECT_DIR}/datasets/1x1 \
      --mode $MODE \
      --originals $ORIGINALS \
      --epochs 150 \
      --bs 16 \
      --lr 0.01 \
      --tp 0.4 \
      --vp 0.1 \
      --seed $SEED \
      --result_dir "${HOME}"/${PROJECT_DIR}/results/${ORIGINALS}/${MODE}/${SEED}
    done
  done
done
