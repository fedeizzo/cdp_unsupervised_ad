#!/bin/sh

python3 -u ${HOME}/cdp_fastflow/src/main.py --data /home/users/p/pulfer/cdp_fastflow/1x1 --epochs 100 --bs 8 --lr 0.0001 --tp 0.3 --fc 8 --nl 8 --pretrained --seed 0
