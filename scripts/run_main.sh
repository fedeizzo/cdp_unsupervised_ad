#!/bin/sh

python3 -u ${HOME}/cdp_fasatflow/src/main.py --data /users/p/pulfer/cdp_fastflow/1x1 --epochs 100 --bs 8 --lr 0.001 --tp 0.3 --fc 8 --nl 8 --seed 0
