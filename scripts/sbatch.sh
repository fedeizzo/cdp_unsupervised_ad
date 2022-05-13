#!/bin/sh
#SBATCH --job-name cdp_unsupervised_ad
#SBATCH --error error.e%j
#SBATCH --output out.o%j
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem-per-cpu 20000
#SBATCH --partition private-sip-gpu
#SBATCH --gpus=ampere:1
#SBATCH --time 11:55:00

# Loading required modules
module load GCC/10.2.0 CUDA/11.1.1
module load Python/3.8.6

# Installing torchvision in a Python Virtual environment
# virtualenv ~/cdp_unsupervised_ad/venv
. ~/cdp_unsupervised_ad/venv/bin/activate
# ~/cdp_unsupervised_ad/venv/bin/python -m pip install --upgrade pip
# pip install numpy matplotlib opencv-python sklearn
# pip install torch
# pip install albumentations anomalib
# pip list
# pip install -r ~/cdp_unsupervised_ad/requirements.txt

# Running Main Program
srun ${HOME}/cdp_unsupervised_ad/scripts/run_main.sh
