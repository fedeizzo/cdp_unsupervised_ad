#!/bin/sh
#SBATCH --job-name cdp_fastflow
#SBATCH --error error.e%j
#SBATCH --output out.o%j
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem-per-cpu 20000
#SBATCH --partition shared-gpu
#SBATCH --gpus=titan:1
#SBATCH --time 11:55:00

# Loading required modules
module load GCC/10.2.0 CUDA/11.1.1
module load Python/3.8.6

# Installing torchvision in a Python Virtual environment
# virtualenv ~/cdp_fastflow/venv
. ~/cdp_fastflow/venv/bin/activate
# ~/cdp_fastflow/venv/bin/python -m pip install --upgrade pip
# pip install numpy matplotlib opencv-python
# pip install torch
# pip install albumentations anomalib
# pip list
# pip install -r ~/cdp_fastflow/requirements.txt

# Running Main Program
# srun ${HOME}/cdp_fastflow/scripts/run_t2x.sh
srun ${HOME}/cdp_fastflow/scripts/run_t2ax.sh
