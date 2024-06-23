#!/bin/bash

#SBATCH --job-name="GPU_BiLSTM"
#SBATCH --time=2:00:00
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:2
#SBATCH --mem-per-gpu=50GB
#SBATCH --account=education-eemcs-courses-cse3000

module load 2022r2
module load cuda/11.6
module load 2023r1
module load python
module load py-pip
pip3 install torch
pip3 install numpy
pip3 install scikit-learn

srun python BiLSTM_k_fold.py > BiLSTM_k_fold.log