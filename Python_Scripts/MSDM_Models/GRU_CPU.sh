#!/bin/bash

#SBATCH --job-name="GRU_Compute"
#SBATCH --time=07:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu-a100
#SBATCH --mem-per-cpu=180GB
#SBATCH --account=education-eemcs-courses-cse3000

module load 2022r2
module load cuda/11.6
module load 2023r1
module load python
module load py-pip
pip3 install torch
pip3 install numpy
pip3 install scikit-learn

srun python GRU_k_fold.py > GRU_k_fold_CPU.log