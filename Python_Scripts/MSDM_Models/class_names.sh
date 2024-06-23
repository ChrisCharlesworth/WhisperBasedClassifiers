#!/bin/bash

#SBATCH --job-name="Class_Names"
#SBATCH --time=08:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu-a100
#SBATCH --mem-per-cpu=180GB
#SBATCH --account=education-eemcs-courses-cse3000

module load 2023r1
module load python
module load py-pip
pip3 install numpy

srun python calculate_class_names.py > class_names.log