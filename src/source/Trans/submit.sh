#!/bin/bash -l
#SBATCH --job-name=Trans
#SBATCH --ntasks=12
#SBATCH -p k2-gpu
#SBATCH --time=70:00:00
#SBATCH --mem 2G


python main.py