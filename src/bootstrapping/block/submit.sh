#!/bin/bash -l
#SBATCH --job-name=blockbcn
#SBATCH --ntasks=12
#SBATCH -p k2-lowpri
#SBATCH --time=220:00:00
#SBATCH --mem 2G

python transfer_model.py 
