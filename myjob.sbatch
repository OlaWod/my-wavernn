#!/bin/bash

#SBATCH --account=
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1

module load anaconda
source ~/.bashrc
conda activate universal

cd $SLURM_SUBMIT_DIR

python train.py