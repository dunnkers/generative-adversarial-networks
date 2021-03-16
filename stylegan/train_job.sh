#!/bin/bash

#SBATCH --time=1-0:00:00

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=1

#SBATCH --mem=40GB

#SBATCH --job-name=training-run-continue

#SBATCH --mail-type=ALL

#SBATCH --mail-user=email@example.com

#SBATCH --partition=gpu

#SBATCH --gres=gpu:k40:2

STYLEGAN_PATH=/your/path/to/stylegan

module load TensorFlow/1.10.1-fosscuda-2018a-Python-3.6.4

cd $STYLEGAN_PATH

source venv/bin/activate

srun python train.py
