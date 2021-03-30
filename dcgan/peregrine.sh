#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --mem=50000
#SBATCH --nodes=1
#SBATCH --partition=gpushort
#SBATCH --gres=gpu:1
#SBATCH --job-name=dc-gan
#SBATCH --output=/data/s2995697/dcgan/slurm/slurm-%A_%a.out
#SBATCH --array=0
# → do make sure /logs directory exists!

module load TensorFlow
python dcgan/dcgan.py
