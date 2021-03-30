#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --mem=100000
#SBATCH --nodes=1
#SBATCH --partition=gpushort
#SBATCH --gres=gpu:1
#SBATCH --job-name=dc-gan-van-gogh
#SBATCH --output=/data/s2995697/dcgan/slurm/slurm-%A_%a.out
#SBATCH --array=0
# → do make sure /logs directory exists!

module load TensorFlow
python dcgan/dcgan.py \
    --dataset /data/s2995697/van-gogh-paintings-augmented \
    --output-images /data/s2995697/dcgan/images \
    --saved-model /data/s2995697/dcgan/models
