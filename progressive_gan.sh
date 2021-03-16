#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --mem=100000
#SBATCH --nodes=1
#SBATCH --partition=gpushort
#SBATCH --gres=gpu:1
#SBATCH --job-name=progressive-gan
#SBATCH --output=/data/s2995697/progressive-gan/slurm/slurm-%A_%a.out
#SBATCH --array=0
# → do make sure /logs directory exists!

module load TensorFlow
pip install tensorflow-gan tensorflow-probability tensorflow-datasets absl-py matplotlib --user
python progressive_gan/train_main.py --alsologtostderr --train_log_dir /data/s2995697/progressive-gan/tflogs