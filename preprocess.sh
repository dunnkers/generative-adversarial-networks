#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --mem=50000
#SBATCH --nodes=1
#SBATCH --partition=short
#SBATCH --job-name=van-gogh-preprocessing
#SBATCH --output=/data/s2995697/dcgan/slurm/slurm-%A_%a.out
#SBATCH --array=0
# → do make sure /logs directory exists!

module load Python
module load OpenCV
python v0/preprocess.py --dataset-folder /data/s2995697/van-gogh-paintings --output-folder /data/s2995697/van-gogh-paintings-augmented/
