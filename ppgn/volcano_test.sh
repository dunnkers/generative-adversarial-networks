#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --mem=4000
#SBATCH --output=test.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module load Python
module load Python/3.8.6-GCCcore-10.2.0
module load PyTorch/1.6.0-fosscuda-2019b-Python-3.7.4 
module load OpenCV/4.2.0-fosscuda-2019b-Python-3.7.4
module load matplotlib/3.1.1-fosscuda-2019b-Python-3.7.4
module load TensorFlow/2.3.1-fosscuda-2019b-Python-3.7.4
module load binutils/2.35-GCCcore-10.2.0
module load scikit-image
module load gompic/2019b
module load gcccuda/2019b

./6_class_conditional_sampling_from_real_image.sh 980
