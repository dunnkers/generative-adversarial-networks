#!/bin/bash

#SBATCH --time=2:00:00

#SBATCH --ntasks=1

#SBATCH --mem=4GB

#SBATCH --job-name=dataset-creation

#SBATCH --mail-type=ALL

#SBATCH --mail-user=email@example.com

#SBATCH --partition=regular

STYLEGAN_PATH=/your/path/to/stylegan
IMAGES_PATH=/your/path/to/images

module load TensorFlow/1.10.1-foss-2018a-Python-3.6.4 

cd $STYLEGAN_PATH

source venv/bin/activate

srun python dataset_tool.py create_from_images datasets/van-gogh $IMAGES_PATH 
