#!/bin/bash -l

#SBATCH --workdir /home/cchen/projects/DyNav/dynav
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 16
#SBATCH --mem 48G
#SBATCH --time 24:00:00
#SBATCH --account civil-459

module load gcc

source ~/.bashrc
pyenv shell py3

python train.py --policy cadrl --output_dir data/cadrl
