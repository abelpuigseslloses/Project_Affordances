#!/bin/sh
#SBATCH --time=00:15:00
#SBATCH -N 1
#SBATCH --gres=gpu:1

. /etc/bashrc
. /etc/profile.d/modules.sh
module load cuda80/toolkit
./cuda-app opts

python3 train_script_wandb.py --num_attributes 102 --command train --frozen_layers 0