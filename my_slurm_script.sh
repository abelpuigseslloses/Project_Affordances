#!/bin/bash
#SBATCH --job-name=example_job
#SBATCH --output=output.txt
#SBATCH --time=47:59:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -C TitanX

# Unload the default Python module
module unload python/2.7.13

# Load the desired Python 3.5 module
module load python/3.6.0

python3 train_script.py