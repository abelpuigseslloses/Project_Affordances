#!/bin/sh
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --array=0-1
#SBATCH --exclude=node436
. /etc/bashrc
. /etc/profile.d/modules.sh
module load cuda80/toolkit

# Set the CUDA_VISIBLE_DEVICES to use the GPU assigned to this array job
export CUDA_VISIBLE_DEVICES=$SLURM_ARRAY_TASK_ID

# Set W&B environment variables
export WANDB_ENTITY='abelpuigseslloses'
export WANDB_PROJECT='Prova'

# Start the W&B agent for the sweep
CUDA_VISIBLE_DEVICES=0 wandb agent vkast29l
CUDA_VISIBLE_DEVICES=1 wandb agent vkast29l
#CUDA_VISIBLE_DEVICES=2 wandb agent vkast29l
#CUDA_VISIBLE_DEVICES=3 wandb agent 6ds8r29h
#CUDA_VISIBLE_DEVICES=4 wandb agent oinmvt61
#CUDA_VISIBLE_DEVICES=5 wandb agent oinmvt61
#CUDA_VISIBLE_DEVICES=6 wandb agent oinmvt61
#CUDA_VISIBLE_DEVICES=7 wandb agent oinmvt61
#CUDA_VISIBLE_DEVICES=8 wandb agent oinmvt61
#CUDA_VISIBLE_DEVICES=9 wandb agent oinmvt61

wandb artifact put --type raw-logs --name slurm-outputs-$SLURM_JOB_ID slurm-$SLURM_JOB_ID.out
wandb artifact put --type raw-logs --name slurm-errors-$SLURM_JOB_ID slurm-$SLURM_JOB_ID.err



