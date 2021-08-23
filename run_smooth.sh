#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:1
##SBATCH --mem=512m
#SBATCH --time=05:00:00


# Train the model
module load cuda/11.2.1
module load cudnn/8.1.1-cuda112
#module load magma/2.6.0-ipl64-cuda112
module load intel-mkl/2020.1.217
module load python/3.9.4
module load pytorch/1.9.0-py39-cuda112
module load torchvision/0.9.1-py39 

srun -u python -um code.scripts.regression_test $1
