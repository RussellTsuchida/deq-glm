#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:1
#SBATCH --mem=512m
#SBATCH --time=24:00:00

module load cudnn/8.1.1-cuda112
module load intel-mkl/2020.1.217
module load python/3.9.4
module load cuda/11.2.1
module load pytorch/1.8.1-py39-cuda112
module load torchvision/0.9.1-py39

# Activate virtual environment
#source env/bin/activate

# Train the model
srun -u python -um code.scripts.cifar_test $1
