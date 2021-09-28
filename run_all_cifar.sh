#!/bin/bash -l

# Train the model
for i in $(seq 0 100)
do
    sbatch run_cifar.sh $i
done
