#!/bin/bash -l

# Train the model
for i in $(seq 0 99)
do
    sbatch run_smooth.sh $i
done
