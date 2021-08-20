# deq-glm

**CIFAR10 TEST**
1. Download onto Bracewell. 
2. Make a new empty folder at 
>/deq-glm/outputs/cifar10
3. Download the CIFAR10 dataset. To do this, I had to run 
>python -m code.scripts.cifar_test 

OUTSIDE of slurm, then terminate the job after the files are downloaded. Not sure why but slurm doesn't like downloading the file.
4. Run
> ./run_all_cifar.sh

You may change 
>for i in $(seq 0 5)

on line 4 of run_all_cifar.sh to run 100 jobs in parallel.



**REGRESSION TEST**
>python3 -m code.scripts.regression_test
