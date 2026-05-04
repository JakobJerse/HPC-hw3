#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --partition=gpu
#SBATCH --job-name=lennard-jones
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=2
#SBATCH --nodes=1
#SBATCH --output=lj_out.log

#LOAD MODULES 
module load CUDA

#BUILD
make

#RUN
srun ./lj.out

