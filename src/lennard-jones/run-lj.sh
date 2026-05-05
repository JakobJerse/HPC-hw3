#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --partition=gpu
#SBATCH --job-name=lennard-jones
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --output=./src/results/sequential/full_benchmark.log

#LOAD MODULES 
module load CUDA

#BUILD
make

# Array of particle counts to test
PARTICLES=(1000 2000 4000 8000)
STEPS=5000

for N in "${PARTICLES[@]}"
do
    echo "------------------------------------------------"
    echo "Running simulation for N=$N particles..."
    # We send the output of each run to its own specific log file
    srun ./lj.out $N $STEPS > "./src/results/sequential/lj_${N}_out.log"
done
