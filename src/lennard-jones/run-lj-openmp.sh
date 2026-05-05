#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --partition=gpu
#SBATCH --job-name=lj-final-bench
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --nodes=1
#SBATCH --output=./src/results/full_benchmark.log

module load CUDA

# Build
make clean
make

# Parameters
PARTICLES=(1000 2000 4000 8000)
STEPS=5000
CORES=24
TRIALS=5

echo "Starting Benchmark: $TRIALS trials per particle count using $CORES cores."

for N in "${PARTICLES[@]}"
do
    echo "------------------------------------------------"
    echo "Testing N=$N"


    export OMP_NUM_THREADS=$CORES
    export OMP_PROC_BIND=true

    echo "  Trial $i..."

    # We save each raw run to a separate file
    srun ./lj.out $N $STEPS > "./src/results/lj_${N}.log"
done

echo "Done!