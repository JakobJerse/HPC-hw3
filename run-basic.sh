#!/bin/bash
#SBATCH --reservation=fri
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=benchmarking_report.csv

module load CUDA

N_LIST=(1000 2000 4000 8000)
STEPS=5000

mkdir -p simulation_logs

echo "N,Implementation,AvgTime_Seconds,BestBlockSize"

for n in "${N_LIST[@]}"; do
    cd src/lennard-jones || exit
    make clean && make > /dev/null
    
    seq_total=0
    if [ $n -ge 8000 ]; then runs_seq=1; else runs_seq=5; fi
    
    for i in $(seq 1 $runs_seq); do
        LOG_FILE="../../simulation_logs/seq_N${n}_run${i}.log"
        srun ./lj.out $n $STEPS > "$LOG_FILE" 2>&1
        
        t=$(grep "Simulation time" "$LOG_FILE" | awk '{print $5}')
        if [ -z "$t" ]; then t=0; fi
        seq_total=$(echo "$seq_total + $t" | bc -l)
    done
    seq_avg=$(echo "scale=6; $seq_total / $runs_seq" | bc -l)
    echo "$n,Sequential,$seq_avg,N/A"
    cd ../..

    cd src_para/lennard-jones || exit
    make clean && make > /dev/null
    
    para_total=0
    best_bs="Unknown"
    
    for i in {1..5}; do
        LOG_FILE="../../simulation_logs/para_N${n}_run${i}.log"
        srun ./lj.out $n $STEPS > "$LOG_FILE" 2>&1
        
        t=$(grep "Simulation time" "$LOG_FILE" | awk '{print $5}')
        
        if [ "$i" -eq 1 ]; then
            bs=$(grep ">>> Best block size:" "$LOG_FILE" | awk '{print $4}')
            if [ ! -z "$bs" ]; then best_bs=$bs; fi
        fi
        
        if [ -z "$t" ]; then t=0; fi
        para_total=$(echo "$para_total + $t" | bc -l)
    done
    para_avg=$(echo "scale=6; $para_total / 5" | bc -l)
    echo "$n,Parallel,$para_avg,$best_bs"
    cd ../..
done
