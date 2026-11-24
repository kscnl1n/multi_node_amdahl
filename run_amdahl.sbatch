#!/bin/bash
#SBATCH --job-name=amdahl
#SBATCH --nodes=4                 # largest number of nodes you want to test
#SBATCH --ntasks-per-node=16      # MPI ranks per node (<= cores per node)
#SBATCH --time=00:30:00
#SBATCH --output=amdahl_%j.out
##SBATCH --partition=<name>       # uncomment and set to your cluster's partition
##SBATCH --account=<name>         # uncomment if your cluster requires an account

# Edit the module line to match your cluster (run `module avail` to see names).
# module load python openmpi
# pip install --user mpi4py        # once, if mpi4py isn't already available

P=${P:-0.90}                       # parallel fraction; override: sbatch --export=ALL,P=0.95 run_amdahl.sbatch
OUT=results_${SLURM_JOB_ID}.csv

# Strong scaling: same problem, more ranks. N=1 is the baseline.
for N in 1 2 4 8 16 32 64; do
  if [ "$N" -le "$SLURM_NTASKS" ]; then
    # Spread ranks across nodes so larger N actually uses more than one node.
    srun --ntasks=$N --nodes=$(( (N + SLURM_NTASKS_PER_NODE - 1) / SLURM_NTASKS_PER_NODE )) \
         python3 amdahl_mpi.py run -p "$P" --out "$OUT"
  fi
done

python3 amdahl_mpi.py report "$OUT"
