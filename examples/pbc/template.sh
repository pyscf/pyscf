#!/bin/bash
#SBATCH --partition=debug
#SBATCH --time=1:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1

CURRDIR=$SLURM_SUBMIT_DIR
WORKDIR=$SLURM_SUBMIT_DIR

cd $WORKDIR
source /home/jmcclain/module_load.sh

export OMP_NUM_THREADS=1
srun -n 2 -c $OMP_NUM_THREADS --label python -u 23-k_points_mpi_ccsd.py
