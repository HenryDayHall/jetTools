#!/bin/bash
# Set job requirements

#SBATCH -J abcpy1
#SBATCH -t 10:00:00
#SBATCH -o o_%j.%x
#SBATCH -e e_%j.%x
#SBATCH -p ngcm
#SBATCH --ntasks-per-node=20
#SBATCH --mail-type=END
#SBATCH --mail-user=henrydayhall@pm.com
#============================ PBS part ends =====================



cd $SLURM_SUBMIT_DIR


echo "running $SLURM_JOB_NAME"
mpirun -np 20 ipython3 scripts/mpi_cluster_run.py

echo "May need to up the duration in mpi_cluster_run.py"


