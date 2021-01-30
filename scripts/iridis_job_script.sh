#!/bin/bash
# Set job requirements

#SBATCH -J SpectralFullscanscore
#SBATCH -t 10:00:00
#SBATCH -o o_%j.%x
#SBATCH -e e_%j.%x
#SBATCH -p batch
#SBATCH --ntasks-per-node=20
#SBATCH --mail-type=END
#SBATCH --mail-user=henrydayhall@pm.com
#============================ PBS part ends =====================

source activate myenv


cd $SLURM_SUBMIT_DIR
export last_journal=""
#export jet_class="SpectralKMeans"
echo "running $SLURM_JOB_NAME"
#mpirun -np 20 ipython3 scripts/mpi_cluster_run.py
ipython3 scripts/run_scan_score.py SpectralFull 12000

#echo "May need to up the duration in mpi_cluster_run.py"


