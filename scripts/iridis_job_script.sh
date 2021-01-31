#!/bin/bash
# Set job requirements

#SBATCH -J TradIterCopyScore
#SBATCH -t 10:00:00
#SBATCH -o o_%j.%x
#SBATCH -e e_%j.%x
#SBATCH -p batch
#SBATCH --ntasks-per-node=20
#SBATCH --mail-type=END
#SBATCH --mail-user=henrydayhall@pm.com
#============================ PBS part ends =====================



cd $SLURM_SUBMIT_DIR

source activate myenv


export last_journal="/home/hadh1g17/jets/logs/Journal004.jnl"
export jet_class="SpectralFull"
echo "running $SLURM_JOB_NAME"
#mpirun -np 20 ipython3 scripts/mpi_cluster_run.py
#ipython3 scripts/run_scan_score.py SpectralFull 12000

#echo ipython3 scripts/run_copy_jets.py SpectralFull 12000 nlo 1
#ipython3 scripts/run_copy_jets.py SpectralFull 12000 nlo 1



echo ipython3 scripts/run_copy_jets.py Traditional 12000 lo 4
ipython3 scripts/run_copy_jets.py Traditional 12000 lo 4
echo ipython3 scripts/run_copy_jets.py Traditional 12000 nlo 4
ipython3 scripts/run_copy_jets.py Traditional 12000 nlo 4
echo ipython3 scripts/run_copy_jets.py IterativeCone 12000 nlo 4
ipython3 scripts/run_copy_jets.py IterativeCone 12000 nlo 4
echo ipython3 scripts/run_copy_jets.py IterativeCone 12000 lo 4
ipython3 scripts/run_copy_jets.py IterativeCone 12000 lo 4

#echo "May need to up the duration in mpi_cluster_run.py"


