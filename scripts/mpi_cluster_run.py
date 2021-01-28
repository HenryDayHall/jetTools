""" Script to run OptimiseCluster under mpi"""
from jet_tools import OptimiseCluster
# it will checkpoint
duration = 100*60*60  # makes sense to run it for longer than any job
log_dir = "../logs"
OptimiseCluster.run_optimisation_abcpy("../megaIgnore/best_v3.awkd", duration=duration,
                                       log_dir=log_dir)
