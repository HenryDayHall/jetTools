""" Script to run OptimiseCluster under mpi"""
from jet_tools import OptimiseCluster
duration = 100*60*60
log_dir = "../logs_test"
OptimiseCluster.run_optimisation_abcpy("../megaIgnore/best_v3.awkd", duration=duration,
                                       log_dir=log_dir)
