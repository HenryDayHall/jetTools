""" Script to run OptimiseCluster under mpi"""
from jet_tools import OptimiseCluster
import os
# it will checkpoint
duration = 100*60*60  # makes sense to run it for longer than any job
log_dir = "../logs"
# some kwargs will be read as shell varaibles, if they exist
kwargs = {}
last_journal = os.environ["last_journal"]
if os.path.exists(last_journal):
    print(f"Starting from {last_journal}")
    kwargs["last_journal"] = last_journal
else:
    print("Run from scratch")

OptimiseCluster.run_optimisation_abcpy("../megaIgnore/best_v3.awkd", duration=duration,
                                       log_dir=log_dir, **kwargs)
