from tree_tagger import FormJets, Components, InputTools, CompareClusters
import csv
import tabulate
import time
import os
import numpy as np
import multiprocessing
from ipdb import set_trace as st

def worker(eventWise_path, run_condition, cluster_algorithm, cluster_parameters, batch_size):
    if isinstance(cluster_algorithm, str):
        # functions in modules are attributes too :)
        cluster_algorithm = getattr(FormJets, cluster_algorithm)
    eventWise = Components.EventWise.from_file(eventWise_path)
    print(eventWise.dir_name)
    i = 0
    finished = False
    if run_condition is 'continue':
        while os.path.exists('continue') and not finished:
            print(f"batch {i}", flush=True)
            i+=1
            finished = FormJets.cluster_multiapply(eventWise, cluster_algorithm, cluster_parameters, batch_length=batch_size, silent=True)
    elif isinstance(run_condition, int):
        while time.time() < run_condition and not finished:
            print(f"batch {i}", flush=True)
            i+=1
            finished = FormJets.cluster_multiapply(eventWise, cluster_algorithm, cluster_parameters, batch_length=batch_size, silent=True)
    if finished:
        print(f"Finished {i} batches, dataset {eventWise_path} complete")
    else:
        print(f"Finished {i} batches, dataset {eventWise_path} incomplete")


def make_n_working_fragments(eventWise_path, n_fragments, jet_name):
    """ make n fragments, splitting of unfinished components as needed """
    # if an awkd file is given, and a progress directory exists, change to that
    if eventWise_path.endswith('awkd') and os.path.exists(eventWise_path[:-5]+"_progress"):
        print("This awkd has already been split into progress")
        eventWise_path = eventWise_path[:-5]+"_progress"
    # same logic to fragment
    if eventWise_path.endswith('awkd') and os.path.exists(eventWise_path[:-5]+"_fragment"):
        print("This awkd has already been split into fragments")
        eventWise_path = eventWise_path[:-5]+"_fragment"
    if not eventWise_path.endswith('.awkd'):  # this is probably a dir name
        if '.' in eventWise_path:
            raise ValueError(f"eventWise_path {eventWise_path} is neither a directory name not the path to an eventWise")
        print(f"{eventWise_path} appears to be directory")
        # if it's a directory look for subdirectories whos name starts with the directory name
        # these indicate existing splits
        leaf_dir = os.path.split(eventWise_path)[-1]
        sub_dir = [name for name in os.listdir(eventWise_path)
                   if name.startswith(leaf_dir) and os.path.isdir(os.path.join(eventWise_path, name))]
        while sub_dir:
            print(f"Entering {sub_dir[0]}")
            eventWise_path = os.path.join(sub_dir[0])
            sub_dir = [name for name in os.listdir(eventWise_path)
                       if name.startswith(sub_dir[0]) and os.path.isdir(os.path.join(eventWise_path, name))]
        existing_fragments = [name for name in os.listdir(eventWise_path)
                              if name.endswith(".awkd")]
        if len(existing_fragments) == 0:
            raise RuntimeError(f"Directory {eventWise_path} has no eventWise file in")
        elif len(existing_fragments) == 1:
            print("Path contains one eventWise")
            eventWise_path = os.path.join(eventWise_path, existing_fragments[0])
        elif len(existing_fragments) == n_fragments:
            print("Path already contains correct number of eventWise. (may be semicomplete)")
            # the correct number of fragments already exists
            all_paths = [os.path.join(eventWise_path, fragment) for fragment in existing_fragments]
            return all_paths
        else:  #there are a number of fragments, not equal to the desired number
            print(f"Path contains {len(existing_fragments)} eventWise.")
            print("Extracting unfinished components")
            unfinished_fragments = []
            finished_fragments = []
            jet_components = None
            for fragment in existing_fragments:
                fragment_path = os.path.join(eventWise_path, fragment)
                ew_fragment = Components.EventWise.from_file(fragment_path)
                if jet_components is None:
                    jet_components = [name for name in ew_fragment.columns if name.startswith(jet_name)]
                finished_path, unfinished_path = ew_fragment.split_unfinished("JetInputs_Energy", jet_components)
                if unfinished_path is not None:
                    unfinished_fragments.append(os.path.split(unfinished_path)[1])
                if finished_path is not None:
                    finished_fragments.append(os.path.split(finished_path)[1])
            if len(unfinished_fragments) == 0:
                print("Everthing is finished")
                # there should nowbe finished fragments
                if len(finished_fragments) > 1:
                    finished_path = Components.EventWise.combine(eventWise_path, "Finished_" + jet_name,
                                                                 fragments=finished_fragments, del_fragments=True)
                return True
            # merge both collections
            print("Creating collective finished and unfinished parts")
            if len(finished_fragments) > 0:
                finished_path = Components.EventWise.combine(eventWise_path, "Finished_" + jet_name,
                                                             fragments=finished_fragments, del_fragments=True)
            unfinished_path = Components.EventWise.combine(eventWise_path, "temp", 
                                                           fragments=unfinished_fragments, del_fragments=True)
            eventWise_path = unfinished_path
    # if this point is reached all the valid events are in one eventwise at eventWise_path 
    print(f"In possesion of one eventWise object at {eventWise_path}")
    eventWise = Components.EventWise.from_file(eventWise_path)
    # check if the jet is in the eventWise
    jet_components = [name for name in eventWise.columns if name.startswith(jet_name)]
    if len(jet_components) > 0:
        print(f"eventWise at {eventWise_path} is partially completed, seperating completed component")
        finished_path, unfinished_path = eventWise.split_unfinished("JetInputs_Energy", jet_components)
        if unfinished_path is None:
            print("Everything is finished")
            return True
        eventWise = Components.EventWise.from_file(unfinished_path)
    print("Fragmenting eventwise")
    all_paths = eventWise.fragment("JetInputs_Energy", n_fragments=n_fragments)
    return all_paths


def generate_pool(eventWise_path, multiapply_function, jet_params, leave_one_free=False):
    batch_size = 500
    # decide on a stop condition
    if os.path.exists('continue'):
        run_condition = 'continue'
    else:
        if InputTools.yesNo_question("Would you like to do a time based run? "):
            run_time = InputTools.get_time("How long should it run?")
            stop_time = time.time() + run_time
            run_condition = stop_time
        elif InputTools.yesNo_question("Would you like to create a continue file?"
                                       +" (be sure you can delete it while the"
                                       +" program is running!) "):
            open('continue', 'w').close()
            run_condition = 'continue'
        else:
            return 
    profile_start_time = time.time()
    # work out how many threads
    # cap this out at 20, more seems to create a performance hit
    n_threads = min(multiprocessing.cpu_count()-leave_one_free, 20)
    if n_threads < 1:
        n_threads = 1
    print("Running on {} threads".format(n_threads))
    jet_name = jet_params['jet_name']
    all_paths = make_n_working_fragments(eventWise_path, n_threads, jet_name)
    job_list = []
    # now each segment makes a worker
    args = [(path, run_condition, multiapply_function, jet_params, batch_size)
            for path in all_paths]
    for a in args:
        job = multiprocessing.Process(target=worker, args=a)
        job.start()
        job_list.append(job)
    for job in job_list:
        job.join()
    print("All processes ended")


if __name__ == '__main__':
    eventWise_path = InputTools.get_file_name("Where is the eventwise of collection fo eventWise? ", '.awkd')
    record_path = "records.csv"
    records = CompareClusters.Records(record_path)
    eventWise = Components.EventWise.from_file(eventWise_path)
    cols = [c for c in eventWise.columns]
    del eventWise
    DeltaR = np.linspace(0.1, 1.9, 10)
    exponents = [-0.5, 0, 0.5]
    #for exponent in exponents:
    #    for dR in DeltaR:
    #        print(f"Exponent {exponent}")
    #        print(f"DeltaR {dR}")
    #        jet_class = "HomeJet"
    #        jet_params = dict(DeltaR=dR, ExponentMultiplier=exponent)
    #        jet_id = records.append(jet_class, jet_params)
    #        jet_params["jet_name"] = jet_class + str(jet_id)
    #        generate_pool(eventWise_path, 'Traditional', jet_params, True)
    #records.write()
    exponents = [-1, 0, 1]
    NumEigenvectors = [8, 20, np.inf]
    for exponent in exponents:
        for dR in DeltaR:
            for n_eig in NumEigenvectors:
                print(f"Exponent {exponent}")
                print(f"DeltaR {dR}")
                print(f"NumEigenvectors {n_eig}")
                jet_class = "SpectralMeanJet"
                jet_params = dict(DeltaR=dR, ExponentMultiplier=exponent,
                                  NumEigenvectors=n_eig,
                                  Laplacien='symmetric',
                                  AffinityType='linear',
                                  AffinityCutoff=('knn', 7))
                jet_id = records.append(jet_class, jet_params)
                jet_params["jet_name"] = jet_class + str(jet_id)
                generate_pool(eventWise_path, 'SpectralMean', jet_params, True)
                jet_class = "SpectralJet"
                jet_params = dict(DeltaR=dR, ExponentMultiplier=exponent,
                                  NumEigenvectors=n_eig,
                                  Laplacien='symmetric',
                                  AffinityType='linear',
                                  AffinityCutoff=('knn', 7))
                jet_id = records.append(jet_class, jet_params)
                jet_params["jet_name"] = jet_class + str(jet_id)
                generate_pool(eventWise_path, 'Spectral', jet_params, True)
    records.write()

