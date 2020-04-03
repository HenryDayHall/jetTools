from tree_tagger import FormJets, Components, InputTools, CompareClusters
import csv
import tabulate
import time
import os
import numpy as np
import multiprocessing
#from ipdb import set_trace as st
import debug

def worker(eventWise_path, run_condition, cluster_algorithm, cluster_parameters, batch_size):
    """
    

    Parameters
    ----------
    eventWise_path :
        param run_condition:
    cluster_algorithm :
        param cluster_parameters:
    batch_size :
        
    run_condition :
        
    cluster_parameters :
        

    Returns
    -------

    """
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
    """
    make n fragments, splitting of unfinished components as needed

    Parameters
    ----------
    eventWise_path :
        param n_fragments:
    jet_name :
        
    n_fragments :
        

    Returns
    -------

    """
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
        # remove a joined component if it exists
        in_eventWise_path = os.listdir(eventWise_path)
        # if it has alredy been joined erase that
        try:
            joined_name = next(name for name in in_eventWise_path if name.endswith("joined.awkd"))
            os.remove(os.path.join(eventWise_path, joined_name))
            print("Removed previous joined component")
        except (StopIteration, OSError) as e:
            pass
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


def generate_pool(eventWise_path, jet_class, jet_params, leave_one_free=False):
    """
    

    Parameters
    ----------
    eventWise_path :
        param multiapply_function:
    jet_params :
        param leave_one_free: (Default value = False)
    jet_class :
        
    leave_one_free :
         (Default value = False)

    Returns
    -------

    """
    class_to_function = {"HomeJet": "Traditional",
                         "HomeInvarientJet": "TraditionalInvarient",
                         "SpectralJet": "Spectral",
                         "SpectralMeanJet": "SpectralMean",
                         "SpectralMAfterJet": "SpectralMAfter",
                         "SpectralFullJet": "SpectralFull",
                         "SpectralAfterJet": "SpectralAfter"}
    multiapply_function = class_to_function[jet_class]
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
    wait_time = 3*60  # in seconds
    # note that the longest wait will be n_cores time this time
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
        job.join(wait_time)
    # check they all stopped
    stalled = [job.is_alive() for job in job_list]
    if np.any(stalled):
        # stop everything
        for job in job_list:
            job.terminate()
            job.join()
        remove_partial(all_paths, jet_params['jet_name'])
        print(f"Problem in {sum(stalled)} out of {len(stalled)} threads")
        with open("problem_jet_params.log", 'a') as log_file:
            log_file.write(str(jet_params) + '\n')
        return False
    print("All processes ended")
    return True


def remove_partial(all_paths, jet_name):
    """
    

    Parameters
    ----------
    all_paths :
        param jet_name:
    jet_name :
        

    Returns
    -------

    """
    for ew_name in all_paths:
        ew = Components.EventWise.from_file(ew_name)
        rewrite = False
        for name in ew.columns:
            if name.startswith(jet_name):
                rewrite = True
                ew.remove(name)
        if rewrite:
            ew.write()


def recombine_eventWise(eventWise_path):
    """
    

    Parameters
    ----------
    eventWise_path :
        

    Returns
    -------

    """
    split_dir = eventWise_path[:-5]+"_fragment"
    if not os.path.exists(split_dir):
        return Components.EventWise.from_file(eventWise_path)
    in_split_dir = os.listdir(split_dir)
    # if it has alredy been joined erase that
    try:
        joined_name = next(name for name in in_split_dir if name.endswith("joined.awkd"))
        os.remove(os.path.join(split_dir, joined_name))
    except (StopIteration, OSError) as e:
        pass
    base_name = os.path.split(eventWise_path)[1].split(".")[0]
    new_eventWise = Components.EventWise.combine(split_dir, base_name)
    return new_eventWise


# In deltaR
# optimum 0.5 for Homejet, 2 for HomeJet Invarient, 1.5 for SpectralFull jet, 0.2 otherwise
# range from 0 to 2
# good varience

# in Exponent multiplier
# optimum -1 for Invarient, 0 otherwise
# vary between -1 and 1
# key varience

# with laplacien scaling
# optimum is no

# affinity type
# optimum is inverse, exponent2

# affinity cutoff
# optimum is None, or knn 1
# try low valued distance (0-4) low value knn (0-4)

# number eigenvectors
# optimum 4

# laplacin
# optimum is unnormalised

# jet class
# optimum is spectralFull
# spectral has been changed to try that too

# scan0 = deltaR + ExponentMultiplier + Invarience
def scan0(eventWise_path):
    """
    

    Parameters
    ----------
    eventWise_path :
        

    Returns
    -------

    """
    record_path = "scans.csv"
    records = CompareClusters.Records(record_path)
    # prepare to check for existing entries
    epsilon = 0.01
    initial_typed = records.typed_array()
    initial_jet_class = initial_typed[:, records.indices['jet_class']]
    initial_DeltaR    = initial_typed[:, records.indices['DeltaR']]
    initial_Exponent  = initial_typed[:, records.indices['ExponentMultiplier']]
    initial_Invarient = initial_typed[:, records.indices['Invarient']]
    eventWise = Components.EventWise.from_file(eventWise_path)
    cols = [c for c in eventWise.columns]
    del eventWise
    DeltaR = np.linspace(0.1, 2., 20)
    exponents = np.linspace(-1, 1, 11)
    invarience = [True, False]
    for exponent in exponents:
        exp_indices = np.where(np.abs(initial_Exponent - exponent) < epsilon)[0]
        for dR in DeltaR:
            dr_indices = exp_indices[np.abs(initial_DeltaR[exp_indices] - dR) < epsilon]
            for invar in invarience:
                invar_indices = dr_indices[initial_Invarient[dr_indices] == invar]
                print(f"Exponent {exponent}")
                print(f"DeltaR {dR}")
                jet_class = "HomeJet"
                if jet_class not in initial_jet_class[invar_indices]:
                    jet_params = dict(DeltaR=dR, ExponentMultiplier=exponent, Invarient=invar)
                    jet_id = records.append(jet_class, jet_params)
                    jet_params["jet_name"] = jet_class + str(jet_id)
                    generate_pool(eventWise_path, jet_class, jet_params, True)
                else:
                    print(f"Already done {jet_class}")
                jet_class = "SpectralFullJet"
                if jet_class not in initial_jet_class[invar_indices]:
                    jet_params = dict(DeltaR=dR, ExponentMultiplier=exponent,
                                      NumEigenvectors=4,
                                      Laplacien='unnormalised',
                                      AffinityType='inverse',
                                      AffinityCutoff=None,
                                      Invarient=invar)
                    jet_id = records.append(jet_class, jet_params)
                    jet_params["jet_name"] = jet_class + str(jet_id)
                    generate_pool(eventWise_path, jet_class, jet_params, True)
                else:
                    print(f"Already done {jet_class}")
    records.write()
# extended
# scan0 = deltaR + ExponentMultiplier
def scan00(eventWise_path):
    """
    

    Parameters
    ----------
    eventWise_path :
        

    Returns
    -------

    """
    record_path = "scans.csv"
    records = CompareClusters.Records(record_path)
    # prepare to check for existing entries
    epsilon = 0.01
    initial_typed = records.typed_array()
    initial_jet_class = initial_typed[:, records.indices['jet_class']]
    initial_DeltaR    = initial_typed[:, records.indices['DeltaR']]
    initial_Exponent  = initial_typed[:, records.indices['ExponentMultiplier']]
    eventWise = Components.EventWise.from_file(eventWise_path)
    initial_Invarient = initial_typed[:, records.indices['Invarient']]
    cols = [c for c in eventWise.columns]
    del eventWise
    DeltaR = np.linspace(0.1, 2., 20)
    exponents = np.linspace(-1, 1, 11)
    invar = "Luclus"
    for exponent in exponents:
        exp_indices = np.where(np.abs(initial_Exponent - exponent) < epsilon)[0]
        for dR in DeltaR:
            dr_indices = exp_indices[np.abs(initial_DeltaR[exp_indices] - dR) < epsilon]
            invar_indices = dr_indices[initial_Invarient[dr_indices] == invar]
            print(f"Exponent {exponent}")
            print(f"DeltaR {dR}")
            jet_class = "HomeJet"
            if jet_class not in initial_jet_class[invar_indices]:
                jet_params = dict(DeltaR=dR, ExponentMultiplier=exponent, Invarient=invar)
                jet_id = records.append(jet_class, jet_params)
                jet_params["jet_name"] = jet_class + str(jet_id)
                generate_pool(eventWise_path, jet_class, jet_params, True)
            else:
                print(f"Already done {jet_class}")
            jet_class = "SpectralFullJet"
            if jet_class not in initial_jet_class[invar_indices]:
                jet_params = dict(DeltaR=dR, ExponentMultiplier=exponent,
                                  NumEigenvectors=4,
                                  Laplacien='unnormalised',
                                  AffinityType='inverse',
                                  AffinityCutoff=None,
                                  Invarient=invar)
                jet_id = records.append(jet_class, jet_params)
                jet_params["jet_name"] = jet_class + str(jet_id)
                generate_pool(eventWise_path, jet_class, jet_params, True)
            else:
                print(f"Already done {jet_class}")
    records.write()
# results; exponentMultipler small pos for all spectral
#          exponentMultipler small neg for homeJet
#          deltaR lesss than 0.5 for spectral 
#          deltaR 0.3 to 0.6 for homeJet

# scan1 = deltaR + jet class
def scan1(eventWise_path):
    """
    

    Parameters
    ----------
    eventWise_path :
        

    Returns
    -------

    """
    record_path = "scans.csv"
    records = CompareClusters.Records(record_path)
    eventWise = Components.EventWise.from_file(eventWise_path)
    cols = [c for c in eventWise.columns]
    del eventWise
    exponents = np.linspace(-0.4, 0.4, 3)
    jet_classes = list(FormJets.cluster_classes.keys())
    jet_classes.remove("FastJet")
    jet_classes.remove("SpectralFullJet")  # we already seen this
    jet_classes.remove("HomeJet")  # we already seen this
    invarients = ['Luclus', 'invarient']
    DeltaR = np.linspace(0.1, 1., 10)
    for jet_class in jet_classes:
        for dR in DeltaR:
            for exponent in exponents:
                for invar in invarients:
                    print(f"JetClass {jet_class}, DeltaR {dR}, Exponent {exponent}, Invarient {invar}")
                    jet_params = dict(DeltaR=dR, ExponentMultiplier=exponent,
                                      NumEigenvectors=4,
                                      Laplacien='unnormalised',
                                      AffinityType='inverse',
                                      AffinityCutoff=None,
                                      Invarient=invar)
                    jet_id = records.append(jet_class, jet_params)
                    jet_params["jet_name"] = jet_class + str(jet_id)
                    generate_pool(eventWise_path, jet_class, jet_params, True)
    records.write()
# results SpectralJet, HomeJet and SpectralFullJet are viable 

# scan2 = deltaR + num eigenvectors
def scan2(eventWise_path):
    """
    

    Parameters
    ----------
    eventWise_path :
        

    Returns
    -------

    """
    record_path = "scans.csv"
    records = CompareClusters.Records(record_path)
    eventWise = Components.EventWise.from_file(eventWise_path)
    cols = [c for c in eventWise.columns]
    del eventWise
    exponents = np.linspace(0, 0.4, 2)
    jet_classes = ["SpectralJet", "SpectralFullJet"]
    invarients = {'SpectralJet':['Luclus'], 'SpectralFullJet':['Luclus', 'angular']}
    DeltaR = np.linspace(0.1, 0.5, 5)
    num_eigenvectors = [2, 3, 4, 5, 6, np.inf]
    for jet_class in jet_classes:
        invars = invarients[jet_class]
        for dR in DeltaR:
            for exponent in exponents:
                for invar in invars:
                    for n_eig in num_eigenvectors:
                        print(f"JetClass {jet_class}, DeltaR {dR}, Exponent {exponent}, Invarient {invar}")
                        jet_params = dict(DeltaR=dR, ExponentMultiplier=exponent,
                                          NumEigenvectors=n_eig,
                                          Laplacien='unnormalised',
                                          AffinityType='inverse',
                                          AffinityCutoff=None,
                                          Invarient=invar)
                        jet_id = records.append(jet_class, jet_params)
                        jet_params["jet_name"] = jet_class + str(jet_id)
                        generate_pool(eventWise_path, jet_class, jet_params, True)
    records.write()
# result 4 eigenvectors does apear to be the sweet spot

# scan3 = deltaR + affinity type
def scan3(eventWise_path):  # TODO
    """
    

    Parameters
    ----------
    eventWise_path :
        

    Returns
    -------

    """
    record_path = "scans.csv"
    records = CompareClusters.Records(record_path)
    eventWise = Components.EventWise.from_file(eventWise_path)
    cols = [c for c in eventWise.columns]
    del eventWise
    exponents = np.linspace(0, 0.4, 2)
    jet_classes = ["HomeJet", "SpectralFullJet"]
    invarients = {'HomeJet':['Luclus'], 'SpectralFullJet':['Luclus', 'angular']}
    DeltaR = np.linspace(0.1, 0.5, 5)
    num_eigenvectors = [2, 3, 4, 5, 6, np.inf]
    for jet_class in jet_classes:
        invars = invarients[jet_class]
        for dR in DeltaR:
            for exponent in exponents:
                for invar in invars:
                    for n_eig in num_eigenvectors:
                        print(f"JetClass {jet_class}, DeltaR {dR}, Exponent {exponent}, Invarient {invar}")
                        jet_params = dict(DeltaR=dR, ExponentMultiplier=exponent,
                                          NumEigenvectors=n_eig,
                                          Laplacien='unnormalised',
                                          AffinityType='inverse',
                                          AffinityCutoff=None,
                                          Invarient=invar)
                        jet_id = records.append(jet_class, jet_params)
                        jet_params["jet_name"] = jet_class + str(jet_id)
                        generate_pool(eventWise_path, jet_class, jet_params, True)
    records.write()
# scan5 = deltaR + affinity cutoff
# scan6 = deltaR + laplacien scaling + laplacien

def loops(eventWise_path):
    """
    

    Parameters
    ----------
    eventWise_path :
        

    Returns
    -------

    """
    record_path = "records.csv"
    records = CompareClusters.Records(record_path)
    eventWise = Components.EventWise.from_file(eventWise_path)
    cols = [c for c in eventWise.columns]
    del eventWise
    DeltaR = np.linspace(2., 4., 15)
    exponents = [-1, 0, 1]
    for exponent in exponents:
        for dR in DeltaR:
            print(f"Exponent {exponent}")
            print(f"DeltaR {dR}")
            jet_class = "HomeInvarientJet"
            jet_params = dict(DeltaR=dR, ExponentMultiplier=exponent)
            jet_id = records.append(jet_class, jet_params)
            jet_params["jet_name"] = jet_class + str(jet_id)
            generate_pool(eventWise_path, jet_class, jet_params, True)
    records.write()
    exponents = [-1, 0, 2]
    NumEigenvectors = [1, 3, 4, 6]
    distance = [3.5, 4.5, 5.5]
    for exponent in exponents:
        for dR in DeltaR:
            for n_eig in NumEigenvectors:
                for dis in distance:
                    print(f"Exponent {exponent}")
                    print(f"DeltaR {dR}")
                    print(f"NumEigenvectors {n_eig}")
                    print(f"Distance {dis}")
                    jet_class = "SpectralAfterJet"
                    jet_params = dict(DeltaR=dR, ExponentMultiplier=exponent,
                                      NumEigenvectors=n_eig,
                                      Laplacien='symmetric',
                                      AffinityType='exponent',
                                      AffinityCutoff=('distance', dis))
                    jet_id = records.append(jet_class, jet_params)
                    jet_params["jet_name"] = jet_class + str(jet_id)
                    generate_pool(eventWise_path, jet_class, jet_params, True)
                jet_params = dict(DeltaR=dR, ExponentMultiplier=exponent,
                                  NumEigenvectors=n_eig,
                                  Laplacien='symmetric',
                                  AffinityType='exponent',
                                  AffinityCutoff=None)
                jet_id = records.append(jet_class, jet_params)
                jet_params["jet_name"] = jet_class + str(jet_id)
                generate_pool(eventWise_path, jet_class, jet_params, True)
    records.write()


def iterate(eventWise_path, jet_class):
    """
    

    Parameters
    ----------
    eventWise_path :
        param jet_class:
    jet_class :
        

    Returns
    -------

    """
    eventWise = Components.EventWise.from_file(eventWise_path)
    record_path = "records.csv"
    records = CompareClusters.Records(record_path)
    print("Delete the continue file when you want to stop")
    if not os.path.exists('continue'):
        open('continue', 'a').close()
    count = 0
    while os.path.exists('continue'):
        if count % 10 == 0:
            print("Rescoring")
            combined = recombine_eventWise(eventWise_path)
            records.score(combined)
        next_best = CompareClusters.parameter_step(records, jet_class)
        if next_best is None:
            print("Couldn't find new target, exiting")
            return
        print(f"Next best is {next_best}")
        jet_id = records.append(jet_class, next_best)
        next_best["jet_name"] = jet_class + str(jet_id)
        generate_pool(eventWise_path, jet_class, next_best, True)
        records.write()
        count += 1


def random_parameters(jet_class=None):
    """
    

    Parameters
    ----------
    jet_class :
        Default value = None)

    Returns
    -------

    """
    if jet_class is None:
        jet_classes = ['SpectralMeanJet', 'SpectralFullJet', 'SpectralMAfterJet', 'HomeInvarientJet']
        jet_class = np.random.choice(jet_classes)
    if jet_class in ['SpectralMeanJet', 'SpectralMAfterJet', 'SpectralFullJet']:
        params = {}
        params['DeltaR'] = np.random.uniform(0., 1.5)
        params['ExponentMultiplier'] = np.random.uniform(-1., 1.)
        params['NumEigenvectors'] = np.random.randint(1, 10)
        #affinites = ['exponent', 'exponent2', 'linear', 'inverse']
        affinites = ['exponent', 'exponent2', 'inverse']
        params['AffinityType'] = np.random.choice(affinites)
        laplaciens = ['unnormalised']
        if params['AffinityType'] in ['linear']:
            laplaciens.append('symmetric')
        params['Laplacien'] = np.random.choice(laplaciens)
        params['WithLaplacienScaling'] = np.random.choice([True, False])
        cutofftypes = [None, 'knn', 'distance']
        cutofftype = np.random.choice(cutofftypes)
        if cutofftype is None:
            params['AffinityCutoff'] = cutofftype
        elif cutofftype == 'knn':
            params['AffinityCutoff'] = (cutofftype, np.random.randint(1, 6))
        elif cutofftype == 'distance':
            params['AffinityCutoff'] = (cutofftype, np.random.uniform(0., 10.))
    elif jet_class in ['HomeJet', 'HomeInvarientJet']:
        params = {}
        params['DeltaR'] = np.random.uniform(0., 1.5)
        params['ExponentMultiplier'] = np.random.uniform(-1., 1.)
    else:
        raise NotImplementedError
    return jet_class, params


def monte_carlo(eventWise_path, jet_class=None):
    """
    

    Parameters
    ----------
    eventWise_path :
        param jet_class:  (Default value = None)
    jet_class :
         (Default value = None)

    Returns
    -------

    """
    if jet_class is None:
        change_class = True
    eventWise = Components.EventWise.from_file(eventWise_path)
    record_path = "records.csv"
    records = CompareClusters.Records(record_path)
    print("Delete the continue file when you want to stop")
    if not os.path.exists('continue'):
        open('continue', 'a').close()
    while os.path.exists('continue'):
        if change_class:
            jet_class = None
        jet_class, next_try = random_parameters(jet_class)
        print(f"Next try is {next_try}")
        jet_id = records.append(jet_class, next_try)
        next_try["jet_name"] = jet_class + str(jet_id)
        generate_pool(eventWise_path, jet_class, next_try, True)
        records.write()


def full_run_best(eventWise_path, records, jet_class=None):
    pass  # TODO

if __name__ == '__main__':
    names = FormJets.cluster_classes
    eventWise_path = InputTools.get_file_name("Where is the eventwise of collection fo eventWise? ", '.awkd')
    #loops(eventWise_path)
    #jet_class = InputTools.list_complete("Jet class? ", list(names.keys())).strip()
    #iterate(eventWise_path, jet_class)
    #monte_carlo(eventWise_path)
    scan2(eventWise_path)

