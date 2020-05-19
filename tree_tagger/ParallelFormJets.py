from tree_tagger import FormJets, Components, InputTools, CompareClusters
import csv
import cProfile
import tabulate
import time
import os
import numpy as np
import multiprocessing
from ipdb import set_trace as st
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
    profiling_path = eventWise_path.replace('awkd', 'prof')
    cProfile.runctx("_worker(eventWise_path, run_condition, cluster_algorithm, cluster_parameters, batch_size)", globals(), locals(), profiling_path)


def _worker(eventWise_path, run_condition, cluster_algorithm, cluster_parameters, batch_size):
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
    elif isinstance(run_condition, (int, float)):
        while time.time() < run_condition and not finished:
            print(f"batch {i}", flush=True)
            i+=1
            finished = FormJets.cluster_multiapply(eventWise, cluster_algorithm, cluster_parameters, batch_length=batch_size, silent=True)
    else:
        raise ValueError(f"Dont recognise run_condition {run_condition}")
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


def generate_pool(eventWise_path, jet_class, jet_params, leave_one_free=False, end_time=None):
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
    if end_time is not None:
        run_condition = end_time
    elif os.path.exists('continue'):
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
    wait_time = 30*60  # in seconds
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


def scan_spectralfull(eventWise_path):
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
    if len(initial_typed) == 0:
        initial_typed = np.zeros((0, 30))
        initial_jet_class = np.array([])
        initial_DeltaR    =  np.array([])
        initial_Exponent  =  np.array([])
        initial_Cutoff =  np.array([])
        initial_Invarient =  np.array([])
        initial_NEigen=  np.array([])
    else:
        initial_jet_class = initial_typed[:, records.indices['jet_class']]
        initial_DeltaR    = initial_typed[:, records.indices['DeltaR']]
        initial_Exponent  = initial_typed[:, records.indices['ExpofPTMultiplier']]
        initial_Cutoff = initial_typed[:, records.indices['AffinityCutoff']]
        initial_Invarient = initial_typed[:, records.indices['Invarient']]
        initial_NEigen= initial_typed[:, records.indices['NumEigenvectors']].astype(int)
    eventWise = Components.EventWise.from_file(eventWise_path)
    cols = [c for c in eventWise.columns]
    del eventWise
    DeltaR = np.linspace(0.05, 0.2, 4)
    exponents = np.linspace(0, 0.6, 4)
    affinitycutoff = [('distance', 9), None]
    invarient = ['Luclus', 'normed']
    numeigenvectors = [3, 5, 7]
    for exponent in exponents:
        exp_indices = np.where(np.abs(initial_Exponent - exponent) < epsilon)[0]
        for dR in DeltaR:
            dr_indices = exp_indices[np.abs(initial_DeltaR[exp_indices] - dR) < epsilon]
            for cutoff in affinitycutoff:
                if cutoff is None:
                    co_indices = dr_indices[initial_Cutoff[dr_indices] == cutoff]
                else:
                    co_mask = []
                    for initial in initial_Cutoff[dr_indices]:
                        if isinstance(initial, tuple):
                            co_mask.append(initial[0] == cutoff[0] and
                                           abs(initial[1] - cutoff[1]) < epsilon)
                        else:
                            co_mask.append(False)
                    co_indices = dr_indices[co_mask]
                for invar in invarient:
                    invar_indices = co_indices[initial_Invarient[co_indices] == invar]
                    for eig in numeigenvectors:
                        eig_indices = invar_indices[initial_NEigen[invar_indices] == eig]
                        if (not os.path.exists('continue')) or os.path.exists('stopscan'):
                            return
                        jet_class = "SpectralFullJet"
                        jet_params = dict(DeltaR=dR,
                                          ExpofPTMultiplier=exponent,
                                          ExpofPTPosition='eigenspace',
                                          NumEigenvectors=eig,
                                          Laplacien='symmetric',
                                          AffinityType='exponent2',
                                          AffinityCutoff=cutoff,
                                          Invarient=invar,
                                          StoppingConditon='standard')
                        if jet_class not in initial_jet_class[eig_indices]:
                            print(jet_params)
                            jet_id = records.append(jet_class, jet_params)
                            jet_params["jet_name"] = jet_class + str(jet_id)
                            generate_pool(eventWise_path, jet_class, jet_params, True)
                        else:
                            print(f"Already done {jet_class}, {jet_params}")
    records.write()


def scan_spectralmean(eventWise_path):
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
    if len(initial_typed) == 0:
        initial_typed = np.zeros((0, 30))
        initial_jet_class = np.array([])
        initial_DeltaR    =  np.array([])
        initial_Exponent  =  np.array([])
        initial_Cutoff =  np.array([])
        initial_Invarient =  np.array([])
        initial_NEigen=  np.array([])
    else:
        initial_jet_class = initial_typed[:, records.indices['jet_class']]
        initial_DeltaR    = initial_typed[:, records.indices['DeltaR']]
        initial_Exponent  = initial_typed[:, records.indices['ExpofPTMultiplier']]
        initial_Position  = initial_typed[:, records.indices['ExpofPTPosition']]
        initial_Cutoff = initial_typed[:, records.indices['AffinityCutoff']]
        initial_Invarient = initial_typed[:, records.indices['Invarient']]
        initial_NEigen= initial_typed[:, records.indices['NumEigenvectors']].astype(int)
    eventWise = Components.EventWise.from_file(eventWise_path)
    cols = [c for c in eventWise.columns]
    del eventWise
    position = ['input', 'eigenspace']
    DeltaR = np.linspace(0.2, 0.6, 4)
    affinitycutoff = [('distance', 2), None]
    invarient = ['invarient', 'angular', 'Luclus']
    numeigenvectors = [2, 4, 6]
    time = time.time()
    runtime = 3*60*60
    endtime = time+rumtime
    for pos in position:
        pos_indices = np.where(initial_Position==pos)[0]
        for dR in DeltaR:
            dr_indices = pos_indices[np.abs(initial_DeltaR[pos_indices] - dR) < epsilon]
            for cutoff in affinitycutoff:
                if cutoff is None:
                    co_indices = dr_indices[initial_Cutoff[dr_indices] == cutoff]
                else:
                    co_mask = []
                    for initial in initial_Cutoff[dr_indices]:
                        if isinstance(initial, tuple):
                            co_mask.append(initial[0] == cutoff[0] and
                                           abs(initial[1] - cutoff[1]) < epsilon)
                        else:
                            co_mask.append(False)
                    co_indices = dr_indices[co_mask]
                for invar in invarient:
                    invar_indices = co_indices[initial_Invarient[co_indices] == invar]
                    for eig in numeigenvectors:
                        eig_indices = invar_indices[initial_NEigen[invar_indices] == eig]
                        if (not os.path.exists('continue')) or os.path.exists('stopscan'):
                            return
                        if time.time() > endtime:
                            return
                        jet_class = "SpectralFullJet"
                        jet_params = dict(DeltaR=dR,
                                          ExpofPTMultiplier=-0.2,
                                          ExpofPTPosition=pos,
                                          NumEigenvectors=eig,
                                          Laplacien='symmetric',
                                          AffinityType='exponent2',
                                          AffinityCutoff=cutoff,
                                          Invarient=invar,
                                          StoppingConditon='beamparticle')
                        if jet_class not in initial_jet_class[eig_indices]:
                            print(jet_params)
                            jet_id = records.append(jet_class, jet_params)
                            jet_params["jet_name"] = jet_class + str(jet_id)
                            generate_pool(eventWise_path, jet_class, jet_params, True)
                        else:
                            print(f"Already done {jet_class}, {jet_params}")
    records.write()


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
        jet_classes = ['SpectralMeanJet', 'SpectralFullJet', 'HomeJet']
        jet_class = np.random.choice(jet_classes)
    if 'Spectral' in jet_class:
        permitted = FormJets.Spectral.permited_values
        params = {}
        params['DeltaR'] = np.random.uniform(0., 1.5)
        params['ExpofPTMultiplier'] = np.random.uniform(-1., 1.)
        exppos = permitted['ExpofPTPosition']
        params['ExpofPTPosition'] = np.random.choice(exppos)
        params['NumEigenvectors'] = np.random.randint(1, 10)
        laplaciens = permitted['Laplacien']
        params['Laplacien'] = np.random.choice(laplaciens)
        affinites = permitted['AffinityType']
        params['AffinityType'] = np.random.choice(affinites)
        cutofftypes = [None, 'knn', 'distance']
        cutofftype = np.random.choice(cutofftypes)
        if cutofftype is None:
            params['AffinityCutoff'] = cutofftype
        elif cutofftype == 'knn':
            params['AffinityCutoff'] = (cutofftype, np.random.randint(1, 6))
        elif cutofftype == 'distance':
            params['AffinityCutoff'] = (cutofftype, np.random.uniform(0., 10.))
        invarients = permitted['Invarient']
        params['Invarient'] = np.random.choice(invarients)
        stopcon = permitted['StoppingCondition']
        params['StoppingCondition'] = np.random.choice(stopcon)
    elif 'Home' in jet_class:
        permitted = FormJets.Traditional.permited_values
        params = {}
        params['DeltaR'] = np.random.uniform(0., 1.5)
        params['ExpofPTMultiplier'] = np.random.uniform(-1., 1.)
        invarients = permitted['Invarient']
        params['Invarient'] = np.random.choice(invarients)
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
    change_class = jet_class is None
    eventWise = Components.EventWise.from_file(eventWise_path)
    record_path = "records.csv"
    records = CompareClusters.Records(record_path)
    #print("Delete the continue file when you want to stop")
    #if not os.path.exists('continue'):
    #    open('continue', 'a').close()
    #while os.path.exists('continue'):
    end_time = time.time() + InputTools.get_time("How long to make clusters for? ")
    print(f"Ending at {end_time}")
    while time.time() < end_time:
        if change_class:
            jet_class = None
        jet_class, next_try = random_parameters(jet_class)
        print(f"Next try is {jet_class}; {next_try}")
        jet_id = records.append(jet_class, next_try)
        next_try["jet_name"] = jet_class + str(jet_id)
        generate_pool(eventWise_path, jet_class, next_try, True, end_time=end_time)
        records.write()


def full_run_best(eventWise_path, records, jet_class=None):
    pass  # TODO

if __name__ == '__main__':
    #names = FormJets.cluster_classes
    eventWise_path = InputTools.get_file_name("Where is the eventwise of collection fo eventWise? ", '.awkd')
    #loops(eventWise_path)
    #jet_class = InputTools.list_complete("Jet class? ", list(names.keys())).strip()
    #iterate(eventWise_path, jet_class)
    #monte_carlo(eventWise_path, jet_class = "SpectralFullJet")
    scan_spectralfull(eventWise_path)

