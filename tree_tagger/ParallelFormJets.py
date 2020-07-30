from tree_tagger import FormJets, Components, InputTools, CompareClusters
import time
import csv
import cProfile
import tabulate
import os
import numpy as np
import multiprocessing
from ipdb import set_trace as st


def make_jet_name(jet_class, jet_id):
    """
    Form a standardised jet name from the name of the
    class and the id.

    Parameters
    ----------
    jet_class : str
        name of the jet class
    jet_id : int
        id of this clustering

    Returns
    -------
    : str
        standardised name

    """
    return jet_class + "Jet" + str(int(jet_id))


def worker(eventWise_path, run_condition, cluster_algorithm, cluster_parameters, batch_size):
    """
    Runs a worker, with profiling.
    Same inputs as _worker, see _worker's docstring.
    """
    profiling_path = eventWise_path.replace('awkd', 'prof')
    cProfile.runctx("_worker(eventWise_path, run_condition, cluster_algorithm, cluster_parameters, batch_size)", globals(), locals(), profiling_path)


def _worker(eventWise_path, run_condition, jet_class, cluster_parameters, batch_size):
    """
    A worker to cluster jets in one process.
    Not thread safe with respect to the eventWise file,
    the eventWise file should be unique to this worker,
    if multiple workers are needed the eventWise file needs to
    be split ahead of time.

    Parameters
    ----------
    eventWise_path : str
        File path the the eventWise file that data
        should be read and written to.
    run_condition: str, int or float
        If run condition is the string "continue"
        the worker will continue clustering so long
        as a file called continue is in the same directory.
        If the run condition is a float or an int
        then the cluster algorithm will continur for the
        number of seconds equal to this number.
    jet_class : str or callable
        The algorithm to do the clustering.
        If it's a string it is the algorithms name
        in the module FormJets
    cluster_parameters : dict
        Dictionary of parameters to be given to the
        clustering algorithm.
    batch_size : int
        Number of events to cluster in one jump,
        higher numbers speed up the process,
        but a batch is not interuptable,
        so they also reduce the precision of the stopping
        condition.
    
    """
    if isinstance(jet_class, str):
        # functions in modules are attributes too :)
        jet_class = getattr(FormJets, jet_class)
    eventWise = Components.EventWise.from_file(eventWise_path)
    print(eventWise.dir_name)
    i = 0
    finished = False
    if run_condition == 'continue':
        while os.path.exists('continue') and not finished:
            print(f"batch {i}", flush=True)
            i+=1
            finished = FormJets.cluster_multiapply(eventWise, jet_class, cluster_parameters, batch_length=batch_size, silent=True)
    elif isinstance(run_condition, (int, float)):
        while time.time() < run_condition and not finished:
            print(f"batch {i}", flush=True)
            i+=1
            finished = FormJets.cluster_multiapply(eventWise, jet_class, cluster_parameters, batch_length=batch_size, silent=True)
    else:
        raise ValueError(f"Dont recognise run_condition {run_condition}")
    if finished:
        print(f"Finished {i} batches, dataset {eventWise_path} complete")
    else:
        print(f"Finished {i} batches, dataset {eventWise_path} incomplete")


def make_n_working_fragments(eventWise_path, n_fragments, jet_name):
    """
    Make n unfinished fragments, recombining
    and splitting the unfinished components as needed.
    Normally for multithreaded processing.

    Parameters
    ----------
    eventWise_path : str
        path to the dataset to be split, can be 
        an awkd file, or a directory containgin a number of awkd files
    n_fragments : int
        number of fragemtns required.
    jet_name : str
        prefix of the jet variables being wored on in the file

    Returns
    -------
    state : bool of list of str
        is everything is finished returns True
        else returns a list of paths to the unfinihed fragments
    """
    # if an awkd file is given, and a progress directory exists, change to that
    if eventWise_path.endswith('awkd') and os.path.exists(eventWise_path[:-5]+"_progress"):
        print("This awkd has already been split into progress")
        eventWise_path = eventWise_path[:-5]+"_progress"
        # inside this directory, whatever is called progress1 is the thing we want
        # becuase that is the unfinished part
        unfinished_part = next(name for name in os.listdir(eventWise_path)
                              if 'progress1' in name)
        eventWise_path = os.path.join(eventWise_path, unfinished_part)
    # same logic to fragment
    if eventWise_path.endswith('awkd') and os.path.exists(eventWise_path[:-5]+"_fragment"):
        print("This awkd has already been split into fragments")
        eventWise_path = eventWise_path[:-5]+"_fragment"
    if not eventWise_path.endswith('.awkd'):  # this is probably a dir name
        if '.' in eventWise_path:
            raise FileNotFoundError(f"eventWise_path {eventWise_path} is neither a directory name not the path to an eventWise")
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
            raise FileNotFoundError(f"Directory {eventWise_path} has no eventWise file in")
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
                    finished_path = Components.EventWise.combine(eventWise_path, "finished_" + jet_name,
                                                                 fragments=finished_fragments, del_fragments=True)
                return True
            # merge both collections and move them back up a layer
            eventWise_path = eventWise_path.rstrip(os.sep)
            new_path = os.sep.join(eventWise_path.split(os.sep)[:-1])
            print("Creating collective finished and unfinished parts")
            if len(finished_fragments) > 0:
                finished = Components.EventWise.combine(eventWise_path, "finished_" + jet_name,
                                                        fragments=finished_fragments, del_fragments=True)
                os.rename(os.path.join(eventWise_path, finished.save_name),
                          os.path.join(new_path, finished.save_name).replace('_joined', ''))
            unfinished = Components.EventWise.combine(eventWise_path, "remaning_"+jet_name, 
                                                      fragments=unfinished_fragments, del_fragments=True)
            os.rename(os.path.join(eventWise_path, unfinished.save_name),
                      os.path.join(new_path, unfinished.save_name).replace('_joined', ''))
            unfinished.save_name = unfinished.save_name.replace("_joined", "")
            # then remove the old directory
            os.rmdir(eventWise_path)
            eventWise_path = os.path.join(new_path, unfinished.save_name)
    # if this point is reached all the valid events are in one eventwise at eventWise_path 
    print(f"In possesion of one eventWise object at {eventWise_path}")
    eventWise = Components.EventWise.from_file(eventWise_path)
    # check if the jet is in the eventWise
    jet_components = [name for name in eventWise.columns if name.startswith(jet_name)]
    unfinished_path = None
    if len(jet_components) > 0:
        print(f"eventWise at {eventWise_path} is partially completed, seperating completed component")
        finished_path, unfinished_path = eventWise.split_unfinished("JetInputs_Energy", jet_components)
        if unfinished_path is None:
            print("Everything is finished")
            return True
        eventWise = Components.EventWise.from_file(unfinished_path)
    print("Fragmenting eventwise")
    all_paths = eventWise.fragment("JetInputs_Energy", n_fragments=n_fragments)
    if unfinished_path is not None:
        # get rid of the unfishied part becuase it exists in the fragments already
        os.remove(unfinished_path)
    return all_paths


def generate_pool(eventWise_path, jet_class, jet_params, jet_name, leave_one_free=False, end_time=None):
    """
    Split the input file and create a pool of workers each with their own process
    to cluster the required jets on the split input file.

    Parameters
    ----------
    eventWise_path : str
        path to the dataset to be split, can be 
        an awkd file, or a directory containgin a number of awkd files
    jet_class : str or callable
        The algorithm to do the clustering.
        If it's a string it is the algorithms name
        in the module FormJets
    jet_params : dict
        Dictionary of parameters to be given to the
        clustering algorithm.
    jet_name : str
        prefix of the jet variables being worked on in the file
    leave_one_free : bool
        should one core be left free so the computer remains responsive?
        (Default value = False)
    end_time : int or float
        max time to run the processes for
        if None and no continue file exists
        then the user will be asked to give a number
         (Default value = None)

    Returns
    -------
    : bool
        Did all the jobs run without stalling
    
    """
    # read from FormJets
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
    all_paths = make_n_working_fragments(eventWise_path, n_threads, jet_name)
    job_list = []
    # now each segment makes a worker
    args = [(path, run_condition, jet_class, jet_params, batch_size)
            for path in all_paths]
    for a in args:
        job = multiprocessing.Process(target=_worker, args=a)
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
        remove_partial(all_paths, jet_name)
        print(f"Problem in {sum(stalled)} out of {len(stalled)} threads")
        with open("problem_jet_params.log", 'a') as log_file:
            log_file.write(str(jet_params) + '\n')
        return False
    print("All processes ended")
    return True


# prevents stalled jets from causing issues
def remove_partial(all_paths, jet_name):
    """
    Remove a jet from all specified paths, optimising for the case
    where this jet may not exist in many of the jets.

    Parameters
    ----------
    all_paths : list of str
        list of fiel name sof the eventWise datasets
    jet_name : str
        prefix of the jet variables being removed from the files
    
    """
    if not jet_name.endswith('_'):
        jet_name += '_'  # to ensure that two jets that start with the same string
        # cannot be confused
    for ew_name in all_paths:
        ew = Components.EventWise.from_file(ew_name)
        rewrite = any(name.startswith(jet_name) for name in ew.columns)
        ew.remove_prefix(jet_name)
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


def scan_spectralfull(eventWise_path, end_time):
    """
    

    Parameters
    ----------
    eventWise_path :
        
    end_time :
        

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
        initial_PhyDistance =  np.array([])
        initial_NEigen=  np.array([])
    else:
        initial_jet_class = initial_typed[:, records.indices['jet_class']]
        initial_DeltaR    = initial_typed[:, records.indices['DeltaR']]
        initial_Exponent  = initial_typed[:, records.indices['ExpofPTMultiplier']]
        initial_Cutoff = initial_typed[:, records.indices['AffinityCutoff']]
        initial_PhyDistance = initial_typed[:, records.indices['Invarient']]
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
                    invar_indices = co_indices[initial_PhyDistance[co_indices] == invar]
                    for eig in numeigenvectors:
                        eig_indices = invar_indices[initial_NEigen[invar_indices] == eig]
                        if (not os.path.exists('continue')) or os.path.exists('stopscan'):
                            return
                        if time.time()>end_time:
                            return
                        jet_class = "SpectralFull"
                        jet_params = dict(DeltaR=dR,
                                          ExpofPTMultiplier=exponent,
                                          ExpofPTPosition='eigenspace',
                                          NumEigenvectors=eig,
                                          Laplacien='symmetric',
                                          AffinityType='exponent2',
                                          AffinityCutoff=cutoff,
                                          PhyDistance=invar,
                                          StoppingConditon='standard')
                        if jet_class not in initial_jet_class[eig_indices]:
                            print(jet_params)
                            jet_id = records.append(jet_class, jet_params)
                            jet_name = make_jet_name(jet_class, jet_id)
                            generate_pool(eventWise_path, jet_class, jet_params, jet_name, True)
                        else:
                            print(f"Already done {jet_class}, {jet_params}")
    records.write()


def scan_spectralmean(eventWise_path, end_time):
    """
    

    Parameters
    ----------
    eventWise_path :
        
    end_time :
        

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
        initial_Position  =  np.array([])
        initial_Cutoff =  np.array([])
        initial_PhyDistance =  np.array([])
        initial_NEigen=  np.array([])
    else:
        initial_jet_class = initial_typed[:, records.indices['jet_class']]
        initial_DeltaR    = initial_typed[:, records.indices['DeltaR']]
        initial_Exponent  = initial_typed[:, records.indices['ExpofPTMultiplier']]
        initial_Position  = initial_typed[:, records.indices['ExpofPTPosition']]
        initial_Cutoff = initial_typed[:, records.indices['AffinityCutoff']]
        initial_PhyDistance = initial_typed[:, records.indices['Invarient']]
        initial_NEigen= initial_typed[:, records.indices['NumEigenvectors']].astype(int)
    eventWise = Components.EventWise.from_file(eventWise_path)
    cols = [c for c in eventWise.columns]
    del eventWise
    position = ['input', 'eigenspace']
    DeltaR = np.linspace(0.2, 0.6, 4)
    affinitycutoff = [('distance', 2), None]
    invarient = ['invarient', 'angular', 'Luclus']
    numeigenvectors = [2, 4, 6]
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
                    invar_indices = co_indices[initial_PhyDistance[co_indices] == invar]
                    for eig in numeigenvectors:
                        eig_indices = invar_indices[initial_NEigen[invar_indices] == eig]
                        if (not os.path.exists('continue')) or os.path.exists('stopscan'):
                            return
                        if time.time() > end_time:
                            return
                        jet_class = "SpectralFull"
                        jet_params = dict(DeltaR=dR,
                                          ExpofPTMultiplier=-0.2,
                                          ExpofPTPosition=pos,
                                          NumEigenvectors=eig,
                                          Laplacien='symmetric',
                                          AffinityType='exponent2',
                                          AffinityCutoff=cutoff,
                                          PhyDistance=invar,
                                          StoppingConditon='beamparticle')
                        if jet_class not in initial_jet_class[eig_indices]:
                            print(jet_params)
                            jet_id = records.append(jet_class, jet_params)
                            jet_name = make_jet_name(jet_name, jet_id)
                            generate_pool(eventWise_path, jet_class, jet_params, jet_name, True)
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
            jet_class = "Home"
            jet_params = dict(DeltaR=dR, ExponentMultiplier=exponent)
            jet_id = records.append(jet_class, jet_params)
            jet_name = make_jet_name(jet_class, jet_id)
            generate_pool(eventWise_path, jet_class, jet_params, jet_name, True)
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
                    jet_class = "SpectralAfter"
                    jet_params = dict(DeltaR=dR, ExponentMultiplier=exponent,
                                      NumEigenvectors=n_eig,
                                      Laplacien='symmetric',
                                      AffinityType='exponent',
                                      AffinityCutoff=('distance', dis))
                    jet_id = records.append(jet_class, jet_params)
                    jet_name = make_jet_name(jet_class, jet_id)
                    generate_pool(eventWise_path, jet_class, jet_params, jet_name, True)
                jet_params = dict(DeltaR=dR, ExponentMultiplier=exponent,
                                  NumEigenvectors=n_eig,
                                  Laplacien='symmetric',
                                  AffinityType='exponent',
                                  AffinityCutoff=None)
                jet_id = records.append(jet_class, jet_params)
                jet_name = make_jet_name(jet_class, jet_id)
                generate_pool(eventWise_path, jet_class, jet_params, jet_name, True)
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
        jet_name = make_jet_name(jet_class, jet_id)
        generate_pool(eventWise_path, jet_class, next_best, jet_name, True)
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
        jet_classes = ['SpectralMean', 'SpectralFull', 'Splitting', 'Home']
        jet_class = np.random.choice(jet_classes)
    params = {}
    if 'Spectral' in jet_class or 'Splitting' in jet_class:
        if 'Spectral' in jet_class:
            permitted = FormJets.Spectral.permited_values
            params['DeltaR'] = np.random.uniform(0., 1.5)
            exppos = permitted['ExpofPTPosition']
            params['ExpofPTPosition'] = np.random.choice(exppos)
            stopcon = permitted['StoppingCondition']
            params['StoppingCondition'] = np.random.choice(stopcon)
        else:
            permitted = FormJets.Splitting.permited_values
        params['ExpofPTMultiplier'] = np.random.uniform(-1., 1.)
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
        invarients = permitted['PhyDistance']
        params['PhyDistance'] = np.random.choice(invarients)
    elif 'Home' == jet_class:
        permitted = FormJets.Traditional.permited_values
        params['DeltaR'] = np.random.uniform(0., 1.5)
        params['ExpofPTMultiplier'] = np.random.uniform(-1., 1.)
        invarients = permitted['PhyDistance']
        params['PhyDistance'] = np.random.choice(invarients)
    else:
        raise NotImplementedError
    return jet_class, params


def monte_carlo(eventWise_path, end_time, jet_class=None):
    """
    

    Parameters
    ----------
    eventWise_path :
        param jet_class:  (Default value = None)
    jet_class :
        (Default value = None)
    end_time :
        

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
    while time.time() < end_time:
        if change_class:
            jet_class = None
        jet_class, next_try = random_parameters(jet_class)
        print(f"Next try is {jet_class}; {next_try}")
        jet_id = records.append(jet_class, next_try)
        jet_name = make_jet_name(jet_class, jet_id)
        generate_pool(eventWise_path, jet_class, next_try, jet_name, True, end_time=end_time)
        records.write()


def full_run_best(eventWise_path, records, jet_class=None):
    """
    

    Parameters
    ----------
    eventWise_path :
        
    records :
        
    jet_class :
         (Default value = None)

    Returns
    -------

    """
    pass  # TODO

if __name__ == '__main__':
    eventWise_path = InputTools.get_file_name("Where is the eventwise of collection fo eventWise? ", '.awkd')
    duration = InputTools.get_time("How long to make clusters for (negative for unending)? ")
    if duration < 0:
        end_time = np.Inf
    else:
        end_time = time.time() + duration
    print(f"Ending at {end_time}")
    if InputTools.yesNo_question("Monte carlo? "):
        names = FormJets.cluster_classes
        jet_class = InputTools.list_complete("Jet class? ", names).strip()
        if jet_class == '':
            jet_class = None
        monte_carlo(eventWise_path, end_time, jet_class=jet_class)
    elif InputTools.yesNo_question("SpectralFullScan? "):
        scan_spectralfull(eventWise_path, end_time)
    elif InputTools.yesNo_question("SpectralMeanScan? "):
        scan_spectralmean(eventWise_path, end_time)
    #loops(eventWise_path)
    #iterate(eventWise_path, jet_class)

