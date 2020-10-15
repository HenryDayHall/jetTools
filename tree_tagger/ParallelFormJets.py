from tree_tagger import FormJets, Components, InputTools, CompareClusters
import time
import csv
import cProfile
import tabulate
import os
import numpy as np
import multiprocessing
from ipdb import set_trace as st
import itertools


def name_generator(jet_class, existing_names):
    """
    Returns a generator object that can make jet names that are not found in the existing names.

    Parameters
    ----------
    jet_class : str
        All generated names start with the jet class
    existing_names : list of str
        A list of names that are to be avoided

    Yields
    -------
    : str
        A name for a new jet
    
    """
    used_ids = [int(''.join(filter(str.isdigit, name))) for name in existing_names
                if name.startswith(jet_class)]
    current_id = 0
    while True:
        current_id += 1
        if current_id not in used_ids:
            yield make_jet_name(jet_class, current_id)
            used_ids.append(current_id)


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


def worker(*args, **kwargs):
    """
    Runs a worker, with profiling.
    Same inputs as _worker, see _worker's docstring.
    """
    if args:
        eventWise_path = args[0]
    else:
        eventWise_path = kwargs['eventWise_path']
    profiling_path = eventWise_path.replace('awkd', 'prof')
    cProfile.runctx("_worker(*args, **kwargs)", globals(), locals(), profiling_path)


def _worker(eventWise_path, run_condition, jet_class,
            jet_name, cluster_parameters, batch_size):
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
    jet_name : str
        prefix of the jet variables being worked on in the file
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
    #print(eventWise.dir_name)
    i = 0
    finished = False
    if run_condition == 'continue':
        while os.path.exists('continue') and not finished:
            #print(f"batch {i}", flush=True)
            i+=1
            finished = FormJets.cluster_multiapply(eventWise, jet_class, cluster_parameters,
                                                   jet_name=jet_name, batch_length=batch_size,
                                                   silent=True)
    elif isinstance(run_condition, (int, float)):
        while time.time() < run_condition and not finished:
            #print(f"batch {i}", flush=True)
            i+=1
            finished = FormJets.cluster_multiapply(eventWise, jet_class, cluster_parameters,
                                                   jet_name=jet_name, batch_length=batch_size,
                                                   silent=True)
    else:
        raise ValueError(f"Dont recognise run_condition {run_condition}")
    #if finished:
    #    print(f"Finished {i} batches, dataset {eventWise_path} complete")
    #else:
    #    print(f"Finished {i} batches, dataset {eventWise_path} incomplete")


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
        #print("This awkd has already been split into progress")
        eventWise_path = eventWise_path[:-5]+"_progress"
        # inside this directory, whatever is called progress1 is the thing we want
        # becuase that is the unfinished part
        unfinished_part = next(name for name in os.listdir(eventWise_path)
                              if 'progress1' in name)
        eventWise_path = os.path.join(eventWise_path, unfinished_part)
    # same logic to fragment
    if eventWise_path.endswith('awkd') and os.path.exists(eventWise_path[:-5]+"_fragment"):
        #print("This awkd has already been split into fragments")
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
        #print(f"{eventWise_path} appears to be directory")
        # if it's a directory look for subdirectories whos name starts with the directory name
        # these indicate existing splits
        leaf_dir = os.path.split(eventWise_path)[-1]
        sub_dir = [name for name in os.listdir(eventWise_path)
                   if name.startswith(leaf_dir) and os.path.isdir(os.path.join(eventWise_path, name))]
        while sub_dir:
            #print(f"Entering {sub_dir[0]}")
            eventWise_path = os.path.join(sub_dir[0])
            sub_dir = [name for name in os.listdir(eventWise_path)
                       if name.startswith(sub_dir[0]) and os.path.isdir(os.path.join(eventWise_path, name))]
        existing_fragments = [name for name in os.listdir(eventWise_path)
                              if name.endswith(".awkd")]
        if len(existing_fragments) == 0:
            raise FileNotFoundError(f"Directory {eventWise_path} has no eventWise file in")
        elif len(existing_fragments) == 1:
            #print("Path contains one eventWise")
            eventWise_path = os.path.join(eventWise_path, existing_fragments[0])
        elif len(existing_fragments) == n_fragments:
            #print("Path already contains correct number of eventWise. (may be semicomplete)")
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
            #print("Creating collective finished and unfinished parts")
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
    #print(f"In possesion of one eventWise object at {eventWise_path}")
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
    #print("Fragmenting eventwise")
    all_paths = eventWise.fragment("JetInputs_Energy", n_fragments=n_fragments)
    if unfinished_path is not None:
        # get rid of the unfishied part becuase it exists in the fragments already
        os.remove(unfinished_path)
    return all_paths


def generate_pool(eventWise_path, jet_class, jet_params, jet_name, leave_one_free=True, end_time=None):
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
    args = [(path, run_condition, jet_class, jet_name, jet_params, batch_size)
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
def remove_partial(all_paths, expected_length=None):
    """
    Remove a jet from all specified paths, optimising for the case
    where this jet may not exist in many of the jets.

    Parameters
    ----------
    all_paths : list of str
        list of fiel name sof the eventWise datasets
    expected_length : int
        the number of events that should exist
    
    """
    for ew_name in all_paths:
        ew = Components.EventWise.from_file(ew_name)
        length = len(ew.Event_n)
        if expected_length:
            assert length == expected_length
        jets = FormJets.get_jet_names(ew)
        too_short = [name + '_' for name in jets
                     if len(getattr(ew, name + "_InputIdx")) < length]
        for name in too_short:
            ew.remove_prefix(name)
        ew.write()


def recombine_eventWise(eventWise_path):
    """
    Function to recombine processed fragments of an eventWise that has been split,
    while avoiding accidentally combining with a proeviously combined component.

    Parameters
    ----------
    eventWise_path : str
        path to the eventWise that was split previously

    Returns
    -------
    new_eventWise : EventWise
        the combined processed dataset
    
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


# Spectral mean normed eigenspace -------------------------
scan_SpectralMean1 = dict(DeltaR = np.linspace(0.1, 0.9, 5),
                          ExpofPTMultiplier = [0.0, 0.2, 0.4],
                          AffinityCutoff = [None, ('knn', 3)],
                          Laplacien = ['unnormalised', 'symmetric'],
                        )
fix_SpectralMean1a = dict(
                          NumEigenvectors = 4,
                          AffinityType = 'exponent2',
                         ExpofPTFormat = 'Luclus',
                         ExpofPTPosition = 'input',
                          Eigenspace = 'normalised',
                          PhyDistance = 'taxicab',
                          StoppingCondition = 'beamparticle')
fix_SpectralMean1b = dict(
                          NumEigenvectors = 4,
                          AffinityType = 'exponent2',
                         ExpofPTFormat = 'Luclus',
                         ExpofPTPosition = 'input',
                          Eigenspace = 'normalised',
                          PhyDistance = 'taxicab',
                          StoppingCondition = 'standard')
fix_SpectralMean1c = dict(
                          NumEigenvectors = 4,
                          AffinityType = 'exponent2',
                         ExpofPTFormat = 'Luclus',
                         ExpofPTPosition = 'input',
                          Eigenspace = 'normalised',
                          PhyDistance = 'angular', 
                          StoppingCondition = 'beamparticle')

fix_SpectralMean1d = dict(
                          NumEigenvectors = 4,
                          AffinityType = 'exponent2',
                         ExpofPTFormat = 'Luclus',
                         ExpofPTPosition = 'input',
                          Eigenspace = 'normalised',
                          PhyDistance = 'angular', 
                          StoppingCondition = 'standard')

# spectral full ------------------------
scan_SpectralFull = dict(
                          ExpofPTMultiplier = [0.2, 0.0, -1.],
                          AffinityCutoff = [None, ('knn', 3), ('distance', 5)],
                          Laplacien = ['unnormalised', 'symmetric'],
                          Eigenspace = ['normalised', 'unnormalised'],
                          PhyDistance = ['angular', 'taxicab'],
                        )
fix_SpectralFull = dict(DeltaR=0.5,
                          AffinityType = 'exponent2',
                          NumEigenvectors = 4,
                         ExpofPTFormat = 'Luclus',
                         ExpofPTPosition = 'input',
                          StoppingCondition = 'beamparticle')
# Traditional -----------------------------

scan_Traditional = dict(DeltaR=np.linspace(0.2, 1.5, 10),
                        ExpofPTMultiplier=np.linspace(-1, 1, 5),
                        PhyDistance=['invarient', 'angular', 'Luclus'])

scan_Traditional1 = dict(DeltaR=np.linspace(0.2, 1.5, 10),
                        ExpofPTMultiplier=np.linspace(-1, 1, 5))

fix_Traditional1 = dict(PhyDistance='taxicab')

# Indicator -------------------------------
scan_Indicator = dict(
                      ExpofPTMultiplier=np.linspace(-1, 0, 3),
                      BaseJump=np.linspace(0.01, 0.11, 6),
                      AffinityCutoff = [None, ('distance', 3)],
                         )
fix_Indicator = dict(
                      JumpEigenFactor=10.,
                      AffinityType='exponent',
                     Laplacien='unnormalised',
                     ExpofPTFormat='Luclus',
                     ExpofPTPosition='input',
                     NumEigenvectors=np.inf,
                     PhyDistance='angular',
                     )

def scan(eventWise_path, jet_class, end_time, scan_parameters, fix_parameters=None):
    """
    Scan over all combinations of a range of options.

    Parameters
    ----------
    eventWise_path : str
        Path to the dataset used for input and writing outputs.
    jet_class : str
    end_time : int
        time to stop scanning.
    scan_parameters : dict
    fix_parameters : dict

    Returns
    -------
    time_remaining : float
        estimate for how long it would take to finish this scan.
    
    """
    start_time = time.time()
    # get the current jets
    eventWise = Components.EventWise.from_file(eventWise_path)
    existing_jets = [name for name in FormJets.get_jet_names(eventWise) if name.startswith(jet_class)]
    name_gen = name_generator(jet_class, existing_jets)
    # put the things to be iterated over into a fixed order
    key_order = list(scan_parameters.keys())
    ordered_values = [scan_parameters[key] for key in key_order]
    num_combinations = np.product([len(vals) for vals in ordered_values])
    print(f"This scan contains {num_combinations} combinations to test.")
    if existing_jets:
        complete = num_combinations/len(existing_jets)
        print(f"Assuming all existing jets are from this scan it is {complete:.1%} complete")
    finished = 0
    if fix_parameters is None:
        fix_parameters = {}
    for i, combination in enumerate(itertools.product(*ordered_values)):
        print(f"{i/num_combinations:.1%}", end='\r', flush=True)
        # check if it's been done
        parameters = {**dict(zip(key_order, combination)), **fix_parameters}
        if FormJets.check_for_jet(eventWise, parameters, pottentials=existing_jets):
            print(f"Already done {jet_class}, {parameters}\n")
        else:
            jet_name = next(name_gen)
            generate_pool(eventWise_path, jet_class, parameters, jet_name, True)
            finished += 1
        if time.time() > end_time:
            break
    time_elapsed = time.time() - start_time
    print(f"Done {finished} in {time_elapsed/60:.1f} minutes")
    if i >= num_combinations - 1:
        print(f"Finished {num_combinations} parameter combinations")
    else:
        per_combinations = time_elapsed/finished
        remaining = num_combinations - i - 1
        time_needed = (remaining*per_combinations)/60
        print(f"Estimate {time_needed:.1f} additional minutes needed to complete")


def parameter_step(eventWise, jet_class, ignore_parameteres=None, current_best=None):
    """
    Select a varient of the best jet in class that has not yet been tried

    Parameters
    ----------
    records :
        param jet_class:
    ignore_parameteres :
        Default value = None)
    jet_class :
        

    Returns
    -------

    
    """
    # get the best jet of this class
    if current_best is None:
        current_best = CompareClusters.get_best(eventWise, jet_class)
    if ignore_parameteres is None:
        ignore_parameteres = []
    # use the name of the best and get its parameters
    best_parameters = FormJets.get_jet_params(eventWise, current_best, True)
    new_parameters = {k: v for k, v in best_parameters.items()}
    tries = 0
    stopping_point = 100
    while True:
        tries += 1
        # pick one at random and change it
        to_change = np.random.choice(best_parameters)
        new_parameters[to_change] = random_parameters(jet_class, [to_change])
        possibles = FormJets.check_for_jet(eventWise, new_parameters, jet_class)
        if not possibles:
            yield new_parameters
        # reset that variable
        new_parameters[to_change] = best_parameters[to_change]
        if tries > stopping_point:
            raise StopIteration

#def iterate(eventWise_path, jet_class):
#    """
#    
#
#    Parameters
#    ----------
#    eventWise_path :
#        param jet_class:
#    jet_class :
#        
#
#    Returns
#    -------
#
#    
#    """
#    eventWise = Components.EventWise.from_file(eventWise_path)
#    record_path = "records.csv"
#    records = CompareClusters.Records(record_path)
#    print("Delete the continue file when you want to stop")
#    if not os.path.exists('continue'):
#        open('continue', 'a').close()
#    count = 0
#    while os.path.exists('continue'):
#        if count % 10 == 0:
#            print("Rescoring")
#            combined = recombine_eventWise(eventWise_path)
#            records.score(combined)
#        next_best = CompareClusters.parameter_step(records, jet_class)
#        if next_best is None:
#            print("Couldn't find new target, exiting")
#            return
#        print(f"Next best is {next_best}")
#        jet_id = records.append(jet_class, next_best)
#        jet_name = make_jet_name(jet_class, jet_id)
#        generate_pool(eventWise_path, jet_class, next_best, jet_name, True)
#        records.write()
#        count += 1


def random_parameters(jet_class=None, desired_parameters=None, omit_parameters=None):
    """
    

    Parameters
    ----------
    jet_class :
        Default value = None)

    Returns
    -------

    
    """
    if jet_class is None:
        jet_classes = ['SpectralMean', 'SpectralFull', 'Splitting', 'Indicator', 'Home']
        jet_class = np.random.choice(jet_classes)
    params = {}
    permitted_vals = getattr(FormJets, jet_class).permited_values
    if desired_parameters is None:
        desired_parameters = permitted_vals.keys()
    if omit_parameters is not None:
        desired_parameters = [x for x in desired_parameters if x not in omit_parameters]
    for key, selection in permitted_vals.items():
        if key not in desired_parameters:
            continue
        if key == 'DeltaR':
            params[key] = np.around(np.random.uniform(0, 1.5), 1)
        elif key == 'ExpofPTMultiplier':
            params[key] = np.around(np.random.uniform(-1., 1.), 1)
        elif key == 'NumEigenvectors':
            params[key] = np.random.randint(1, 10)
        elif key == 'MaxCutScore':
            params[key] = np.around(np.random.uniform(0.2, 1.), 1)
        elif key == 'BaseJump':
            params[key] = np.around(np.random.uniform(0.01, 0.5), 2)
        elif key == 'JumpEigenFactor':
            params[key] = np.around(np.random.uniform(0., 100), -1)
        elif key == 'AffinityCutoff':
            cutofftypes = [None if x is None else x[0] for x in selection]
            cutofftype = np.random.choice(cutofftypes)
            if cutofftype is None:
                params[key] = cutofftype
            elif cutofftype == 'knn':
                params[key] = (cutofftype, np.random.randint(1, 6))
            elif cutofftype == 'distance':
                params[key] = (cutofftype, np.around(np.random.uniform(0., 10.), 1))
        else:  # all the remaining ones are selected from lists
            params[key] = np.random.choice(selection)
    return jet_class, params


def monte_carlo(eventWise_path, end_time, jet_class=None, fixed_parameters=None):
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
    if jet_class == '':
        jet_class = None
    change_class = jet_class is None
    eventWise = Components.EventWise.from_file(eventWise_path)
    existing_jets = FormJets.get_jet_names(eventWise)
    name_gen = name_generator(jet_class, existing_jets)
    #print("Delete the continue file when you want to stop")
    #if not os.path.exists('continue'):
    #    open('continue', 'a').close()
    #while os.path.exists('continue'):
    if fixed_parameters is None:
        fixed_parameters = {}
    while time.time() < end_time:
        if change_class:
            jet_class = None
        jet_class, next_try = random_parameters(jet_class, omit_parameters=fixed_parameters)
        next_try.update(fixed_parameters)
        print(f"Next try is {jet_class}; {next_try}")
        jet_name = next(name_gen)
        generate_pool(eventWise_path, jet_class, next_try, jet_name, True, end_time=end_time)


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
        #monte_carlo(eventWise_path, end_time, jet_class=jet_class)
        monte_carlo(eventWise_path, end_time, jet_class=jet_class, fixed_parameters=cetrain_akt_Spectral)

