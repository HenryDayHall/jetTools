from tree_tagger import FormJets, Components, InputTools
import multiprocessing
import time
import os
import numpy as np

def worker(eventWise_path, run_condition, multiapply_function, jet_params, batch_size=50):
    if isinstance(multiapply_function, 'str'):
        # functions in modules are attributes too :)
        multiapply_function = getattr(FormJets, multiapply_function)
    eventWise = Components.EventWise.from_file(eventWise_path)
    i = 0
    if run_condition is 'continue':
        while os.path.exists('continue'):
            print(f"batch {i}", flush=True)
            i+=1
            multiapply_function(eventWise, *jet_params, batch_length=batch_size, silent=True)
    elif isinstance(run_condition, int):
        while time.time() < run_condition:
            print(f"batch {i}", flush=True)
            i+=1
            multiapply_function(eventWise, **jet_params, batch_length=batch_size, silent=True)


def make_n_working_fragments(eventWise_path, n_fragments, jet_name):
    """ make n fragments, splitting of unfinished components as needed """
    if not eventWise_path.endswith('.awkd'):  # this is probably a dir name
        if '.' in eventWise_path:
            raise ValueError(f"eventWise_path {eventWise_path} is neither a directory name not the path to an eventWise")
        print(f"{eventWise_path} appears to be directory")
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
    batch_size = 50
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
    jet_prefix = jet_params['jet_name'] + "_"
    if eventWise_path.endswith(".awkd"):
        eventWise = Components.EventWise.from_file(eventWise_path)
        # cut of any finished componenets
        unfinished_components = [c for c in eventWise.columns if c.startswith(jet_prefix)]
        if len(unfinished_components) > 0:
            finished_path, unfinished_path = eventWise.split_unfinished("JetInputs_Energy", unfinished_components)
            eventWise = Components.EventWise.from_file(unfinished_path)
        # split the eventWise
        per_event_components = [c for c in eventWise.columns if c.startswith("JetInput_")]
        all_paths = np.array(eventWise.fragment(per_event_component=per_event_components, n_fragments=n_threads))
    else:
        all_paths = np.array([os.path.join(eventWise_path, name)
                              for name in os.listdir(eventWise_path)
                              if name.endswith('.awkd')])
        # go through each path cutting off any finished components
    pool = multiprocessing.Pool(n_threads)
    # now each segment makes a worker
    args = [(path, run_condition, multiapply_function, jet_params, batch_size)
            for path in all_paths]
    finished = pool.starmap(worker, args)
    pool.close()
    print(str(len(finished)) + " pools used")
    batches_finished = sum(finished)
    profile_run_time = time.time() - profile_start_time
    items_ran = batch_size * batches_finished
    profile_message = "Ran {} items in {} seconds\n".format(items_ran, profile_run_time)
    if items_ran != 0:
        profile_message += "{} seconds per item".format(profile_run_time/items_ran)
    print(profile_message)
    if np.all(finished):
        print("Finished last batch.")
    else:
        print("stopped before last batch")
        incomplete = ~np.array(finished, dtype=bool)
        print("Incomplete sections; ")
        print(all_paths[incomplete])

if __name__ == '__main__':
    eventWise_path = InputTools.get_file_name("Where is the eventwise of collection fo eventWise? ", '.awkd')
    jet_params = dict(deltaR=0.4, exponent_multiplyer=-1, jet_name="FastJet")
    generate_pool(eventWise_path, 'homejet_multiapply', jet_params, leave_one_free=False)



