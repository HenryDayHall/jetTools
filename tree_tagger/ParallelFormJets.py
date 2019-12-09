from tree_tagger import FormJets, Components, InputTools
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


def display_grid(*eventWises, jet_name=None):
    assert eventWises, "Must supply at least one eventWise"
    names = {"FastJet": FormJets.Traditional, "HomeJet": FormJets.Traditional,
             "SpectralJet": FormJets.Spectral}
    if jet_name is None:
        jet_name = InputTools.list_complete("Which jet? ", names.keys()).strip()
    default_params = names[jet_name].param_list
    param_list = sorted(default_params.keys())
    all_params = {}
    for eventWise in eventWises:
        matching_names = {name.split('_', 1)[0] for name in eventWise.columns
                          if name.startswith(jet_name)}
        for name in matching_names:
            if name in all_params:
                print(f"Name {name} appears in multiple eventWise")
            else:
                all_params[name] = FormJets.get_jet_params(eventWise, name)
    print(f"Params = {param_list}")
    horizontal_param = InputTools.list_complete("Horizontal param? ", param_list).strip()
    vertical_param = InputTools.list_complete("Vertical param? ", param_list).strip()
    horizontal_bins = sorted({p[horizontal_param] for p in all_params.values()})
    if isinstance(horizontal_bins[0], str):
        def get_h_index(value):
            return horizontal_bins.index(value)
    else:
        horizontal_bins_a = np.array(horizontal_bins)
        def get_h_index(value):
            return np.argmin(np.abs(horizontal_bins_a - value))
    vertical_bins = sorted({p[vertical_param] for p in all_params.values()})
    if isinstance(vertical_bins[0], str):
        def get_v_index(value):
            return vertical_bins.index(value)
    else:
        vertical_bins_a = np.array(vertical_bins)
        def get_v_index(value):
            return np.argmin(np.abs(vertical_bins_a - value))
    grid = [[[] for _ in horizontal_bins] for _ in vertical_bins]
    for name, values in all_params.items():
        v_index = get_v_index(values[vertical_param])
        h_index = get_h_index(values[horizontal_param])
        grid[v_index][h_index].append(name)
    table = [[value] + [len(entry) for entry in row] for value, row in zip(vertical_bins, grid)]
    first_row = [["\\".join([vertical_param, horizontal_param])] + horizontal_bins]
    table = first_row + table
    str_table = tabulate.tabulate(table, headers="firstrow")
    print(str_table)
    if InputTools.yesNo_question("Again? "):
        display_grid(*eventWises)


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

class Records:
    delimiter = '\t'
    def __init__(self, file_path):
        self.file_path = file_path
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r') as existing:
                reader = csv.reader(existing, delimiter=self.delimiter)
                header = next(reader)
                assert header[1] == 'jet_class'
                self.param_names = header[2:]
                self.content = []
                for line in reader:
                    self.content.append(line)
        else:
            with open(self.file_path, 'w') as new:
                writer = csv.writer(new, delimiter=self.delimiter)
                header = ['id', 'jet_class']
                writer.writerow(header)
            self.content = []
            self.param_names = []
        self.next_uid = int(np.max(self.jet_ids, initial=0)) + 1
        self.uid_length = len(str(self.next_uid))

    def write(self):
        with open(self.file_path, 'w') as overwrite:
            writer = csv.writer(overwrite, delimiter=self.delimiter)
            all_rows = [['', 'jet_class'] + self.param_names] + self.content
            writer.writerows(all_rows)

    @property
    def jet_ids(self):
        ids = [int(row[0]) for row in self.content]
        return ids

    def append(self, jet_class, param_dict):
        """ gives the new jet a unique ID and returns that value"""
        chosen_id = self.next_uid
        new_row = [f"{chosen_id:0{self.uid_length}d}", jet_class]
        new_params = list(set(param_dict.keys()) - set(self.param_names))
        if new_params:  # then we need to add some columns
            self.param_names += new_params
            new_blanks = ['' for _ in new_params]
            self.content = [row + new_blanks for row in self.content]
        for name in self.param_names:
            new_row.append(str(param_dict.get(name, '')))
        # write to disk
        with open(self.file_path, 'a') as existing:
            writer = csv.writer(existing, delimiter=self.delimiter)
            writer.writerow(new_row)
        # update content in memeory
        self.content.append(new_row)
        self.next_uid += 1
        self.uid_length = len(str(self.next_uid))
        return chosen_id

if __name__ == '__main__':
    eventWise_path = InputTools.get_file_name("Where is the eventwise of collection fo eventWise? ", '.awkd')
    record_path = "records.csv"
    records = Records(record_path)
    eventWise = Components.EventWise.from_file(eventWise_path)
    cols = [c for c in eventWise.columns]
    del eventWise
    #DeltaR = np.linspace(0.2, 1., 5)
    #exponents = [-1, 0, 1]
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
    DeltaR = np.linspace(0.05, 0.4, 8)
    exponents = [-1, 0, 1]
    NumEigenvectors = [4, 8, np.inf]
    for exponent in exponents:
        for dR in DeltaR:
            for n_eig in NumEigenvectors:
                print(f"Exponent {exponent}")
                print(f"DeltaR {dR}")
                print(f"NumEigenvectors {n_eig}")
                #jet_class = "SpectralMeanJet"
                #jet_params = dict(DeltaR=dR, ExponentMultiplier=exponent,
                #                  NumEigenvectors=n_eig)
                #jet_id = records.append(jet_class, jet_params)
                #jet_params["jet_name"] = jet_class + str(jet_id)
                #generate_pool(eventWise_path, 'SpectralMean', jet_params, True)
                jet_class = "SpectralJet"
                jet_params = dict(DeltaR=dR, ExponentMultiplier=exponent,
                                  NumEigenvectors=n_eig)
                jet_id = records.append(jet_class, jet_params)
                jet_params["jet_name"] = jet_class + str(jet_id)
                generate_pool(eventWise_path, 'Spectral', jet_params, True)
    records.write()

