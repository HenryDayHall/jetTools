""" Module for optimising the clustering process without gradient decent """
import scipy
import matplotlib
from matplotlib import pyplot as plt
import ast
import collections
import multiprocessing
import os
import time
import awkward
import nevergrad as ng
from ipdb import set_trace as st
from jet_tools.src import InputTools, CompareClusters, Components, FormJets, TrueTag, Constants, FormShower, PlottingTools
import numpy as np
from torch.utils.data import RandomSampler, BatchSampler


def event_loss(eventWise, jet_class, spectral_jet_params, other_hyperparams, generic_data):
    assert eventWise.selected_index is not None
    jets = jet_class(eventWise, dict_jet_params=spectral_jet_params,
                     assign=True)
    jets = jets.split()
    # get the tags
    tags = eventWise.BQuarkIdx.tolist()
    tag_phi = eventWise.Phi[tags]
    tag_rapidity = eventWise.Rapidity[tags]
    # get the valiables to cut on
    jet_pt = np.fromiter((jet.PT for jet in jets), dtype=float)
    if len(jet_pt) == 0:
        return None
    jet_rapidity = np.fromiter((jet.Rapidity for jet in jets), dtype=float)
    jet_phi = np.fromiter((jet.Phi for jet in jets), dtype=float)
    source_idx = eventWise.JetInputs_SourceIdx
    len_source = len(source_idx)
    input_idxs = awkward.fromiter([jet.InputIdx for jet in jets])
    jet_idxs = [source_idx[idx[idx < len_source]] for idx in input_idxs]
    num_tracks = np.fromiter((len(jet) for jet in jet_idxs), dtype=int)
    valid_jets = np.where(num_tracks > other_hyperparams['min_tracks']-0.1)[0]
    jets_tags = [[] for _ in jet_pt]
    if len(tags) == 0 or len(valid_jets) == 0:
        # there may not be any of the particles we wish to tag in the event
        # or there may not be any jets
        return None
    closest_matches = TrueTag.allocate(jet_phi, jet_rapidity, tag_phi, tag_rapidity,
                                       other_hyperparams['max_angle2'], valid_jets)
    # keep only the indices for space reasons
    for match, particle in zip(closest_matches, tags):
        if match != -1:
            jets_tags[match].append(particle)
    tag_groups = eventWise.DetectableTag_Roots
    # need to calculate the mass of each tag group in each jet
    jet_masses2 = np.zeros((len(jet_idxs), len(tags)))
    energy = eventWise.Energy
    px = eventWise.Px
    py = eventWise.Py
    pz = eventWise.Pz
    source_idx_list = source_idx.tolist()
    for tag_n, b_idx in enumerate(tags):
        b_decendants = {source_idx_list.index(d) for d in
                        FormShower.descendant_idxs(eventWise, b_idx)
                        if d in source_idx}
        for jet_n, jet_idx in enumerate(jet_idxs):
            b_in_jet = list(b_decendants.intersection(jet_idx))
            mass2 = np.sum(energy[b_in_jet])**2 - np.sum(px[b_in_jet])**2 - \
                    np.sum(py[b_in_jet])**2 - np.sum(pz[b_in_jet])**2
            jet_masses2[jet_n, tag_n] = mass2
    jet_masses = np.sqrt(np.maximum(jet_masses2, 0))
    # add the pt cut onto the valid jets
    valid_jets = valid_jets[jet_pt[valid_jets] > other_hyperparams['min_jetpt']]
    seperate_jets, matched_jets = CompareClusters.match_jets(tags, jets_tags, tag_groups,
                                                             jet_masses, valid_jets)
    mass = np.zeros((len(matched_jets), 2))
    for tag_n, tag_leaves in enumerate(eventWise.DetectableTag_Leaves):
        tag_mass2, bg_mass2, _, _, _, _, _, _, _ = \
               CompareClusters.event_detectables(tag_leaves, matched_jets[tag_n],
                                                 input_idxs, source_idx, energy,
                                                 px, py, pz, mass_only=True)
        mass[tag_n, 0] += bg_mass2
        mass[tag_n, 1] += tag_mass2
    mass = np.sqrt(np.maximum(mass, 0))
    mass[:, 1] = eventWise.DetectableTag_Mass - mass[:, 1]
    mass = np.nan_to_num(mass)
    assert np.all(mass >= -1e-5)
    loss = np.sum(mass)
    return loss


def batch_loss(batch, eventWise, jet_class, spectral_jet_params, other_hyperparams, generic_data):
    loss = 0
    unclusterable_counter = 0
    for event_n in batch:
        eventWise.selected_index = event_n
        try:
            loss_n = event_loss(eventWise, jet_class,
                                spectral_jet_params, other_hyperparams, generic_data)
            loss += loss_n
            generic_data["SuccessCount"][event_n] += 1
        except (np.linalg.LinAlgError, TypeError, Exception):
            generic_data["FailCount"][event_n] += 1
            unclusterable_counter += 1
            continue
        # we didn't manage a clustering
    loss += 2**unclusterable_counter
    loss = min(loss/len(batch), 1e5)  # cap the value of loss
    return loss


def parameter_values(jet_class, stopping_condition):
    if isinstance(jet_class, str):
        jet_class = getattr(FormJets, jet_class)
    # make some default continuous params
    continuous_params = dict(ExpofPTMultiplier=dict(mean=0., std=1., minimum=None),
                             AffinityExp=dict(mean=2., std=1., minimum=None),
                             Sigma=dict(mean=.2, std=1., minimum=0.001),
                             CutoffDistance=dict(mean=6., std=3., minimum=0.),
                             EigNormFactor=dict(mean=1.5, std=1., minimum=None))
    if stopping_condition == 'meandistance':
        continuous_params['DeltaR'] = dict(mean=1.28, std=1., minimum=0.)
    else:
        continuous_params['DeltaR'] = dict(mean=0.8, std=0.5, minimum=0.)

    ordered_discreet_params = dict(NumEigenvectors=list(range(1, 10)) + [np.inf],
                                   CutoffKNN=list(range(1, 10)) + [None])

    if jet_class == FormJets.SpectralFull:
        pass
    else:
        raise NotImplementedError(f"Implement params for {jet_class}")

    discreete_params = {key: values for key, values in jet_class.permited_values.items()
                        if key not in continuous_params and
                           key not in ordered_discreet_params}
    # remove some bad eggs
    discreete_params['StoppingCondition'] = ['standard', 'beamparticle', 'meandistance']
    discreete_params['Laplacien'] = ['unnormalised', 'symmetric', 'pt']
    return discreete_params, ordered_discreet_params, continuous_params


class ParameterTranslator:
    def __init__(self, jet_class, fixed_params):
        if isinstance(jet_class, str):
            jet_class = getattr(FormJets, jet_class)
        self.jet_class = jet_class
        if jet_class == FormJets.SpectralFull:
            self.stopping_condition = fixed_params.get('StoppingCondition', 'meandistance')
        else:
            raise NotImplementedError(f"Implement translator for {jet_class}")
        self.fixed_params = fixed_params
        self._discrete, self._ordered, self._continuous = parameter_values(jet_class,
                                                                    self.stopping_condition)
        self.parameter_order = list(self._discrete.keys()) +\
                               list(self._ordered.keys()) +\
                               list(self._continuous.keys())

    
    def generate_nevergrad_variables(self):
        variables = {key: ng.p.Choice(values) for key, values in self._discrete.items()}
        variables.update({key: ng.p.TransitionChoice(values) for key, values
                         in self._ordered.items()})
        # the continuous variables have diferent bounds so we add them indervidually
        for key, range_dict in self._continuous.items():
            param = ng.p.Scalar()
            if range_dict['minimum'] is not None:
                shifted_min = (range_dict['minimum'] - range_dict['mean'])/range_dict['std']
                param.set_bounds(shifted_min, None, 'bouncing')
            variables[key] = param
        # remove anything we are not changing
        for key in self.fixed_params:
            if key in variables:
                del variables[key]
        # create the correct type
        variables = ng.p.Instrumentation(**variables)
        return variables

    def translate_to_clustering(self, variables):
        # switching stopping condition changes the mean and range of deltar
        try:
            new_stopping_condition = variables.kwargs['StoppingCondition']
            if new_stopping_condition != self.stopping_condition:
                self.stopping_condition = new_stopping_condition
                self._discrete, self._ordered, self._continuous\
                        = parameter_values(self.jet_class, self.stopping_condition)
        except KeyError:
            pass  # it's in the fixed_params
        clustering_params = {}
        for key, value in variables.kwargs.items():
            if key in self._continuous:
                mean = self._continuous[key]['mean']
                std = self._continuous[key]['std']
                clustering_params[key] = value*std + mean
            else:
                clustering_params[key] = value
        # throw in the fixed ones
        clustering_params.update(self.fixed_params)
        return clustering_params


def get_usable_events(eventWise):
    eventWise.selected_index = None
    try:
        # must make a copy so it is alterable
        sucesses = awkward.fromiter(eventWise.SuccessCount)
        fails = awkward.fromiter(eventWise.FailCount)
    except AttributeError:
        n_events = len(eventWise.JetInputs_PT)
        sucesses = awkward.fromiter([0]*n_events)
        fails = awkward.fromiter([0]*n_events)
    # anything that hasn't failed more than 100 times should be tried again
    # if there is at least one success for every 20 fails use it
    usable = np.where(np.logical_or(sucesses*20 >= fails, fails < 100))[0]
    return usable


def make_sampler(usable_events, batch_size, test_size, end_time, total_calls):
    if end_time is not None:
        total_calls = int(np.nan_to_num(np.inf))
    train_end = len(usable_events) - test_size  # hold the last bit out for test
    # must convert the test set to a list becuase otherwise the
    # loss calculation fails on the np data type
    test_set = usable_events[train_end:].tolist()
    sampler = RandomSampler(usable_events[:train_end], replacement=True,
                            num_samples=total_calls)
    sampler = BatchSampler(sampler, batch_size, drop_last=True)
    return test_set, sampler


def run_optimisation(eventWise_name, batch_size=100, end_time=None,
                     total_calls=10000, silent=True, **kwargs):
    eventWise = Components.EventWise.from_file(eventWise_name)
    usable = get_usable_events(eventWise)
    if not silent:
        print(f"{len(usable)} events usable out of {len(eventWise.JetInputs_PT)} events")
    # this will be passed to the loss calculator
    # so it can be updated
    n_events = len(eventWise.JetInputs_PT)
    generic_data = dict(SuccessCount=np.zeros(n_events),
                        FailCount=np.zeros(n_events))
    # make a sampler
    test_set, sampler = make_sampler(usable, batch_size, test_size=500,
                                     end_time=end_time,
                                     total_calls=total_calls)
    # check we have object needed for tagging
    if "DetectableTag_Roots" not in eventWise.columns:
        TrueTag.add_detectable_fourvector(eventWise, silent=False)
    other_hyperparams = {}
    other_hyperparams['min_tracks'] = Constants.min_ntracks
    other_hyperparams['min_jetpt'] = Constants.min_pt
    max_angle = Constants.max_tagangle
    other_hyperparams['max_angle2'] = max_angle**2
    # get the initial clustering parameters
    jet_class = FormJets.SpectralFull
    fixed_params = dict(StoppingCondition='meandistance',
                        EigDistance='abscos',
                        Laplacien='symmetric',
                        PhyDistance='angular',
                        CombineSize='sum',
                        ExpofPTFormat='Luclus',
                        #ExpofPTPosition='input',
                        AffinityType='exponent')
    translator = ParameterTranslator(jet_class, fixed_params)
    variables = translator.generate_nevergrad_variables()
    # there are soem logged hyper parameters
    hyper_to_log = ["iteration", "loss", "test_loss", "batch_size"]
    # the parameters that will change should be logged
    params_to_log = [name for name in variables.kwargs if name not in fixed_params]
    # set up an optimiser
    budget = int(total_calls/batch_size)
    # get the name
    optimiser_name = kwargs.get("optimiser_name", "NGOpt")
    optimiser_params = kwargs.get("optimiser_params", {})
    other_records = {}
    other_records["optimiser_name"] = optimiser_name
    for key in optimiser_params:
        other_records["optimiser_params_" + key] = optimiser_params[key]
    # get the class
    if len(optimiser_params):
        optimiser_name = "Parametrized" + optimiser_name
        optimiser = getattr(ng.optimizers, optimiser_name)(**optimiser_params)
    else:
        optimiser = getattr(ng.optimizers, optimiser_name)
    # make an object
    optimiser = optimiser(variables, budget=budget, num_workers=1)
    # inital values keep the loop simple
    print_wait = 10
    test_interval = 10
    #log_interval = int(60000/batch_size)
    hyper_log = []
    param_log = []
    best_test_score = np.inf
    best_params = {}
    for i, batch in enumerate(sampler):
        if i%print_wait == 0 and not silent:
            recent_loss = np.sum([line[0] for line in hyper_log[-print_wait:]])
            if recent_loss == 0:
                num_spaces = 0
            else:
                num_spaces = min(int(np.log(recent_loss)*5), 45)
            progress = " "*num_spaces + "<"
            print(f"{i/budget:.2%}, {best_test_score}, {recent_loss:.2f}| {progress}", flush=True, end='\r')
        if end_time is not None and time.time() > end_time:
            break
        new_vars = optimiser.ask()
        spectral_jet_params = translator.translate_to_clustering(new_vars)
        param_log.append([spectral_jet_params[key] for key in params_to_log])
        loss = batch_loss(batch, eventWise, jet_class,
                          spectral_jet_params, other_hyperparams, generic_data)
        if i%test_interval == 0:
            test_loss = batch_loss(test_set, eventWise, jet_class,
                                   spectral_jet_params, other_hyperparams, generic_data)
            if test_loss < best_test_score:
                best_test_score = test_loss
                best_params = spectral_jet_params
        else:
            test_loss = np.nan  # the test loss is only occasionally calculated
        hyper_log.append([i, loss, test_loss, batch_size])
        optimiser.tell(new_vars, loss)
    if not silent:
        print(new_vars.kwargs)
        print("Makes params")
        print(spectral_jet_params)
    # try the sugested_params
    new_vars = optimiser.provide_recommendation()
    spectral_jet_params = translator.translate_to_clustering(new_vars)
    param_log.append([spectral_jet_params[key] for key in params_to_log])
    test_loss = batch_loss(test_set, eventWise, jet_class,
                           spectral_jet_params, other_hyperparams, generic_data)
    if test_loss < best_test_score:
        best_test_score = test_loss
        best_params = spectral_jet_params
    hyper_log.append([i+1, np.nan, test_loss, batch_size])
    # print a log
    log_dir = kwargs.get("log_dir", "./logs")
    log_index = print_log(hyper_to_log, hyper_log,
                          params_to_log, param_log,
                          best_params, log_dir, other_records)
    # print the sucesses and fails
    print_successes_fails(log_dir, log_index, generic_data["SuccessCount"],
                          generic_data["FailCount"])


def log_text(hyper_to_log, hyper_log, params_to_log, param_log, full_final_params, other_records):
    text = str(full_final_params)
    if other_records is not None:
        text += "\nother_records " + str(other_records)
    text += "\n" + "\t".join(hyper_to_log) + "\t"
    text += "\t".join(params_to_log) + "\n"
    for hyper, params in zip(hyper_log, param_log):
        text += "\t".join([str(p) for p in hyper])
        text += "\t"
        text += "\t".join([str(p) for p in params])
        text += "\n"
    return text


def print_log(hyper_to_log, hyper_log,
              params_to_log, param_log, full_final_params,
              log_dir="./logs", other_records=None):
    try:
        os.mkdir(log_dir)
    except FileExistsError:
        pass
    text = log_text(hyper_to_log, hyper_log, params_to_log, param_log, full_final_params, other_records)
    log_name = os.path.join(log_dir, "log{:03d}.txt")
    #difficulty_name = os.path.join(log_dir, "difficulty{:03d}.awkd")
    i = 0
    while True:
        try:
            with open(log_name.format(i), 'x') as new_file:
                new_file.write(text)
            return i
        except FileExistsError:
            i += 1


def print_successes_fails(log_dir, log_index, sucesses, fails):
    np.save(os.path.join(log_dir, f"sucesses{log_index:03d}.npy"), sucesses)
    np.save(os.path.join(log_dir, f"fails{log_index:03d}.npy"), fails)


def append_sucesses_fails(log_dir, eventWise_name):
    """ Go through the log dir and gather files that count sucesses and fails,
    add them to the eventWise and delete after use """
    eventWise = Components.EventWise.from_file(eventWise_name)
    n_events = len(eventWise.JetInputs_PT)
    new_sucesses = np.zeros(n_events)
    new_fails = np.zeros(n_events)
    for name in os.listdir(log_dir):
        if name.startswith("sucesses"):
            name = os.path.join(log_dir, name)
            new_sucesses += np.load(name)
        elif name.startswith("fails"):
            name = os.path.join(log_dir, name)
            new_fails += np.load(name)
        else:
            continue
        os.remove(name)
    sucesses = new_sucesses + getattr(eventWise, "SuccessCount",  0.)
    fails = new_fails + getattr(eventWise, "FailCount", 0.)
    eventWise.append(SuccessCount=awkward.fromiter(sucesses),
                     FailCount=awkward.fromiter(fails))
        
        
def generate_pool(eventWise_name, max_workers=10,
                  end_time=None, duration=None, leave_one_free=True,
                  log_dir="./logs"):
    # decide on a stop condition
    if duration is not None:
        end_time = time.time() + duration
    if end_time is not None and duration is None:
        duration = end_time - time.time()
    # work out how many threads
    n_threads = min(multiprocessing.cpu_count()-leave_one_free, max_workers)
    if n_threads < 1:
        n_threads = 1
    wait_time = duration  # in seconds
    # note that the longest wait will be n_cores time this time
    print(f"Running on {n_threads} threads", flush=True)
    job_list = []
    # now each segment makes a worker
    kwargs = {'log_dir': log_dir, 'optimiser_name': 'TBPSA',
              'optimiser_params': {'naive': False}}
    for batch_size in np.linspace(100, 300, n_threads, dtype=int):
        job = multiprocessing.Process(target=run_optimisation,
                                      args=(eventWise_name, int(batch_size),
                                            end_time), kwargs=kwargs)
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
        print(f"Problem in {sum(stalled)} out of {len(stalled)} threads")
        return False
    print("All processes ended")
    append_sucesses_fails(log_dir, eventWise_name)


def str_to_dict(string):
    # pull out any numpy inf
    out = ast.literal_eval(string.strip().replace('inf,', '"np.inf",'))
    if not isinstance(out, dict):
        out = {"string": out}
    out = {k: np.inf if isinstance(v, str) and v == 'np.inf' else v
           for k, v in out.items()}
    return out


def visulise_training(log_name=None, sep='\t'):
    if log_name is None:
        log_dir = "./logs"
        names = os.listdir(log_dir)
        log_name = InputTools.list_complete("Chose a log file: ", names).strip()
        log_name = os.path.join(log_dir, log_name)
    fig, (final_ax, score_ax) = plt.subplots(1, 2)
    fig.set_size_inches(9, 5)
    fig.suptitle(log_name)
    # read in the log file 
    with open(log_name, 'r') as log_file:
        final = str_to_dict(log_file.readline())
        line = log_file.readline().strip()
        if line.startswith("other_records"):
            other_records = str_to_dict(line.split(' ', 1)[1])
            final.update(other_records)
            line = log_file.readline().strip()
        headers = np.array(line.split(sep))
        log = awkward.fromiter([line.strip().split(sep) for line in log_file.readlines()])
    # these were added later so check for them for backward compatablity
    has_test = "test_loss" in headers
    has_batch_size = "batch_size" in headers
    # if there is a constant batch size listed, add that
    if has_batch_size:
        batch_sizes = set(log[:, headers.tolist().index("batch_size")])
        if len(batch_sizes) == 1:
            final["batch_size"] = batch_sizes.pop()
    # decide which parameter is being changed in each line
    num_hypers = 2 + has_test + has_batch_size # iteration, loss, test_loss, batch_size
    change = [set(np.where(line)[0]) 
              for line in log[1:, num_hypers:] == log[:-1, num_hypers:]]
    # this becomes a column index if the num_hypers is added
    things_that_change = np.fromiter(set(awkward.fromiter(change).flatten()),
                                     dtype=int) + num_hypers
    highlight_names = headers[things_that_change].tolist() + ["None"]
    cmap = matplotlib.cm.get_cmap('nipy_spectral')
    highlight_colours = [cmap(x) for x in np.linspace(0, 1, len(highlight_names))]
    # get the scores
    st()
    score_col = headers.tolist().index("loss")
    scores = np.fromiter(log[1:, score_col],  # skip the first, it is garbage
                         dtype=float)
    smoothed_score = scipy.ndimage.gaussian_filter(scores, 10)

    if has_test:
        test_score_col = headers.tolist().index("test_loss")
        test_scores = np.fromiter(log[1:, test_score_col],  # skip the first, it is garbage
                                  dtype=float)
        # write out the final configuration
        final["Best score"] = np.nanmin(test_scores)
    else:
        final["Best score"] = np.nanmin(scores)
    jet_name = "best"
    final["jet_name"] = jet_name
    PlottingTools.discribe_jet(jet_name="optimisation result", 
                               properties_dict=final, ax=final_ax)
    # create a legend
    for name, col in zip(highlight_names, highlight_colours):
        final_ax.plot([], [], label=name, c=col)
    final_ax.legend()
    final_ax.set_xlim(3, 13)
    final_ax.set_ylim(1, 11)
    # plot each segment
    point_reached = 0
    #colours = []
    while point_reached < len(scores) - 1:
        print(point_reached, end='\r', flush=True)
        section_end = point_reached + 1
        highlights = change[point_reached]
        if len(highlights) == 0:
            highlight = -1
            while section_end < len(change) and len(change[section_end]) == 0:
                section_end += 1
        else:
            overlap = highlights.intersection(change[section_end])
            while overlap and section_end < len(change):
                highlights = overlap
                overlap = highlights.intersection(change[section_end])
                section_end += 1
            # there may be more than one overlap, just grab one
            highlight = highlights.pop()
        here = scores[point_reached: section_end+1]  # iclude the end point
        # so it connects to the next line
        xs = range(point_reached, point_reached + len(here))
        score_ax.plot(xs, here, alpha=0.2,
                      color=highlight_colours[highlight])
        score_ax.plot(xs, smoothed_score[point_reached: section_end+1],
                      color=highlight_colours[highlight], ls='-')
        if has_test:
            test_here = test_scores[point_reached: section_end+1]
            test_filter = ~np.isnan(test_here)
            score_ax.scatter(np.array(list(xs))[test_filter], test_here[test_filter],
                             color=highlight_colours[highlight])
        #colours += [highlight_colours[highlight]]*(section_end-point_reached)
        point_reached = section_end
    #score_ax.scatter(range(len(scores)), scores, c=colours)
    score_ax.set_xlabel("Time step")
    score_ax.set_ylabel("Score")
    y_max = np.nanmax(scores[int(len(scores)*0.95):])*1.5
    y_max = 100
    score_ax.set_ylim(np.nanmin(scores)-10, y_max)
    #score_ax.set_xlim(0, 1200)
    return final, scores, headers, log
        

def visulise_logs(log_dir=None):
    if log_dir is None:
        log_dir = InputTools.get_file_name("Name the log file or directory; ", '.txt').strip()
    if os.path.isfile(log_dir):
        visulise_training(log_dir)
        return
    # otherwise we want to plot them all and same to disk
    assert os.path.isdir(log_dir)
    plt.interactive(False)
    in_dir = os.listdir(log_dir)
    for name in in_dir:
        if not (name.startswith('log') and name.endswith('txt')):
            continue
        plt_name = name.replace('.txt', '.png')
        if plt_name in in_dir:
            continue
        visulise_training(os.path.join(log_dir, name))
        plt.savefig(os.path.join(log_dir, plt_name))
        plt.close()


def cluster_from_log(log_dirs, eventWise_path, jet_class="SpectralFull", dijet_mass=40):
    # if you wanted to multithread it you would either have to duplicate the eventwise
    # or fragment it
    # it would probably be best to generalise a method in parallelformjets....
    if isinstance(log_dirs, str):
        log_dirs = [log_dirs]
    params_str = set()  # set becuase we only one one of each copy
    for log_dir in log_dirs:
        for name in os.listdir(log_dir):
            if name.startswith('log') and name.endswith('txt'):
                with open(os.path.join(log_dir, name), 'r') as log_file:
                        params_str.add(log_file.readline().strip())
    print(f"Found {len(params_str)} configurations")
    if isinstance(jet_class, str):
        jet_class = getattr(FormJets, jet_class)
    eventWise = Components.EventWise.from_file(eventWise_path)
    # form the jets
    for i, p_str in enumerate(params_str):
        print(f"{i} of {len(params_str)}\n", flush=True)
        params = str_to_dict(p_str)
        jet_name = f"OptimisedJet{i}"
        try:
            FormJets.cluster_multiapply(eventWise, jet_class, params, jet_name, np.inf)
        except Exception as e:
            print(f"couldnt cluster params {p_str}")
            print(e)
    print("Scoring")
    CompareClusters.append_scores(eventWise, dijet_mass=dijet_mass, overwrite=False)


if __name__ == '__main__':
    if InputTools.yesNo_question("Plot run? "):
        visulise_logs()
    elif InputTools.yesNo_question("Optimise? "):
        run_time = InputTools.get_time("How long should it run?")
        eventWise_name = InputTools.get_file_name("Name the eventWise: ").strip()
        log_dir = "./logs"
        generate_pool(eventWise_name, duration=run_time, log_dir=log_dir)


