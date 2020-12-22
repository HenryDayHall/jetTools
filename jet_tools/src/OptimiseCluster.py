""" Module for optimising the clustering process without gradient decent """
import multiprocessing
import os
import time
import awkward
import nevergrad as ng
from ipdb import set_trace as st
from jet_tools.src import InputTools, CompareClusters, Components, FormJets, TrueTag, Constants, FormShower
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
            loss += event_loss(eventWise, jet_class,
                               spectral_jet_params, other_hyperparams, {})
        except (np.linalg.LinAlgError, TypeError, Exception):
            unclusterable_counter += 1
            continue
        # we didn't manage a clustering
    loss += 2**unclusterable_counter
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


def run_optimisation(eventWise_name, batch_size=100, end_time=None, total_calls=10000, silent=True):
    eventWise = Components.EventWise.from_file(eventWise_name)
    n_events = len(eventWise.Px)
    # make a sampler
    if end_time is not None:
        total_calls = int(np.nan_to_num(np.inf))
    budget = int(total_calls/batch_size)
    sampler = RandomSampler(range(n_events), replacement=True, num_samples=total_calls)
    sampler = BatchSampler(sampler, batch_size, drop_last=True)
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
                        AffinityType='exponent')
    translator = ParameterTranslator(jet_class, fixed_params)
    variables = translator.generate_nevergrad_variables()
    # the parameters that will change should be logged
    params_to_log = [name for name in variables.kwargs if name not in fixed_params]
    # set up an optimiser
    optimiser = ng.optimizers.NGOpt(variables, budget=budget, num_workers=1)
    # inital values keep the loop simple
    print_wait = 100
    loss_log = []
    param_log = []
    for i, batch in enumerate(sampler):
        if i%print_wait == 0 and not silent:
            recent_loss = np.sum(loss_log[-print_wait:])
            if recent_loss == 0:
                num_spaces = 0
            else:
                num_spaces = min(int(np.log(recent_loss)*5), 45)
            progress = " "*num_spaces + "<"
            print(f"{i/budget:.2%}, {recent_loss:.2f}| {progress}", flush=True, end='\r')
        if end_time is not None and time.time() > end_time:
            break
        new_vars = optimiser.ask()
        spectral_jet_params = translator.translate_to_clustering(new_vars)
        param_log.append([spectral_jet_params[key] for key in params_to_log])
        loss = batch_loss(batch, eventWise, jet_class,
                                   spectral_jet_params, other_hyperparams, {})
        loss_log.append(loss)
        optimiser.tell(new_vars, loss)
    new_vars = optimiser.provide_recommendation()
    spectral_jet_params = translator.translate_to_clustering(new_vars)
    if not silent:
        print(new_vars.kwargs)
        print("Makes params")
        print(spectral_jet_params)
    print_log(params_to_log, loss_log, param_log, spectral_jet_params)


def log_text(params_to_log, loss_log, param_log, full_final_params):
    text = str(full_final_params)
    text += "\niteration\tloss\t" + "\t".join(params_to_log) + "\n"
    for i, (loss, params) in enumerate(zip(loss_log, param_log)):
        text += f"{i}\t{loss}\t"
        text += "\t".join([str(p) for p in params])
        text += "\n"
    return text


def print_log(params_to_log, loss_log, param_log, full_final_params, log_dir="./logs"):
    try:
        os.mkdir(log_dir)
    except FileExistsError:
        pass
    text = log_text(params_to_log, loss_log, param_log, full_final_params)
    log_name = os.path.join(log_dir, "log{:03d}.txt")
    i = 0
    while True:
        try:
            with open(log_name.format(i), 'x') as new_file:
                new_file.write(text)
            return
        except FileExistsError:
            i += 1
        
def generate_pool(eventWise_name, max_workers=10,
                  end_time=None, duration=None, leave_one_free=True):
    batch_size = 100
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
    for _ in range(n_threads):
        job = multiprocessing.Process(target=run_optimisation,
                                      args=(eventWise_name, batch_size, end_time))
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

if __name__ == '__main__':
    run_time = InputTools.get_time("How long should it run?")
    eventWise_name = InputTools.get_file_name("Name the eventWise: ").strip()
    generate_pool(eventWise_name, duration=run_time)

