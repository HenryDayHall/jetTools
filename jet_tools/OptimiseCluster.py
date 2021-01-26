""" Module for optimising the clustering process without gradient decent """
import warnings
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
from jet_tools import InputTools, CompareClusters, Components, FormJets, TrueTag, Constants, FormShower, PlottingTools
import numpy as np
import abcpy.distances
import abcpy.discretemodels
import abcpy.continuousmodels
import abcpy.probabilisticmodels
import abcpy.statistics
import abcpy.backends
import abcpy.output
import abcpy.inferences
import copy


def event_loss(eventWise, jet_class, jet_params, other_hyperparams, generic_data):
    warnings.filterwarnings('ignore')
    assert eventWise.selected_index is not None
    jets = jet_class(eventWise, dict_jet_params=jet_params,
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
    missed_jets = len(eventWise.DetectableTag_Roots.flatten()) - seperate_jets
    mass2 = np.zeros((len(matched_jets), 2))
    # insist on seperate jets
    for tag_n, tag_leaves in enumerate(eventWise.DetectableTag_Leaves):
        jets_here = matched_jets[tag_n]
        if len(jets_here) and len(eventWise.DetectableTag_Roots[tag_n]) > len(jets_here):
            # at least one jet is a merged jet
            # leave both masses 0
            continue
        tag_mass2, bg_mass2, _, _, _, _, _, _, _ = \
               CompareClusters.event_detectables(tag_leaves, jets_here,
                                                 input_idxs, source_idx, energy,
                                                 px, py, pz, mass_only=True)
        mass2[tag_n, 0] += bg_mass2
        mass2[tag_n, 1] += tag_mass2
    mass2 = np.maximum(mass2, 0)   # imaginary mass is a floating point error
    # take the euclidien norm of the two types of mass
    mass = np.sqrt(mass2[:, 0] + 
                   (eventWise.DetectableTag_Mass - np.sqrt(mass2[:, 1]))**2)
    # penalise missing jet a little
    missed_jets_penalty = 1. + 0.1*(missed_jets)
    loss = np.nansum(mass) * missed_jets_penalty
    #assert np.isfinite(loss), f"loss is {loss}, mass2={mass2}, merge_or_loss_penalty={merge_or_loss_penalty}"
    return loss


def batch_loss(batch, eventWise, jet_class, spectral_jet_params, other_hyperparams, generic_data):
    loss = 0
    unclusterable_counter = 0
    for event_n in batch:
        eventWise.selected_index = int(event_n)
        try:
            loss_n = event_loss(eventWise, jet_class,
                                spectral_jet_params, other_hyperparams, generic_data)
            loss += loss_n
            generic_data["SuccessCount"][event_n] += 1
        except (np.linalg.LinAlgError, TypeError, Exception) as e:
            generic_data["FailCount"][event_n] += 1
            unclusterable_counter += 1
            continue
        # we didn't manage a clustering
    loss += 2**unclusterable_counter
    loss = min(loss/len(batch), 1e5)  # cap the value of loss
    #print(f"loss = {loss}, unclusterable = {unclusterable_counter}")
    if loss < 200:
        print('.', end='', flush=True)
    return loss


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


class BatchSampler:
    def __init__(self, data, batch_size, num_samples=np.inf):
        if isinstance(data, list):
            data = awkward.fromiter(data)
        self.data = data
        float_len = len(data)/batch_size
        self._len = int(np.floor(float_len))
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.samples_dispensed = 0
        # make a seed
        np.random.seed(int(time.time()))

    def __len__(self):
        return self._len

    def __iter__(self):
        data_len = len(self.data)
        # a list of indieces for shuffling
        # avoids accedntally loading data needlessly
        idxs = list(range(data_len))
        np.random.shuffle(idxs)
        batch_size = self.batch_size
        reached = 0
        batch_end = 0
        while self.samples_dispensed < self.num_samples:
            batch_end += batch_size
            if batch_end >= data_len:
                reached = 0
                batch_end = batch_size
                np.random.shuffle(idxs)
            yield self.data[idxs[reached:batch_end]]
            reached = batch_end
            self.samples_dispensed += 1


def make_sampler(usable_events, batch_size, test_size, end_time, total_calls):
    if end_time is not None:
        total_calls = np.inf
        budget = np.inf
    else:
        budget = int(total_calls/batch_size)
    train_end = len(usable_events) - test_size  # hold the last bit out for test
    # must convert the test set to a list becuase otherwise the
    # loss calculation fails on the np data type
    test_set = usable_events[train_end:].tolist()
    sampler = BatchSampler(usable_events[:train_end], batch_size=batch_size,
                           num_samples=budget)
    return test_set, sampler, budget


def parameter_values(jet_class, stopping_condition=None):
    if isinstance(jet_class, str):
        jet_class = getattr(FormJets, jet_class)
    # make some default continuous params
    continuous_params = dict(ExpofPTMultiplier=dict(mean=0., std=1., minimum=None),
                             EigenvalueLimit=dict(mean=0.5, std=0.5, minimum=0.00001),
                             AffinityExp=dict(mean=2., std=1., minimum=0.001),
                             Sigma=dict(mean=.2, std=1., minimum=0.001),
                             CutoffDistance=dict(mean=6., std=3., minimum=0.),
                             EigNormFactor=dict(mean=1.5, std=1., minimum=0.))
    if stopping_condition == 'meandistance':
        continuous_params['DeltaR'] = dict(mean=1.28, std=1., minimum=0.)
    else:
        continuous_params['DeltaR'] = dict(mean=0.8, std=0.5, minimum=0.)

    ordered_discreet_params = dict(NumEigenvectors=list(range(1, 10)) + [np.inf],
                                   CutoffKNN=list(range(1, 10)) + [None])

    discreete_params = {key: values for key, values in jet_class.permited_values.items()
                        if key not in continuous_params and
                           key not in ordered_discreet_params}
    # remove some bad eggs
    discreete_params['StoppingCondition'] = ['standard', 'beamparticle', 'meandistance']
    discreete_params['Laplacien'] = ['unnormalised', 'symmetric', 'pt']
    
    # tweaks for specific classes
    if jet_class == FormJets.SpectralFull:
        pass
    elif jet_class == FormJets.SpectralKMeans:
        del discreete_params["StoppingCondition"]
        del continuous_params["DeltaR"]
    else:
        raise NotImplementedError(f"Implement params for {jet_class}")

    return discreete_params, ordered_discreet_params, continuous_params


class ParameterTranslator:
    def __init__(self, jet_class, fixed_params, ):
        if isinstance(jet_class, str):
            jet_class = getattr(FormJets, jet_class)
        self.jet_class = jet_class
        self.fixed_params = fixed_params
        if jet_class == FormJets.SpectralFull:
            stopping_condition = fixed_params.get("StoppingCondition", "meandistance")
        elif jet_class == FormJets.SpectralKMeans:
            stopping_condition = None
        else:
            raise NotImplementedError
        self.discrete, self.ordered, self.continuous = parameter_values(jet_class,
                                                                        stopping_condition)
        # soemtimes it is useful to have all the listlike data togethere
        self.all_discrete = {**self.discrete, **self.ordered}
        self.parameter_order = list(self.discrete.keys()) +\
                               list(self.ordered.keys()) +\
                               list(self.continuous.keys())
        self.unfixed_order = [name for name in self.parameter_order if name not in fixed_params]
        self._unfixed_stopping = "StoppingCondition" in self.unfixed_order
        if self._unfixed_stopping:  # changes other params
            self._stopping_index = self.unfixed_order.index("StoppingCondition")
            self._current_stopping = self.unfixed_order["StoppingCondition"]\
                                          .index(stopping_condition)
    
    def generate_nevergrad_variables(self):
        variables = {key: ng.p.Choice(values) for key, values in self.discrete.items()}
        variables.update({key: ng.p.TransitionChoice(values) for key, values
                         in self.ordered.items()})
        # the continuous variables have diferent bounds so we add them indervidually
        for key, range_dict in self.continuous.items():
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

    def generate_abcpy_variables(self):
        variables = []
        for name in self.unfixed_order:
            # treat ordered and unordered the same
            if name in self.all_discrete:
                n_items = len(self.all_discrete[name])
                model = abcpy.discretemodels.DiscreteUniform([0, n_items-1], name=name)
            else:
                mean = self.continuous[name]['mean']
                std = self.continuous[name]['std']
                minm = self.continuous[name]['minimum']
                # calculate values
                #maximum = mean + std*5
                #if minm is None:
                #    minimum = mean - std*5
                #else:
                #    minimum = minm
                if minm is None:
                    model = abcpy.continuousmodels.Normal([mean, std], name=name)
                else:
                    model = ClippedNormal([mean, std, minm], name=name)
            variables.append(model)
        return variables
    
    def list_to_clustering(self, param_list):
        if self._unfixed_stopping:
            new_stopping_condition = param_list[self._stopping_index]
            if new_stopping_condition != self._current_stopping:
                self._current_stopping = new_stopping_condition
                # convert from index to string
                stopping_condition = self.discrete["StoppingCondition"][self._current_stopping]
                self.discrete, self.ordered, self.continuous\
                        = parameter_values(self.jet_class, stopping_condition)
        clustering_params = {}
        for name, value in zip(self.unfixed_order, param_list):
            if name in self.continuous:
                clustering_params[name] = value
            else:
                # map the index back into range
                options = self.all_discrete[name]
                index = map_index(int(value), 0, len(options)-1)
                clustering_params[name] = options[index]
        # throw in the fixed ones
        clustering_params.update(self.fixed_params)
        return clustering_params

    def nevergrad_to_clustering(self, variables):
        # switching stopping condition changes the mean and range of deltar
        if self._unfixed_stopping:
            new_stopping_condition = variables.kwargs['StoppingCondition']
            if new_stopping_condition != self.discrete[self._current_stopping]:
                self._current_stopping = self.discrete["StoppingCondition"]\
                                              .index(new_stopping_condition)
                self.discrete, self.ordered, self.continuous\
                        = parameter_values(self.jet_class, new_stopping_condition)
        clustering_params = {}
        for key, value in variables.kwargs.items():
            if key in self.continuous:
                mean = self.continuous[key]['mean']
                std = self.continuous[key]['std']
                clustering_params[key] = value*std + mean
            else:
                clustering_params[key] = value
        # throw in the fixed ones
        clustering_params.update(self.fixed_params)
        return clustering_params


def map_index(value, minimum, maximum):
    valid_range = maximum - minimum
    in_twice_range = abs(value - minimum)%(2*valid_range)
    from_maximum = abs(in_twice_range - valid_range)
    return maximum - from_maximum


################## NEVERGRAD ##################

def run_optimisation_nevergrad(eventWise_name, batch_size=100, end_time=None,
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
    test_set, sampler, budget = make_sampler(usable, batch_size, test_size=500,
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
    jet_class = kwargs.get("jet_class", FormJets.SpectralFull)
    if isinstance(jet_class, str):
        jet_class = getattr(FormJets, jet_class)
    if "fixed_params" in kwargs:
        fixed_params = kwargs["fixed_params"]
    elif jet_class == FormJets.SpectralFull:
        fixed_params = dict(StoppingCondition='meandistance',
                            EigDistance='abscos',
                            Laplacien='symmetric',
                            PhyDistance='angular',
                            CombineSize='sum',
                            ExpofPTFormat='Luclus',
                            #ExpofPTPosition='input',
                            AffinityType='exponent')
    elif jet_class == FormJets.SpectralKMeans:
        fixed_params = dict(EigDistance='abscos',
                            Laplacien='symmetric',
                            PhyDistance='angular',
                            ExpofPTFormat='Luclus',
                            ExpofPTPosition='input',
                            AffinityType='exponent')
    else:
        raise NotImplementedError
    translator = ParameterTranslator(jet_class, fixed_params)
    variables = translator.generate_nevergrad_variables()
    # there are soem logged hyper parameters
    hyper_to_log = ["iteration", "loss", "test_loss", "batch_size"]
    # the parameters that will change should be logged
    params_to_log = [name for name in variables.kwargs if name not in fixed_params]
    # set up an optimiser
    # get the name
    optimiser_name = kwargs.get("optimiser_name", "NGOpt")
    optimiser_params = kwargs.get("optimiser_params", {})
    other_records = {}
    other_records["optimiser_name"] = optimiser_name
    for key in optimiser_params:
        other_records["optimiser_params_" + key] = optimiser_params[key]
    # get the class
    if len(optimiser_params):
        try:
            class_name = "Parametrized" + optimiser_name
            optimiser = getattr(ng.optimizers, class_name)(**optimiser_params)
        except AttributeError:  # there is no consistent naming convention....
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
        spectral_jet_params = translator.nevergrad_to_clustering(new_vars)
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
    spectral_jet_params = translator.nevergrad_to_clustering(new_vars)
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

        
def generate_pool(eventWise_name, max_workers=10,
                  end_time=None, duration=None, leave_one_free=True,
                  log_dir="./logs", **kwargs):
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
    optimiser_name = kwargs.get('optimiser_name', 'NGOpt')
    # now each segment makes a worker
    kwargs.update({'log_dir': log_dir, 'optimiser_name': optimiser_name})
    if optimiser_name == 'DifferentialEvolution':
        kwargs['optimiser_params'] = {'recommendation': "noisy"}
    for batch_size in np.linspace(100, 300, n_threads, dtype=int):
        job = multiprocessing.Process(target=run_optimisation_nevergrad,
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


################### ABCPY  ##############################


# wont do sampling here as the vectoisation is hard to predict
class TimedPMCABC(abcpy.inferences.PMCABC):
    def sample(self, observations, duration,
               epsilon_init, n_samples, n_samples_per_param=1,
               epsilon_percentile=10,
               covFactor=2, full_output=0, journal_file=None,
               log_dir="./logs"):
        """Samples from the posterior distribution of the model parameter given the observed
        data observations.
        Identical to the standard method, appart from uses a time limit rather than a number of steps.

        Parameters
        ----------
        observations : list
            A list, containing lists describing the observed data sets
        duration : float, optional
            Time in seconds to run for. The default value is 5 mins.
        epsilon_init : numpy.ndarray
            An array of proposed values of epsilon to be used at each steps. Can be supplied
            A single value to be used as the threshold in Step 1 or a `steps`-dimensional array of values to be
            used as the threshold in evry steps.
        n_samples : integer
            Number of samples to generate.
        n_samples_per_param : integer, optional
            Number of data points in each simulated data set. The default value is 1.
        epsilon_percentile : float, optional
            A value between [0, 100]. The default value is 10.
        covFactor : float, optional
            scaling parameter of the covariance matrix. The default value is 2 as considered in [1].
        full_output: integer, optional
            If full_output==1, intermediate results are included in output journal.
            The default value is 0, meaning the intermediate results are not saved.
        journal_file: str, optional
            Filename of a journal file to read an already saved journal file, from which the first iteration will start.
            The default value is None.

        Returns
        -------
        abcpy.output.Journal
            A journal containing simulation results, metadata and optionally intermediate results.
        """
        self.accepted_parameters_manager.broadcast(self.backend, observations)
        self.n_samples = n_samples
        self.n_samples_per_param = n_samples_per_param

        # track the journals event if the function crashes
        journal_list = []
        if isinstance(journal_file, list):
            journal_list = journal_file
            if len(journal_list):
                journal_file = journal_list[-1]
            else:
                journal_file = None
        if journal_file is None:
            journal = abcpy.output.Journal(full_output)
            journal.configuration["type_model"] = [type(model).__name__ for model in self.model]
            journal.configuration["type_dist_func"] = type(self.distance).__name__
            journal.configuration["n_samples"] = self.n_samples
            journal.configuration["n_samples_per_param"] = self.n_samples_per_param
            journal.configuration["steps"] = -1
            journal.configuration["epsilon_percentile"] = epsilon_percentile
        else:
            journal = abcpy.output.Journal.fromFile(journal_file)

        journal_name, _ = reserve_name(os.path.join(log_dir, "Journal{:03d}.jnl"))
        journal_list.append(journal_name)

        if journal_file is not None:
            accepted_parameters = journal.get_accepted_parameters(-1)
            accepted_weights = journal.get_weights(-1)

            self.accepted_parameters_manager.update_broadcast(self.backend, accepted_parameters=accepted_parameters,
                                                              accepted_weights=accepted_weights)

            kernel_parameters = []
            for kernel in self.kernel.kernels:
                kernel_parameters.append(
                    self.accepted_parameters_manager.get_accepted_parameters_bds_values(kernel.models))
            self.accepted_parameters_manager.update_kernel_values(self.backend, kernel_parameters=kernel_parameters)

            # 3: calculate covariance
            self.logger.info("Calculateing covariance matrix")
            new_cov_mats = self.kernel.calculate_cov(self.accepted_parameters_manager)
            # Since each entry of new_cov_mats is a numpy array, we can multiply like this
            # accepted_cov_mats = [covFactor * new_cov_mat for new_cov_mat in new_cov_mats]
            accepted_cov_mats = self._compute_accepted_cov_mats(covFactor, new_cov_mats)
            self.accepted_parameters_manager.update_broadcast(self.backend, accepted_cov_mats=accepted_cov_mats)

        else:
            accepted_parameters = None
            accepted_weights = None
            accepted_cov_mats = None
            self.accepted_parameters_manager.update_broadcast(self.backend, accepted_parameters=accepted_parameters,
                                                              accepted_weights=accepted_weights)
            self.accepted_parameters_manager.update_broadcast(self.backend, accepted_cov_mats=accepted_cov_mats)
            distances = None

        # Define epsilon_arr
        if journal_file is not None:
            # check distances
            distances = journal.distances[-1]
            # should be max or min?
            epsilon_arr = np.maximum(epsilon_init, np.max(distances))
        else:
            epsilon_arr = epsilon_init
        epsilon_arr = list(epsilon_arr)

        # main PMCABC algorithm
        self.logger.info("Starting PMC iterations")
        aStep = 0
        end_time = time.time() + duration
        step_begun = time.time()
        last_save = step_begun
        save_wait = 5*60
        projected_end = step_begun
        while projected_end < end_time:
            step_begun = time.time()
            self.logger.debug(f"iteration {aStep} of PMC algorithm, started at {step_begun}".format(aStep))
            seed_arr = self.rng.randint(0, np.iinfo(np.uint32).max, size=n_samples, dtype=np.uint32)
            rng_arr = np.array([np.random.RandomState(seed) for seed in seed_arr])
            rng_pds = self.backend.parallelize(rng_arr)

            # 0: update remotely required variables
            # print("INFO: Broadcasting parameters.")
            self.logger.info("Broadcasting parameters")
            self.epsilon = epsilon_arr[aStep]
            #self.accepted_parameters_manager.update_broadcast(self.backend, accepted_parameters, accepted_weights,
            #                                                  accepted_cov_mats)

            # 1: calculate resample parameters
            # print("INFO: Resampling parameters")
            self.logger.info("Resampling parameters")

            params_and_dists_and_counter_pds = self.backend.map(self._resample_parameter, rng_pds)
            params_and_dists_and_counter = self.backend.collect(params_and_dists_and_counter_pds)
            new_parameters, distances, counter = [list(t) for t in zip(*params_and_dists_and_counter)]
            # need to instist parameters are floats, otherwise other bits break...
            new_parameters = np.array(new_parameters, dtype=float)
            distances = np.array(distances)

            for count in counter:
                self.simulation_counter += count

            # Compute epsilon for next step
            # print("INFO: Calculating acceptance threshold (epsilon).")
            self.logger.info("Calculating acceptances threshold")
            if aStep + 2 > len(epsilon_arr):
                epsilon_arr.append(np.percentile(distances, epsilon_percentile))
            else:
                epsilon_arr[aStep + 1] = np.max(
                    [np.percentile(distances, epsilon_percentile),
                     epsilon_arr[aStep + 1]])

            # 2: calculate weights for new parameters
            self.logger.info("Calculating weights")

            new_parameters_pds = self.backend.parallelize(new_parameters)
            self.logger.info("Calculate weights")
            new_weights_pds = self.backend.map(self._calculate_weight, new_parameters_pds)
            new_weights = np.array(self.backend.collect(new_weights_pds)).reshape(-1, 1)
            sum_of_weights = np.sum(new_weights)
            new_weights = new_weights / sum_of_weights

            # The calculation of cov_mats needs the new weights and new parameters
            self.accepted_parameters_manager.update_broadcast(self.backend, accepted_parameters=new_parameters,
                                                              accepted_weights=new_weights)

            # The parameters relevant to each kernel have to be used to calculate n_sample times. It is therefore more efficient to broadcast these parameters once,
            # instead of collecting them at each kernel in each step
            kernel_parameters = []
            for kernel in self.kernel.kernels:
                kernel_parameters.append(
                    self.accepted_parameters_manager.get_accepted_parameters_bds_values(kernel.models))
            self.accepted_parameters_manager.update_kernel_values(self.backend, kernel_parameters=kernel_parameters)

            # 3: calculate covariance
            self.logger.info("Calculating covariance matrix")
            new_cov_mats = self.kernel.calculate_cov(self.accepted_parameters_manager)
            # Since each entry of new_cov_mats is a numpy array, we can multiply like this
            new_cov_mats = [covFactor * new_cov_mat for new_cov_mat in new_cov_mats]
            self.accepted_parameters_manager.update_broadcast(self.backend, accepted_cov_mats=new_cov_mats)

            # 4: Update the newly computed values
            accepted_parameters = new_parameters
            accepted_weights = new_weights
            accepted_cov_mats = new_cov_mats

            self.logger.info("Save configuration to output journal")
            # guess when the next step will end
            step_end = time.time()
            print(f"Iteration took {(step_end - step_begun)/60:.1f} mins")
            projected_end = 2*step_end - step_begun
            aStep += 1
            write_now = ((full_output == 1) or
                         (full_output == 0 and projected_end > end_time))
            save_now = save_wait < step_end - last_save
            if write_now or save_now:
                journal.configuration["steps"] = aStep
                journal.add_accepted_parameters(copy.deepcopy(accepted_parameters))
                journal.add_distances(copy.deepcopy(distances))
                journal.add_weights(copy.deepcopy(accepted_weights))
                journal.add_ESS_estimate(accepted_weights)
                #self.accepted_parameters_manager.update_broadcast(self.backend, accepted_parameters=accepted_parameters,
                #                                                  accepted_weights=accepted_weights)
                names_and_parameters = self._get_names_and_parameters()
                journal.add_user_parameters(names_and_parameters)
                journal.number_of_simulations.append(self.simulation_counter)
                if save_now:
                    journal.save(journal_name)
                    last_save = step_end
        self.n_samples = aStep
        # Add epsilon_arr to the journal
        journal.configuration["epsilon_arr"] = epsilon_arr
        journal.configuration["steps"] = aStep
        journal.save(journal_name)
        return journal_name, journal


class ClusteringModel(abcpy.probabilisticmodels.ProbabilisticModel,
                      abcpy.continuousmodels.Continuous):
    def __init__(self, parameters, name='clustering', **kwargs):
        """
        Parameters
        ----------
        parameters : list
            list of hypereparameters for clustering
        eventWise_path : str
            name of the data file
        batch_size : int
            number of events to run at each time step
        test_size : int
            size held out for test
        translator : ParameterTranslator
            to get the parameters back to clustring format
	
        """
        if not isinstance(parameters, list):
            raise TypeError("model takes parameters int he form of a list")
        # start by getting the dataset
        eventWise_path = kwargs.get("eventWise_path")
        self.eventWise = Components.EventWise.from_file(eventWise_path)
        n_events = len(self.eventWise.JetInputs_PT)
        # make a sampler
        batch_size = kwargs.get("batch_size", 100)
        test_size = kwargs.get("test_size", 500)
        usable = get_usable_events(self.eventWise)
        self.test_set, sampler, _ = make_sampler(usable, batch_size, test_size=test_size,
                                                 end_time=np.inf, total_calls=1e10)
        self.sampler = iter(sampler)
        # get the translator
        self.translator = kwargs.get("translator")
        self.jet_class = self.translator.jet_class
        # check we got the right number of parameters
        if len(parameters) != len(self.translator.unfixed_order):
            raise RuntimeError(f"Model needs {len(self.translator.unfixed_order)} parameters, "
                               +f"\n({self.translator.unfixed_order})\n"
                               +f" but found {len(parameters)} parameters.")
        # make other objects used
        self.other_hyperparams = {}
        self.other_hyperparams['min_tracks'] = Constants.min_ntracks
        self.other_hyperparams['min_jetpt'] = Constants.min_pt
        max_angle = Constants.max_tagangle
        self.other_hyperparams['max_angle2'] = max_angle**2
        self.generic_data = dict(SuccessCount=np.zeros(n_events),
                                 FailCount=np.zeros(n_events))
        #  must call the super scontructor
        input_connector = abcpy.probabilisticmodels.InputConnector.from_list(parameters)
        super().__init__(input_connector, name)

    def _check_input(self, input_values):
        # check we got the right number of parameters
        variable_order = self.translator.unfixed_order
        if len(input_values) != len(variable_order):
            raise ValueError(f"Model needs {len(self.translator.unfixed_order)} parameters, "
                             +f"\n({variable_order})\n"
                             +f" but found {len(input_values)} parameters.")
        # check each variable
        discreet_dict = self.translator.all_discrete
        for name, value in zip(variable_order, input_values):
            # treat ordered and unordered the same
            if name in discreet_dict:
                pass  # have fixed this in the interpreter instead
                #n_items = len(discreet_dict[name])
                #if value >= n_items:
                #    print(f"Got index {value} for {name} which has {n_items} items")
                #    return False
            else:
                # inequality boundaries here seem to cause crashes
                # better to just let the model eat the wrong value and fail.
                #minimum = self.translator.continuous[name]['minimum']
                #if minimum is not None and value < minimum:
                #    print(f"Got values {value} for {name} which has minimum {minimum}")
                #    return False
                # sigma is specal case
                if name == 'sigma' and value == 0:
                    print(f"Got values {value} for {name} which cannot be 0")
                    return False
        # no checks have failed if we reach here
        return True

    def forward_simulate(self, input_values, k, rng=np.random.RandomState()):
        """Calculates the score of the dataset.

        priors -> distribution of hyperparametres -> hyperparameter choice ->
        score on data

        either the first or the second model should have None as parameters

        Parameters
        ----------
        input_values: Python list
            Parameters for model
        k: int
            number of batches to do
        rng : random number generator
            not used, sampler does that job

        Returns
        -------
        scores : list of k numpy arrays
            each the score of one batch
        """
        results = []
        for batch_n in range(k):
            batch = next(self.sampler)
            full_parameteres = self.translator.list_to_clustering(input_values)
            score = batch_loss(batch, self.eventWise, self.jet_class, full_parameteres,
                               self.other_hyperparams, self.generic_data)
            results.append(np.array(score))
        return results

    def _check_output(self, values):
        if np.any(np.isnan(values)):
            raise ValueError('got nan out of a batch. how?')
        return True

    def get_output_dimension(self):
        return 1


class ClippedNormal(abcpy.probabilisticmodels.ProbabilisticModel,
                    abcpy.probabilisticmodels.Continuous):
    def __init__(self, parameters, name='Normal'):
        """
        This class implements a probabilistic model following a normal distribution with mean mu and variance sigma.
        It optionally bounds the distribution.

        Parameters
        ----------
        parameters: list
            Contains the probabilistic models and hyperparameters from which the model derives.
            The list has four entries: from the first entry mean of the distribution and from the second entry variance is derived.
            Note that the second value of the list is strictly greater than 0.
            the third entry is a lower bound.

        name: string
            The name that should be given to the probabilistic model in the journal file.
        """
        if not isinstance(parameters, list):
            raise TypeError('Input for Normal has to be of type list.')
        if len(parameters) != 3:
            raise ValueError(f'Input for Normal has to be of length 3. Found {parameters}')
        input_parameters = abcpy.probabilisticmodels.InputConnector.from_list(parameters)
        super().__init__(input_parameters, name)
        self.visited = False

    def _check_input(self, input_values):
        """
        Returns True if the standard deviation is negative.
        """
        if len(input_values) != 3:
            return False
        if input_values[1] <= 0:
            return False
        return True

    def _check_output(self, parameters):
        """
        Checks parameter values that are given as fixed values.
        """
        return True

    def forward_simulate(self, input_values, k, rng=np.random.RandomState(), mpi_comm=None):
        """
        Samples from a normal distribution using the current values for each probabilistic model from which the model derives.

        Parameters
        ----------
        input_values: list
            List of input parameters, in the same order as specified in the InputConnector passed to the init function
        k: integer
            The number of samples that should be drawn.
        rng: Random number generator
            Defines the random number generator to be used. The default value uses a random seed to initialize the generator.

        Returns
        -------
        list: [np.ndarray]
            A list containing the sampled values as np-array.
        """
        mu = input_values[0]
        sigma = input_values[1]
        result = np.array(rng.normal(mu, sigma, k))
        np.clip(result, input_values[2], None, result)
        return [np.array([x]).reshape(-1, ) for x in result]

    def get_output_dimension(self):
        return 1

    def pdf(self, input_values, x):
        """
        Calculates the probability density function at point x.
        Commonly used to determine whether perturbed parameters are still valid according to the pdf.

        Parameters
        ----------
        input_values: list
            List of input parameters of the from [mu, sigma]
        x: list
            The point at which the pdf should be evaluated.

        Returns
        -------
        Float:
            The evaluated pdf at point x.
        """
        minimum = input_values[2]
        if x < minimum:
            pdf = 0.
        else:
            mu = input_values[0]
            sigma = input_values[1]
            pdf = scipy.stats.norm(mu, sigma).pdf(x)
        self.calculated_pdf = pdf
        return pdf


def run_optimisation_abcpy(eventWise_name, batch_size=100, end_time=None,
                               total_calls=10000, silent=True, **kwargs):
    # make the translator
    print("setting up optimisation", flush=True)
    jet_class = kwargs.get("jet_class", FormJets.SpectralFull)
    if isinstance(jet_class, str):
        jet_class = getattr(FormJets, jet_class)
    if "fixed_params" in kwargs:
        fixed_params = kwargs["fixed_params"]
    elif jet_class == FormJets.SpectralFull:
        fixed_params = dict(StoppingCondition='meandistance',
                            EigDistance='abscos',
                            Laplacien='symmetric',
                            PhyDistance='angular',
                            CombineSize='sum',
                            ExpofPTFormat='Luclus',
                            #ExpofPTPosition='input',
                            AffinityType='exponent')
    elif jet_class == FormJets.SpectralKMeans:
        fixed_params = dict(EigDistance='abscos',
                            Laplacien='symmetric',
                            PhyDistance='angular',
                            ExpofPTFormat='Luclus',
                            ExpofPTPosition='input',
                            AffinityType='exponent')
    else:
        raise NotImplementedError
    translator = ParameterTranslator(jet_class, fixed_params)
    varaible_list = translator.generate_abcpy_variables()
    statistics_calc = abcpy.statistics.Identity()
    distance_calc = abcpy.distances.Euclidean(statistics_calc)
    #distance_calc = abcpy.distances.LogReg(statistics_calc, seed=42)
    backend = abcpy.backends.BackendDummy()
    #backend = abcpy.backends.BackendMPI()
    print("defining objective", flush=True)
    objective = [np.array(0)]
    # parameters for the optimiser
    #eps_init = np.array([10000])
    eps_init = np.array([70])
    #eps_init = np.array([0.75])
    #n_samples = 10000
    #n_samples = 40
    n_samples = 10
    n_samples_per_param = 5
    epsilon_percentile = 60
    #epsilon_percentile = 10
    duration = kwargs.get("duration", 5*60)
    log_dir = kwargs.get("log_dir", "./logs")
    # set up a loop to restart the optimiser untill it finishes
    journal_name_list = []
    if 'last_journal' in kwargs:
        journal_name_list.append(kwargs['last_journal'])
    print("creating model", flush=True)
    model = ClusteringModel(varaible_list, eventWise_path=eventWise_name,
                            batch_size=batch_size, test_size=500, translator=translator)
    print("creating sampler", flush=True)
    kernal = abcpy.perturbationkernel.DefaultKernel(varaible_list)
    sampler = TimedPMCABC([model], [distance_calc], backend, kernal, seed=1)
    #sampler = abcpy_new_inferences.PMCABC([model], [distance_calc], backend, kernal, seed=1)
    print(f"Running for {duration/(60*60):.1f} hours aprox", flush=True)
    journal_name, journal = sampler.sample([objective], duration, eps_init,
                                           n_samples, n_samples_per_param,
                                           epsilon_percentile, full_output=True,
                                           journal_file=journal_name_list,
                                           log_dir=log_dir)
    journal.configuration["fixed_parameters"] = translator.fixed_parameters
    if not isinstance(jet_class, str):
        jet_class = jet_class.__name__
    journal.configuration["jet_class"] = jet_class
    journal.save(journal_name)
    print_journal_to_log(translator, journal_name_list[-1])


def journal_to_log(translator, journal_name):
    fixed_params = translator.fixed_params
    journal = abcpy.output.Journal.fromFile(journal_name)
    hyper_to_log = ["score", "weights"]
    param_to_log = list(translator.unfixed_order)
    scores = np.concatenate(journal.distances).reshape((-1, 1))
    weights = np.concatenate(journal.weights)
    hyper_log = np.hstack((scores, weights))
    # now loop over each step getting the parameters
    items_per_step = len(journal.weights[0])
    # this will come in raw optimiser format
    raw_log = np.empty((len(scores), len(param_to_log)), dtype=float)
    for step_n, params in enumerate(journal.names_and_parameters):
        step_start = step_n * items_per_step
        for col, name in enumerate(param_to_log):
            # it's nested in a stupid way
            for i, val in enumerate(params[name]):
                raw_log[step_start + i, col] = val[0]
    # then loop through again and convert witht eh translator
    param_log = []
    for row in raw_log:
        converted = translator.list_to_clustering(row)
        param_log.append([converted[name] for name in param_to_log])
    # now make the best score
    best_idx = np.nanargmin(scores)
    best_params = {name: param_log[best_idx][col]
                   for col, name in enumerate(param_to_log)}
    best_params.update(fixed_params)
    return hyper_to_log, hyper_log, param_to_log, param_log, best_params, journal_name


def print_journal_to_log(translator, journal_name):
    log_args = journal_to_log(translator, journal_name)
    string = log_text(*log_args)
    file_name = journal_name + '.txt'
    with open(file_name, 'w') as log_file:
        log_file.write(string)


def journal_to_best(journal, num_best=100, max_steps_back=None):
    if isinstance(journal, str):
        journal = abcpy.output.Journal.fromFile(journal)
    jet_class = journal.configuration.get("jet_class", "SpectralFull")
    if "fixed_parameters" in journal.configuration:
        fixed_params = journal.configuration["fixed_parameters"]
    else:
        fixed_params = dict(StoppingCondition='meandistance',
                            EigDistance='abscos',
                            Laplacien='symmetric',
                            PhyDistance='angular',
                            CombineSize='sum',
                            ExpofPTFormat='Luclus',
                            #ExpofPTPosition='input',
                            AffinityType='exponent')
    translator = ParameterTranslator(jet_class, fixed_params)
    jet_class = translator.jet_class  # converts out of str

    if max_steps_back is None:
        max_steps_back = 0
    distances = awkward.fromiter(journal.distances[-max_steps_back:]
                                 ).flatten()
    parameters = awkward.fromiter(journal.accepted_parameters[-max_steps_back:]
                                  ).flatten()
    best = []
    for idx in np.argsort(distances)[:num_best]:
        params = parameters[idx].flatten()
        best.append(translator.list_to_clustering(params))
    return jet_class, best

def cluster_from_journal(eventWise, journal,
                         dijet_mass=40, num_best=100, max_steps_back=None):
    if isinstance(eventWise, str):
        eventWise = Components.EventWise.from_file(eventWise)
    jet_class, best = journal_to_best(journal, num_best, max_steps_back)
    for i, params in enumerate(best):
        print(f"{i} of {num_best}\n", flush=True)
        jet_name = f"OptimisedJet{i}"
        try:
            warnings.filterwarnings('ignore')
            FormJets.cluster_multiapply(eventWise, jet_class, params, jet_name, np.inf)
        except Exception as e:
            print(f"couldnt cluster params {params}")
            print(e)
    print("Scoring")
    CompareClusters.append_scores(eventWise, dijet_mass=dijet_mass, overwrite=False)

##################### logging ############################

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

def reserve_name(name_form):
    assert "}" in name_form, "Need '{}' to be able to format name_form"
    i = 0
    while True:
        try:
            open(name_form.format(i), 'x').close()
            name = name_form.format(i)
            return name, i
        except FileExistsError:
            i += 1

def print_log(hyper_to_log, hyper_log,
              params_to_log, param_log, full_final_params,
              log_dir="./logs", other_records=None,
              file_name=None):
    try:
        os.mkdir(log_dir)
    except FileExistsError:
        pass
    text = log_text(hyper_to_log, hyper_log, params_to_log, param_log, full_final_params, other_records)
    log_name = os.path.join(log_dir, "log{:03d}.txt")
    file_name, index = reserve_name(log_name)
    with open(file_name, 'w') as new_file:
        new_file.write(text)
    return index


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
    y_max = np.nanmax(scores[int(len(scores)*0.70):])*1.5
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
            warnings.filterwarnings('ignore')
            FormJets.cluster_multiapply(eventWise, jet_class, params, jet_name, np.inf)
        except Exception as e:
            print(f"couldnt cluster params {p_str}")
            print(e)
    print("Scoring")
    CompareClusters.append_scores(eventWise, dijet_mass=dijet_mass, overwrite=False)


if __name__ == '__main__':
    #run_optimisation_abcpy("megaIgnore/show.awkd")
    if InputTools.yesNo_question("Plot run? "):
        visulise_logs()
    elif InputTools.yesNo_question("Optimise with nevergrad? "):
        run_time = InputTools.get_time("How long should it run?")
        eventWise_name = InputTools.get_file_name("Name the eventWise: ").strip()
        log_dir = "./logs"
        generate_pool(eventWise_name, duration=run_time, log_dir=log_dir)
    elif InputTools.yesNo_question("Optimise with abcpy? "):
        run_time = InputTools.get_time("How long should it run?")
        eventWise_name = InputTools.get_file_name("Name the eventWise: ").strip()
        log_dir = "./logs"
        if InputTools.yesNo_question("Add journal? "):
            journal_name = InputTools.get_file_name("Journal file; ", '.jnl').strip()
            run_optimisation_abcpy(eventWise_name, duration=run_time, log_dir=log_dir, last_journal=journal_name)
        else:
            run_optimisation_abcpy(eventWise_name, duration=run_time, log_dir=log_dir)


