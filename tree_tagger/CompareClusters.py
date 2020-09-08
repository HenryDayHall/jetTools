""" compare two jet clustering techniques """
import multiprocessing
import tabulate
import awkward
import ast
import csv
import time
import os
import pickle
import matplotlib
import awkward
from ipdb import set_trace as st
from tree_tagger import Components, TrueTag, InputTools, FormJets, Constants, RescaleJets, FormShower, JetQuality, PlottingTools
import sklearn.metrics
import sklearn.preprocessing
from matplotlib import pyplot as plt
import numpy as np
import scipy.stats
import bokeh, bokeh.palettes, bokeh.models, bokeh.plotting, bokeh.transform
import socket


def get_best(eventWise, jet_class):
    """ return the name of the jet with the highest SignalMassRatio/BGMassRatio """
    scored_names = [name.split('_', 1)[0] for name in eventWise.hyperparameter_columns
                    if jet_class in name and name.endswith("AveSignalMassRatio")]
    score = np.fromiter((getattr(eventWise, name+"_AveSignalMassRatio")
                         /getattr(eventWise, name+"_AveBGMassRatio")
                         for name in scored_names), dtype=float)
    try:
        best_name = scored_names[np.nanargmax(score)]
    except ValueError as e:
        # this amy be becuase there are no jets of this class
        if not scored_names:
            err_message = f"no jets of class {jet_class} in {eventWise.save_name}"
            raise ValueError(err_message)
        # or it may be becuase all the scores were nan
        if np.all(np.isnan(score)):
            # then return the first
            return scored_names[0]
        # else, raise the original error
        raise e from None
    return best_name


# code for making scores

def add_bg_mass(eventWise):
    eventWise.selected_index = None
    n_events = len(eventWise.BQuarkIdx)
    all_bg_mass = np.zeros(n_events)
    for event_n in range(n_events):
        eventWise.selected_index = event_n
        source_idx = eventWise.JetInputs_SourceIdx
        detectable_idx = eventWise.DetectableTag_Leaves
        # rember that each tag gets it's own list of detectables
        bg = list(set(source_idx)  - set(detectable_idx.flatten()))
        all_bg_mass[event_n] = np.sum(eventWise.Energy[bg])**2 - np.sum(eventWise.Px[bg])**2 -\
                               np.sum(eventWise.Py[bg])**2 - np.sum(eventWise.Pz[bg])**2
    all_bg_mass = np.sqrt(all_bg_mass)
    eventWise.append(DetectableBG_Mass = awkward.fromiter(all_bg_mass))


def get_detectable_comparisons(eventWise, jet_name, jet_idxs, append=False):
    """
    

    Parameters
    ----------
    eventWise :
        
    jet_name :
        
    jet_idxs :
        
    ctag :
        

    Returns
    -------

    """
    if "DetectableTag_PT" not in eventWise.columns:
        Components.add_phi(eventWise, "DetectableTag")
        Components.add_PT(eventWise, "DetectableTag")
        Components.add_rapidity(eventWise, "DetectableTag")
    if "DetectableTag_Mass" not in eventWise.columns:
        Components.add_mass(eventWise, "DetectableTag")
    if "DetectableBG_Mass" not in eventWise.columns:
        add_bg_mass(eventWise)
    eventWise.selected_index = None
    tag_groups = eventWise.DetectableTag_Roots
    n_events = len(tag_groups)
    # we assume the tagger behaves perfectly
    # for all jets allocated to each tag group
    # calcualte total contained values
    tag_mass2_in = [[] for _ in range(n_events)]
    bg_mass2_in = [[] for _ in range(n_events)]
    rapidity_in = [[] for _ in range(n_events)]
    phi_in = [[] for _ in range(n_events)]
    pt_in = [[] for _ in range(n_events)]
    # the fraction of the tags that have been connected to some jet
    percent_found = np.zeros(n_events)
    for event_n, event_tags in enumerate(tag_groups):
        # it is possible to have no event tags
        # this happens whn the tags create no detectable particles
        if len(event_tags) == 0:
            # if there isn't anything to be found then ignore this event
            percent_found[event_n] = np.nan
            continue # then skip
        # if we get here, there are detectable particles from the tags
        eventWise.selected_index = event_n
        energy = eventWise.Energy
        px = eventWise.Px
        py = eventWise.Py
        pz = eventWise.Pz
        source_idx = eventWise.JetInputs_SourceIdx
        parent_idxs = getattr(eventWise, jet_name + "_Parent")
        tagmass = getattr(eventWise, jet_name + "_TagMass")
        tag_idxs = eventWise.BQuarkIdx
        matched_jets = [[] for _ in event_tags]
        for jet_n, jet_tags in enumerate(getattr(eventWise, jet_name + "_MTags")):
            if jet_n not in jet_idxs[event_n] or len(jet_tags) == 0:
                continue  # jet not sutable or has no tags
            if len(jet_tags) == 1:
                # no chosing to be done, the jet just has one tag
                tag_idx = jet_tags[0]
            else:
                # chose the tag with the greatest tagmass
                # dimension 0 of tagmass is which jet
                # dimension 1 of tagmass is which tag
                tag_position = np.argmax(tagmass[jet_n])
                # this is the particle index of the tag with greatest massshare in the jet
                tag_idx = tag_idxs[tag_position]
            # which group does the tag belong to
            group_position = next(i for i, group in enumerate(event_tags) if tag_idx in group)
            matched_jets[group_position].append(jet_n)
        # the tag fragment accounts only for tags that could be found
        num_found = sum(len(group) for group, matched in zip(event_tags, matched_jets)
                                 if len(matched))
        percent_found[event_n] = num_found/len(event_tags.flatten())
        for group_n, jets in enumerate(matched_jets):
            if jets:
                jet_inputs = getattr(eventWise, jet_name + "_InputIdx")[jets].flatten()
                # convert to source_idxs
                jet_inputs = jet_inputs[jet_inputs < len(source_idx)]
                jet_inputs = set(source_idx[jet_inputs])
                tag_in_jet = jet_inputs.intersection(eventWise.DetectableTag_Leaves[group_n])
                bg_in_jet = list(jet_inputs - tag_in_jet)
                tag_in_jet = list(tag_in_jet)
                tag_mass2 = np.sum(energy[tag_in_jet])**2 -\
                           np.sum(px[tag_in_jet])**2 -\
                           np.sum(py[tag_in_jet])**2 -\
                           np.sum(pz[tag_in_jet])**2
                bg_mass2 = np.sum(energy[bg_in_jet])**2 -\
                          np.sum(px[bg_in_jet])**2 -\
                          np.sum(py[bg_in_jet])**2 -\
                          np.sum(pz[bg_in_jet])**2
                # for the pt and the phi comparisons use all the jet components
                # not just the ones that come from the truth
                jet_inputs = list(jet_inputs)
                phi, pt = Components.pxpy_to_phipt(np.sum(px[jet_inputs]),
                                                   np.sum(py[jet_inputs]))
                rapidity = Components.ptpze_to_rapidity(pt, np.sum(pz[jet_inputs]),
                                                        np.sum(energy[jet_inputs]))
            else:
                # no jets in this group
                tag_mass2 = bg_mass2 = 0
                phi = pt = rapidity = np.nan
            tag_mass2_in[event_n].append(tag_mass2)
            bg_mass2_in[event_n].append(bg_mass2)
            phi_in[event_n].append(phi)
            pt_in[event_n].append(pt)
            rapidity_in[event_n].append(rapidity)
    eventWise.selected_index = None
    content = {}
    rapidity_distance = awkward.fromiter(rapidity_in) - eventWise.DetectableTag_Rapidity
    pt_distance = awkward.fromiter(pt_in) - eventWise.DetectableTag_PT
    phi_distance = awkward.fromiter(phi_in) - eventWise.DetectableTag_Phi
    content[jet_name + "_DistanceRapidity"] = np.abs(rapidity_distance)
    content[jet_name + "_DistancePT"] = np.abs(pt_distance)
    content[jet_name + "_DistancePhi"] = np.abs(phi_distance)
    tag_mass_in = np.sqrt(awkward.fromiter(tag_mass2_in))
    content[jet_name + "_SignalMassRatio"] = tag_mass_in/eventWise.DetectableTag_Mass
    bg_mass_in = np.sqrt(awkward.fromiter(bg_mass2_in))
    content[jet_name + "_BGMassRatio"] = bg_mass_in/eventWise.DetectableBG_Mass
    content[jet_name + "_PercentFound"] = awkward.fromiter(percent_found) 
    if append:
        eventWise.append(**content)
    return content


def get_mass_gaps(eventWise, jet_name, jet_idxs, append=False):
    """ Should be combined with get detectable?"""
    # get the distance between the signaljetmass and the desired tag mass
    # get the distance between the the signaljet mass ans the whole jet mass
    if "DetectableTag_PT" not in eventWise.columns:
        Components.add_phi(eventWise, "DetectableTag")
        Components.add_PT(eventWise, "DetectableTag")
        Components.add_rapidity(eventWise, "DetectableTag")
    if "DetectableTag_Mass" not in eventWise.columns:
        Components.add_mass(eventWise, "DetectableTag")
    if "DetectableBG_Mass" not in eventWise.columns:
        add_bg_mass(eventWise)
    eventWise.selected_index = None
    tag_groups = eventWise.DetectableTag_Roots
    n_events = len(tag_groups)
    # we assume the tagger behaves perfectly
    # for all jets allocated to each tag group
    # calcualte total contained values
    tag_mass2_in = [[] for _ in range(n_events)]
    all_mass2_in = [[] for _ in range(n_events)]
    for event_n, event_tags in enumerate(tag_groups):
        # it is possible to have no event tags
        # this happens whn the tags create no detectable particles
        if len(event_tags) == 0:
            # if there isn't anything to be found then ignore this event
            continue # then skip
        # if we get here, there are detectable particles from the tags
        eventWise.selected_index = event_n
        energy = eventWise.Energy
        px = eventWise.Px
        py = eventWise.Py
        pz = eventWise.Pz
        source_idx = eventWise.JetInputs_SourceIdx
        parent_idxs = getattr(eventWise, jet_name + "_Parent")
        tagmass = getattr(eventWise, jet_name + "_TagMass")
        tag_idxs = eventWise.BQuarkIdx
        matched_jets = [[] for _ in event_tags]
        for jet_n, jet_tags in enumerate(getattr(eventWise, jet_name + "_MTags")):
            if jet_n not in jet_idxs[event_n] or len(jet_tags) == 0:
                continue  # jet not sutable or has no tags
            if len(jet_tags) == 1:
                # no chosing to be done, the jet just has one tag
                tag_idx = jet_tags[0]
            else:
                # chose the tag with the greatest tagmass
                # dimension 0 of tagmass is which jet
                # dimension 1 of tagmass is which tag
                tag_position = np.argmax(tagmass[jet_n])
                # this is the particle index of the tag with greatest massshare in the jet
                tag_idx = tag_idxs[tag_position]
            # which group does the tag belong to
            group_position = next(i for i, group in enumerate(event_tags) if tag_idx in group)
            matched_jets[group_position].append(jet_n)
        for group_n, jets in enumerate(matched_jets):
            if jets:
                jet_inputs = getattr(eventWise, jet_name + "_InputIdx")[jets].flatten()
                # convert to source_idxs
                jet_inputs = jet_inputs[jet_inputs < len(source_idx)]
                jet_inputs = source_idx[jet_inputs]
                # get the tag mass
                tag_in_jet = list(set(jet_inputs).intersection(eventWise.DetectableTag_Leaves[group_n]))
                tag_mass2 = np.sum(energy[tag_in_jet])**2 -\
                            np.sum(px[tag_in_jet])**2 -\
                            np.sum(py[tag_in_jet])**2 -\
                            np.sum(pz[tag_in_jet])**2
                # get the whole jets mass
                all_mass2 = np.sum(energy[jet_inputs])**2 -\
                           np.sum(px[jet_inputs])**2 -\
                           np.sum(py[jet_inputs])**2 -\
                           np.sum(pz[jet_inputs])**2
            else:
                # no jets in this group
                tag_mass2 = all_mass2 = 0
            tag_mass2_in[event_n].append(tag_mass2)
            all_mass2_in[event_n].append(all_mass2)
    eventWise.selected_index = None
    content = {}
    tag_mass_in = np.sqrt(awkward.fromiter(tag_mass2_in))
    signal_distance = np.abs(tag_mass_in - eventWise.DetectableBG_Mass)
    background_distance = np.abs(tag_mass_in - np.sqrt(awkward.fromiter(all_mass2_in)))
    content[jet_name + "_DistanceSignal"] = signal_distance
    content[jet_name + "_DistanceBG"] = background_distance
    if append:
        eventWise.append(**content)
    return content



def filter_jets(eventWise, jet_name, min_jetpt=None, min_ntracks=None):
    if min_jetpt is None:
        min_jetpt = Constants.min_jetpt
    if min_ntracks is None:
        min_ntracks = Constants.min_ntracks
    jet_idxs = []
    eventWise.selected_index = None
    jet_pt = getattr(eventWise, jet_name + "_PT")
    jet_parent = getattr(eventWise, jet_name + "_Parent")
    jet_child1 = getattr(eventWise, jet_name + "_Child1")
    for pts, parents, child1s in zip(jet_pt, jet_parent, jet_child1):
        pt_passes = np.where(pts[parents==-1].flatten() > min_jetpt)[0]
        long_enough = awkward.fromiter((i for i, children in zip(pt_passes, child1s[pt_passes])
                                        if sum(children == -1) >= min_ntracks))
        jet_idxs.append(long_enough)
    return awkward.fromiter(jet_idxs)


def remove_scores(eventWise):
    suffixes_to_remove = ["DistancePT", "DistancePhi", "DistanceRapidity", "QualityWidth",
                          "QualityFraction", "PercentFound", "BGMassRatio", "SignalMassRatio"]
    if isinstance(eventWise, str):
        eventWise = Components.EventWise.from_file(eventWise)
    remove_cols = [name for name in eventWise.columns + eventWise.hyperparameter_columns
                   if np.any([name.endswith(suf) for suf in suffixes_to_remove])]
    if remove_cols:
        for name in remove_cols:
            eventWise.remove(name)
        eventWise.write()


def append_scores(eventWise, dijet_mass=None, end_time=None, duration=np.inf, overwrite=False, silent=False):
    if isinstance(eventWise, str):
        eventWise = Components.EventWise.from_file(eventWise)
    if dijet_mass is None:
        dijet_mass = Constants.dijet_mass
    eventWise_path = os.path.join(eventWise.dir_name, eventWise.save_name)
    new_hyperparameters = {}
    new_contents = {}
    names = FormJets.get_jet_names(eventWise)
    num_names = len(names)
    save_interval = 10
    if end_time is None:
        end_time = duration + time.time()
    if "DetectableTag_Idx" not in eventWise.columns:
        TrueTag.add_detectable_fourvector(eventWise)
    for i, name in enumerate(names):
        if not silent:
            print(f"\n{i/num_names:.1%}\t{name}\n" + " "*10, flush=True)
        # pick up some content
        content_here = {}
        # check if it has already been scored
        if name + "_QualityWidth" not in eventWise.hyperparameter_columns or overwrite:
            if not silent:
                print("Adding quality scores")
            # if we reach here the jet still needs a score
            try:
                best_width, best_fraction = JetQuality.quality_width_fracton(eventWise, name,
                                                                             dijet_mass)
            except (ValueError, RuntimeError):  # didn't make enough masses
                best_width = best_fraction = np.nan
            new_hyperparameters[name + "_QualityWidth"] = best_width
            new_hyperparameters[name + "_QualityFraction"] = best_fraction
        if name + "_DistancePT" not in eventWise.columns or overwrite:
            if not silent:
                print("Adding kinematic scores")
            # now the mc truth based scores
            if "DetectableTag_Idx" not in eventWise.columns:
                TrueTag.add_mass_share(eventWise, name, batch_length=np.inf)
            jet_idxs = filter_jets(eventWise, name)
            content_here.update(get_detectable_comparisons(eventWise, name, jet_idxs, False))
        # the detectable mass dirnces are new
        if name + "_DistanceSignal" not in eventWise.columns or overwrite:
            print("Adding mass scores")
            if "DetectableTag_Idx" not in eventWise.columns:
                TrueTag.add_mass_share(eventWise, name, batch_length=np.inf)
            jet_idxs = filter_jets(eventWise, name)
            content_here.update(get_mass_gaps(eventWise, name, jet_idxs, False))
        # get averages for all other generated content
        new_averages = {}
        # we are only intrested in finite results
        for key, values in content_here.items():
            flattened = values.flatten()
            finite = np.isfinite(flattened)
            if np.any(finite):
                value = np.mean(flattened[finite])
            else:  # sometimes there could be no finite results at all
                value = np.nan
            new_averages[key.replace('_', '_Ave')] = value
        new_contents.update(content_here)
        new_hyperparameters.update(new_averages)
        if not os.path.exists('continue'):
            eventWise.append_hyperparameters(**new_hyperparameters)
            eventWise.append(**new_contents)
            return
        if (i+1)%save_interval == 0:
            eventWise.append_hyperparameters(**new_hyperparameters)
            eventWise.append(**new_contents)
            new_hyperparameters = {}
            new_contents = {}
            # at each save interval also load the eventWise afresh
            eventWise = Components.EventWise.from_file(eventWise_path)
        if time.time() > end_time:
            break
    eventWise.append_hyperparameters(**new_hyperparameters)
    eventWise.append(**new_contents)


def multiprocess_append_scores(eventWise_paths, end_time, overwrite=False, leave_one_free=True):
    n_paths = len(eventWise_paths)
    # cap this out at 20, more seems to create a performance hit
    n_threads = np.min((multiprocessing.cpu_count()-leave_one_free, 20, n_paths))
    if n_threads < 1:
        n_threads = 1
    wait_time = 0.9*(end_time - time.time())  # in seconds
    # note that the longest wait will be n_cores time this time
    print("Running on {} threads".format(n_threads))
    job_list = []
    # now each segment makes a worker
    args = [(path, None, end_time, None, overwrite, True)
            for path in eventWise_paths]
    #args = [(path,) for path in eventWise_paths]
    # set up some initial jobs
    for _ in range(n_threads):
        job = multiprocessing.Process(target=append_scores, args=args.pop())
        #job = multiprocessing.Process(target=remove_scores, args=args.pop())
        job.start()
        job_list.append(job)
    processed = 0
    for dataset_n in range(n_paths):
        job = job_list[dataset_n]
        job.join(wait_time)
        processed += 1
        # check if we shoudl stop
        if end_time - time.time() < wait_time/10:
            break
        if args:  # make a new job
            job = multiprocessing.Process(target=append_scores, args=args.pop())
            job.start()
            job_list.append(job)
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
    remaining_paths = eventWise_paths[-len(args):]
    print(f"Num remaining jobs {len(remaining_paths)}")
    return True

# plotting code

def tabulate_scores(eventWise_paths, variable_cols=None, score_cols=None):
    if score_cols is None:
        score_cols = ["QualityWidth", "QualityFraction", "AveSignalMassRatio", "AveBGMassRatio",
                      "AveDistancePT", "AveDistancePhi", "AveDistanceRapidity", "AvePercentFound"]
    if variable_cols is None:
        classes = ["Traditional", "SpectralMean", "SpectralFull", "Splitting", "Indicator"]
        variable_cols = set()
        for name in classes:
            variable_cols.update(getattr(FormJets, name).default_params.keys())
        variable_cols = sorted(variable_cols)
    # also record jet class, eventWise.svae_name and jet_name
    all_cols = ["jet_name", "jet_class", "eventWise_name"] + variable_cols + score_cols
    table = []
    for path in eventWise_paths:
        eventWise = Components.EventWise.from_file(path)
        eventWise_name = eventWise.save_name
        jet_names = FormJets.get_jet_names(eventWise)
        for name in jet_names:
            row = [name, name.split("Jet", 1)[0], eventWise_name]
            row += [getattr(eventWise, name+'_'+var, np.nan) for var in variable_cols]
            row += [getattr(eventWise, name+'_'+sco, np.nan) for sco in score_cols]
            table.append(row)
    table = awkward.fromiter(table)
    return all_cols, variable_cols, score_cols, table


def filter_table(*args):
    args = nan_filter_table(*args)
    print(f"Length after nan filter = {len(args[-1])}")
    table = args[-1]
    if len(table) > 200:
        args = quality_filter_table(*args)
        print(f"Length after quality filter = {len(args[-1])}")
    return args


def nan_filter_table(all_cols, variable_cols, score_cols, table):
    # make a mask marking the location of nan
    nan_mask = []
    for row in table:
        nan_mask.append([])
        for x in row:
            try:
                # by calling any on the is_nan
                # we throw a ValueError if x is actually a jaggedArray
                nan_mask[-1].append(np.any(np.isnan(x)))
            except TypeError:
                nan_mask[-1].append(False)
    nan_mask = np.array(nan_mask)
    if np.any(nan_mask):
        # drop any rows where any score_cols are nan
        score_nan = nan_mask[:, [all_cols.index(name) for name in score_cols]]
        any_nan = np.any(score_nan, axis=1)
        table = table[~any_nan]
        nan_mask = nan_mask[~any_nan]
        # then drop any cols where all values are np.nan
        drop_cols = np.fromiter((np.all(nan_mask[:, i]) for i, name in enumerate(all_cols)),
                                dtype=bool)
        for i in np.where(drop_cols)[0][::-1]:
            name = all_cols.pop(i)
            if name in variable_cols:
                del variable_cols[variable_cols.index(name)]
            elif name in score_cols:
                del score_cols[score_cols.index(name)]
        table = awkward.fromiter([row[~drop_cols] for row in table])
    return all_cols, variable_cols, score_cols, table


def quality_filter_table(all_cols, variable_cols, score_cols, table):
    # require at least 0.1 PercentFound
    percent_found = table[:, all_cols.index("AvePercentFound")]
    table = table[percent_found > 0.1]
    # require that rapidity distance be no more than 0.4
    rapidity_distance = table[:, all_cols.index("AveDistanceRapidity")]
    table = table[rapidity_distance < 0.4]
    # require that phi distance be no more than 1.
    phi_distance = table[:, all_cols.index("AveDistancePhi")]
    table = table[phi_distance < 1.]
    # require that PT distance be no more than 15.
    pt_distance = table[:, all_cols.index("AveDistancePT")]
    table = table[pt_distance < 15.]
    # require average BG mass ratio under 10k
    bg_mass_ratio = table[:, all_cols.index("AveBGMassRatio")]
    table = table[bg_mass_ratio < 1e4]
    # require that average Siganl ratio be over 0.1
    sig_mass_ratio = table[:, all_cols.index("AveSignalMassRatio")]
    table = table[sig_mass_ratio > 0.1]
    # require that quality fractio be under 600
    quality_fraction = table[:, all_cols.index("QualityFraction")]
    table = table[np.logical_or(quality_fraction < 600, ~np.isnan(quality_fraction.tolist()))]
    # require that quality width be under 0.1
    quality_width = table[:, all_cols.index("QualityWidth")]
    table = table[np.logical_or(quality_width < 0.1, ~np.isnan(quality_width.tolist()))]
    return all_cols, variable_cols, score_cols, table


def plot_class_bests(eventWise_paths, save_prefix=None):
    all_cols, variable_cols, score_cols, table = filter_table(*tabulate_scores(eventWise_paths))
    variable_cols = ['jet_class'] + variable_cols
    class_col = all_cols.index("jet_class")
    jet_col = all_cols.index("jet_name")
    eventWise_col = all_cols.index("eventWise_name")
    inverted_names = ["QualityWidth", "QualityFraction", "AveDistancePT", "AveDistancePhi",
                      "AveDistanceRapidity", "AveBGMassRatio"]
    output1 = [["jet class"] + score_cols]
    # to identify the best in each score type keep numeric values
    numeric_scores = []
    row_nums = []
    #  pick the jet for each class that creates the highest meta_score
    # AveSignalMassRatio*AvePercentFound*AvePTDistance
    mass_ratio_col, percent_found_col, distance_pt_col = [all_cols.index(name) for name in
                                ["AveSignalMassRatio", "AvePercentFound", "AveDistancePT"]]
    meta_score = np.fromiter((row[mass_ratio_col]*row[percent_found_col]/row[distance_pt_col]
                              for row in table), dtype=float)
    output3 = [["file name", "jet name"] + score_cols]
    for class_name in set(table[:, class_col]):
        row_indices, rows = zip(*[(i, row) for i, row in enumerate(table)
                                  if row[class_col] == class_name])
        best_meta = np.argmax(meta_score[list(row_indices)])
        output3.append([rows[best_meta][eventWise_col][7:], rows[best_meta][jet_col]])
        jet_names = [class_name]
        file_names = ["~  in   ~"]
        scores =     ["~ score ~"]
        numeric_scores.append([])
        row_nums.append([])
        for score_name in score_cols:
            score_col = all_cols.index(score_name)
            output3[-1].append(f"{rows[best_meta][score_col]:.3g}")
            class_scores = [row[score_col] for row in rows]
            if score_name in inverted_names:
                best = np.nanargmin(class_scores)
                numeric_scores[-1].append(-class_scores[best])
            else:
                best = np.nanargmax(class_scores)
                numeric_scores[-1].append(class_scores[best])
            row_nums[-1].append(row_indices[best])
            scores.append(f"{class_scores[best]:.3g}")
            best_row = rows[best]
            jet_names.append(best_row[jet_col])
            file_names.append(best_row[eventWise_col][7:])
        output1 += [jet_names, file_names, scores]
    # now identify and label the best
    numeric_scores = np.array(numeric_scores)
    # for each thign that was the best in critera, 
    # make another table of all it's other scores
    output2 = [["jet_name"] + score_cols]
    for col, criteria in enumerate(numeric_scores.T):
        # mark the best in the first tbale
        best = np.nanargmax(criteria)
        #    header+ 3 rows per class+row 2 in class block
        output_row = 1    + 3*best          + 2
        #             header
        output1[output_row][col+1] += "*best*"
        # add the best to the secodn table
        jet_row = table[row_nums[best][col]]
        jet_scores = [f"{jet_row[all_cols.index(name)]:.4g}" for name in score_cols]
        jet_scores[col] += "*best*"
        output2.append([jet_row[all_cols.index("jet_name")]] + jet_scores)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    PlottingTools.hide_axis(ax1)
    PlottingTools.hide_axis(ax2)
    PlottingTools.hide_axis(ax3)
    PlottingTools.text_table(ax1, output1, "22.22")
    PlottingTools.text_table(ax2, output2, "20.20")
    PlottingTools.text_table(ax3, output3, "20.20")
    if save_prefix:
        plt.savefig(save_prefix+"_class_bests.png")
    return output1, output2


def plot_grid(all_cols, plot_column_names, plot_row_names, table):
    # a list of cols where the axis should be inverted so that up is still better
    inverted_names = ["QualityWidth", "QualityFraction", "AveDistancePT", "AveDistancePhi",
                      "AveDistanceRapidity", "AveBGMassRatio"]
    # a list of cols where the axis must be constructed
    impure_cols = ["jet_class", "NumEigenvectors", "ExpofPTPosition", "ExpofPTFormat", "AffinityType", "AffinityCutoff",
            "Laplacien", "PhyDistance", "StoppingCondition", "MaxJump", "MaxCutScore", "Eigenspace",
                   "BaseJump"]
    n_cols = len(plot_column_names)
    n_rows = len(plot_row_names)
    fig, ax_arr = plt.subplots(n_rows, n_cols, sharex='col', sharey='row')
    # give each of the clusters a random colour and marker shape
    # colours = np.random.rand(len(table)*4).reshape((len(table), 4))
    # colour each cluster by jet_class and AvePercentFound
    class_colour_dict = {'SpectralFull': 'Blues', 'SpectralMean': 'Greens',
                         'Splitting': 'Oranges', 'Indicator': 'Reds',
                         'Traditional': 'Greys'}
    colours = np.zeros((len(table), 4))
    for name in class_colour_dict:
        cmap = matplotlib.cm.get_cmap(class_colour_dict[name])
        rows = [c==name for c in table[:, all_cols.index('jet_class')]]
        values = np.fromiter(table[rows, all_cols.index('AvePercentFound')],
                             dtype=float)
        # rescale to make better use of the colour range
        if len(values) > 0:
            max_val = np.nanmax(values) 
            if np.isfinite(max_val):
                values = values/max_val
            colours[rows] = cmap(values)
            colours[rows, -1] = values  # increasing alpha with increasing percent found
    #markers = np.random.choice(['v', 's', '*', 'D', 'P', 'X'], len(table))
    plotting_params = dict(c=colours) #, marker=markers)
    for col_n, col_name in enumerate(plot_column_names):
        x_positions = table[:, all_cols.index(col_name)]
        # this function will decided what kind of scale and create it
        if col_name in impure_cols:
            x_positions, x_scale_positions, x_scale_labels = make_scale(x_positions)
        else:
            x_positions = x_positions.tolist()  # awkward arrays don't play well
        for row_n, row_name in enumerate(plot_row_names):
            y_positions = table[:, all_cols.index(row_name)]
            if row_name in impure_cols:
                y_positions, y_scale_positions, y_scale_labels = make_scale(y_positions)
            else:
                y_positions = y_positions.tolist()
            ax = ax_arr[row_n, col_n]
            ax.scatter(x_positions, y_positions, **plotting_params)
            if row_name in inverted_names:
                ax.invert_yaxis()
            if col_name in inverted_names:
                ax.invert_xaxis()
            if row_n == n_rows-1:
                ax.set_xlabel(col_name)
                if col_name in impure_cols:  # then we have a custom scale
                    ax.set_xticks(x_scale_positions)
                    ax.set_xticklabels(x_scale_labels, rotation=90)
            if col_n == 0:
                ax.set_ylabel(row_name)
                if row_name in impure_cols:
                    ax.set_yticks(y_scale_positions)
                    ax.set_yticklabels(y_scale_labels)
    # the title at the top required 0.4 of an inch
    title_height = 0.4
    # the bottom and the left may require up to 1.5 inches
    attributes_height = 1.5
    # then there is a max likely screen size of 11.75 by 21 inches
    row_height = min((11.75 - title_height - attributes_height)/n_rows, 2.)
    title = row_height > 1.  # lose the title space if it's too short
    if not title:
        row_height = min((11.75 - attributes_height)/n_rows, 2.)
    col_width = min((11.75 - attributes_height)/n_cols, 2.)
    height = int(title)*title_height + attributes_height + row_height*n_rows
    width = attributes_height + col_width*n_cols

    fig.set_size_inches(width, height)
    fig.tight_layout()
    # now work out what percentage of the total is the title/attributes
    title_percent = int(title)*(title_height/height)
    bottom_margin = attributes_height/height
    left_margin = attributes_height/width
    fig.subplots_adjust(wspace=0, hspace=0, top=1.-title_percent,
                        bottom=bottom_margin, left=left_margin, right=1.)
    return fig, ax_arr


def plot_scores(eventWise_paths, save_prefix=None):
    all_cols, variable_cols, score_cols, table = filter_table(*tabulate_scores(eventWise_paths))
    variable_cols = ['jet_class'] + variable_cols
    # if there are oo many variabel run them in smaller batches
    max_cols = 6
    num_groups = int(np.ceil(len(variable_cols)/max_cols))
    # it's possible that the number of parts actually splits into smaller groups than
    # the max size
    group_length = int(np.ceil(len(variable_cols)/num_groups))
    variable_groups = [variable_cols[i*group_length:(i+1)*group_length]
                       for i in range(num_groups)]
    kinematic_scores = [name for name in score_cols if "Distance" in name]
    ratio_scores = [name for name in score_cols if "Ratio" in name] + ["AvePercentFound"]
    quality_scores = [name for name in score_cols if "Quality" in name]
    score_types = {"Kinematic scores" : kinematic_scores, 
                   "Mass Ratio scores" : ratio_scores,
                   "Peak Quality scores" : quality_scores}
    for title in score_types:
        for i, group in enumerate(variable_groups):
            fig, ax_arr = plot_grid(all_cols, group, score_types[title], table)
            fig.suptitle(title)
            if save_prefix is None or len(save_prefix) == 0:
                plt.show()
                input()
            else:
                save_name = save_prefix + '_' + title.replace(' ', '') + str(i) + ".png"
                plt.savefig(save_name)
            plt.close(fig)


def score_corrilation(eventWise_paths, save_prefix=None):
    all_cols, variable_cols, score_cols, table = filter_table(*tabulate_scores(eventWise_paths))
    fig, ax_arr = plot_grid(all_cols, score_cols, score_cols, table)
    title = "Score corrilations"
    fig.suptitle(title)
    if save_prefix is None or len(save_prefix) == 0:
        plt.show()
        input()
    else:
        save_name = save_prefix + '_' + title.replace(' ', '') + ".png"
        plt.savefig(save_name)
    plt.close(fig)


def input_corrilation(eventWise_paths, save_prefix=None):
    all_cols, variable_cols, score_cols, table = filter_table(*tabulate_scores(eventWise_paths))
    fig, ax_arr = plot_grid(all_cols, variable_cols, variable_cols, table)
    title = "Input corrilations"
    fig.suptitle(title)
    if save_prefix is None or len(save_prefix) == 0:
        plt.show()
        input()
    else:
        save_name = save_prefix + '_' + title.replace(' ', '') + ".png"
        plt.savefig(save_name)
    plt.close(fig)


def make_scale(content):
    for val in content:
        # if it's an array with 2 elements and the first is a string
        # this it's probably a cutoff type, convert to tuple
        if hasattr(val, '__iter__') and len(val) == 2 and isinstance(val[0], str):
            val = tuple(val)
            likely_tuples = [hasattr(x, '__iter__') and len(x) == 2 for x in content]
            # make the tuples into tuples
            content = [tuple(x) if (hasattr(x, '__iter__') and len(x) == 2) else x
                       for x in content]
        # check this first, becuase non floats make np.isnan throw an error
        if isinstance(val, (tuple, str, bool)):
            return make_ordinal_scale(content)
        if val is None or np.isnan(val):
            continue  # then look for another value
        # if we get past the continue statement then it's a float like thing
        return make_float_scale(content)
    if len(content) == 0:
        return [], [], []
    return make_ordinal_scale(content)


def make_float_scale(content, num_increments=11):
    """
    

    Parameters
    ----------
    col_content :
        

    Returns
    -------

    """
    # converting to a list is better for some checks
    positions = np.copy(content.tolist())  # and the llist will be needed later
    has_none = None in content or np.any(np.isnan(positions))
    has_inf = np.any(np.isinf(positions))
    numbers = set(content[np.isfinite(content)]) - {None, np.nan, np.inf, - np.inf}
    # now work out how the scale should work
    if len(numbers) < 2:  # there may be no numbers present, or just one
        numeric_positions = list(numbers)
        if not numeric_positions:
            numeric_positions = [0]
        start = stop = numeric_positions[0]
        step = 1  # we need a non zero step for the next bits
    else:
        start, stop = min(numbers), max(numbers)
        step = (stop - start)/num_increments
        if stop - start > 2: # if it's large enough make the ticks be integer values
            step = int(np.ceil(step))
            numeric_positions = np.arange(start, stop+step, step, dtype=int)
            stop = np.max(numeric_positions)  # the stop should always be the last numertic point
        else:
            numeric_positions = np.arange(start, stop+step, step)
    # as positions for specle values
    scale_positions = np.copy(numeric_positions).tolist()
    if has_inf:
        scale_positions = [scale_positions[0] - step] + scale_positions + [scale_positions[-1] + step]
    if has_none:
        scale_positions = [scale_positions[0] - step] + scale_positions
    scale_labels = int(has_none)*["NaN"] + int(has_inf)*["$-\\inf$"] + \
                   [f"{x:.3g}" for x in numeric_positions] + \
                   int(has_inf)*["$+\\inf$"]
    # now make the positions finite for the special values
    positions[np.logical_or(positions==None, np.isnan(positions))] = start - (has_inf+1)*step
    positions[np.logical_and(positions<0, np.isinf(positions))] = start - step
    positions[np.isinf(positions)] = stop + step
    return positions, scale_positions, scale_labels


def make_ordinal_scale(col_content, max_entries=14):
    """
    

    Parameters
    ----------
    col_content :
        

    Returns
    -------

    """
    # occasionaly equality comparison on the col_content is not possible
    # make a translation to names
    content_names = []
    # if the points need thinning divide into sets
    for con in col_content:
        if isinstance(con, tuple):
            name = ', '.join((con[0], f"{con[1]:.3g}"))
        else:
            name = str(con)
        content_names.append(name)
    scale_labels = sorted(set(content_names))
    scale_positions = np.arange(len(scale_labels))
    positions = np.fromiter((scale_labels.index(name) for name in content_names),
                            dtype=int)
    # now this set may be too long
    if len(scale_positions) > max_entries:  # thin it out
        scale_catigories = []
        for label in scale_labels:
            example = col_content[content_names.index(label)]
            if isinstance(example, tuple):
                # if there are tuples, consider the type to be the first entry
                scale_catigories.append(example[0])
            else:
                scale_catigories.append(type(example))
        # set a maximum numbr of entries per data type
        max_per_catigory = max_entries/len(set(scale_catigories))
        # we will never remove the first of the last entry
        keep = np.zeros_like(scale_positions, dtype=bool)
        for catigory in set(scale_catigories):
            mask = np.fromiter((x == catigory for x in scale_catigories), dtype=bool)
            catigory_keep = np.where(mask)[0]
            if len(catigory_keep) > 1:  # dont throw anything out if the catigory is 1 long
                last_item = catigory_keep[-1]
                keep_one_in = int(np.ceil(np.sum(mask)/max_entries))
                catigory_keep = catigory_keep[:-1:keep_one_in]
                if catigory_keep[-1] != last_item:
                    catigory_keep = np.append(catigory_keep, last_item)
            keep[catigory_keep] = True
        # now reduce the list to the list of things in keep
        scale_labels = np.array(scale_labels)[keep]
        scale_positions = scale_positions[keep]
    return positions, scale_positions, scale_labels


if __name__ == '__main__':
    #for i in range(1, 10):
    #    ew_name = f"megaIgnore/MC{i}.awkd"
    #    print(ew_name)
    #    records.score(ew_name.strip())
    ew_paths = []
    path = True
    while path:
        path = InputTools.get_file_name(f"EventWise file {len(ew_paths)+1}? (empty to complete list) ").strip()
        if path.endswith(".awkd"):
            ew_paths.append(path)
        elif os.path.isdir(path):
            inside = [os.path.join(path, name) for name in os.listdir(path)
                      if name.endswith("awkd") and name.startswith("iridis")]
            ew_paths += inside
            #inside = [os.path.join(path, name) for name in os.listdir(path)
            #          if name.endswith("awkd")]
            #for path in inside:
            #    if InputTools.yesNo_question(f"Add {path}? "):
            #        ew_paths.append(path)
            if not inside:
                print(f"No awkd found in {path}")
        elif path:
            print(f"Error {path} not awkd or folder. ")
    
    if InputTools.yesNo_question("Score an eventWise? "):
        duration = InputTools.get_time("How long to run for? (negative for inf) ")
        if duration < 0:
            duration = np.inf
        if len(ew_paths) == 1:
            append_scores(ew_paths[0], duration=duration)
        else:
            end_time = time.time() + duration
            multiprocess_append_scores(ew_paths, end_time=end_time)
    elif InputTools.yesNo_question("Plot something? "):
        save_prefix = InputTools.get_dir_name("Save prefix? (empty to display) ").strip()
        if not save_prefix:
            save_prefix = None
        again = True
        while again:
            if InputTools.yesNo_question("Plot scores? "):
                plot_scores(ew_paths, save_prefix)
            elif InputTools.yesNo_question("Plot score corrilations? "):
                score_corrilation(ew_paths, save_prefix)
            elif InputTools.yesNo_question("Plot input corrilations? "):
                input_corrilation(ew_paths, save_prefix)
            elif InputTools.yesNo_question("Plot class bests? "):
                plot_class_bests(ew_paths, save_prefix)
                plt.show()
                input()
            again = InputTools.yesNo_question("Plot something else? ")


