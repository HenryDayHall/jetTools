""" compare two jet clustering techniques """
import tabulate
import awkward
import ast
import csv
import os
import pickle
import matplotlib
import awkward
from ipdb import set_trace as st
from tree_tagger import Components, TrueTag, InputTools, FormJets, Constants, RescaleJets, FormShower, JetQuality
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



# TODO what is happening with PT/rapidity/phi distance
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
        inheritance = getattr(eventWise, jet_name + "_Inheritance")
        tag_idxs = eventWise.BQuarkIdx
        matched_jets = [[] for _ in event_tags]
        for jet_n, jet_tags in enumerate(getattr(eventWise, jet_name + "_ITags")):
            if jet_n not in jet_idxs[event_n] or len(jet_tags) == 0:
                continue  # jet not sutable or has no tags
            if len(jet_tags) == 1:
                # no chosing to be done, the jet just has one tag
                tag_idx = jet_tags[0]
            else:
                # chose the tag with the greatest inheritance
                jet_root = np.where(parent_idxs[jet_n] == -1)[0][0]
                # dimension 0 of inheritance is which tag particle
                # dimension 1 of inheritance is which jet
                # dimension 2 of inheritance is particles in the jet
                tag_position = np.argmax(inheritance[:, jet_n, jet_root])
                # this is the particle index of the tag with greatest inheritance in the jet
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


def append_scores(eventWise, dijet_mass=None):
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
    if "DetectableTag_Idx" not in eventWise.columns:
        TrueTag.add_detectable_fourvector(eventWise)
    for i, name in enumerate(names):
        # check if it has already been scored
        if name + "_AvePTDistance" in eventWise.hyperparameter_columns:
            continue
        print(f"\n{i/num_names:.1%}\t{name}\n" + " "*10, flush=True)

        # if we reach here the jet still needs a score
        try:
            best_width, best_fraction = JetQuality.quality_width_fracton(eventWise, name,
                                                                         dijet_mass)
        except (ValueError, RuntimeError):  # didn't make enough masses
            best_width = best_fraction = np.nan
        new_hyperparameters[name + "_QualityWidth"] = best_width
        new_hyperparameters[name + "_QualityFraction"] = best_fraction
        # now the mc truth based scores
        TrueTag.add_inheritance(eventWise, name, batch_length=np.inf)
        jet_idxs = filter_jets(eventWise, name)
        new_content = get_detectable_comparisons(eventWise, name, jet_idxs, False)
        new_averages = {}
        # we are only intrested in finite results
        for key, values in new_content.items():
            flattened = values.flatten()
            finite = np.isfinite(flattened)
            if np.any(finite):
                value = np.mean(flattened[finite])
            else:  # sometimes there could be no finite results at all
                value = np.nan
            new_averages[key.replace('_', '_Ave')] = value
        new_contents.update(new_content)
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
    eventWise.append_hyperparameters(**new_hyperparameters)
    eventWise.append(**new_contents)

# plotting code

def tabulate_scores(eventWise_paths, variable_cols=None, score_cols=None):
    if score_cols is None:
        score_cols = ["QualityWidth", "QualityFraction", "AveSignalMassRatio", "AveBGMassRatio",
                      "AvePTDistance", "AvePhiDistance", "AveRapidityDistance"]
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


def plot_scores(all_cols, variable_cols, score_cols, table, plot_path="./images/scores"):
    inverted_names = ["QualityWidth", "QualityFraction", "AvePTDistance", "AvePhiDistance",
                      "AveRapidityDistance", "AveBGMassRatio"]
    invert = [name in inverted_names for name in score_cols]
    fig, ax_arr = plt.subplots(len(score_cols), len(variable_cols), sharex=True, sharey=True)
    # give each of the clusters a random colour and marker shape
    colours = np.random.rand((len(table), 4))
    markers = np.random.choice(['v', 's', '*', 'D', 'P', 'X'], len(table))
    plotting_params = dict(c=colours, marker=markers)
    for col_n, variable_name in enumerate(variable_cols):
        values = table[all_cols.index(variable_name)]
        # this function will decided what kind of scale and create it
        x_positions, scale_positions, scale_labels = make_scale(values)
        for row_n, score_name in enumerate(score_cols):
            scores = table[all_cols.index(score_name)]
            ax = ax_arr[row_n, col_n]
            ax.set_xlabel(variable_name)
            ax.set_ylabel(score_name)
            ax.scatter(x_positions, scores, **plotting_params)
            ax.set_xticks(scale_positions)
            ax.set_xticklabels(scale_labels)
    return fig, ax_arr


def make_scale(content):
    for val in content:
        if val in [None, np.nan]:
            continue
        if isinstance(val, (tuple, str, bool)):
            return make_ordinal_scale(content)
        else:
            return make_float_scale(content)
    return make_ordinal_scale(content)


def make_float_scale(content):
    """
    

    Parameters
    ----------
    col_content :
        

    Returns
    -------

    """
    has_none = None in content or np.nan in content
    has_inf = np.inf in np.abs(content)
    numbers = set(content)
    content -= {None, np.nan, np.inf, -np.inf}
    min_val, max_val = min(numbers), max(numbers)
    # calculate a good distance to but the None and inf values
    gap_length = 0.1*(max_val-min_val)
    # now make a copy of the content for positions
    positions = np.copy(content)
    positions[positions==-np.inf] = min_val - gap_length
    positions[positions==np.inf] = max_val + gap_length
    positions[np.logical_and(positions==None, positions==np.nan)] = max_val + (has_inf+1)*gap_length
    # now work out how the scale should work
    scale_positions = np.linspace(min(positions), max(positions), 11+has_none+2*has_inf)
    scale_labels = has_none*["NaN"] + has_inf*["$-\\inf$"] + \
                   [f"{x:.3g}" for x in np.linspace(min_val, max_val, 11)] + \
                   has_inf*["$+\\inf$"]
    return positions, scale_positions, scale_labels


def make_ordinal_scale(col_content):
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
    return positions, scale_positions, scale_labels


def chose_ylims(y_data, invert=False):
    """
    

    Parameters
    ----------
    y_data :
        param invert:  (Default value = False)
    invert :
        (Default value = False)

    Returns
    -------

    
    """
    std = np.std(y_data)
    if invert:
        top = np.min(y_data) - 0.3*std
        bottom = np.min((np.max(y_data), np.mean(y_data) + 1.*std))
    else:
        top = np.max(y_data) + 0.3*std
        bottom = np.max((np.min(y_data), np.mean(y_data) - 1.*std))
    return bottom, top


if __name__ == '__main__':
    #for i in range(1, 10):
    #    ew_name = f"megaIgnore/MC{i}.awkd"
    #    print(ew_name)
    #    records.score(ew_name.strip())
    
    ew_name = InputTools.get_file_name("EventWise file? (existing) ")
    append_scores(ew_name.strip())

