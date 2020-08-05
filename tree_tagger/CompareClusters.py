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
    best_name = scored_names[np.nanargmax(score)]
    return best_name


# code for making scores

def add_bg_mass(eventWise):
    n_events = len(eventWise.BQuarkIdx)
    all_bg_mass = np.zeros(n_events)
    for event_n in range(n_events):
        eventWise.selected_index = event_n
        source_idx = eventWise.JetInputs_SourceIdx
        detectable_idx = eventWise.DetectableTag_Idx
        # rember that each tag gets it's own list of detectables
        bg = list(set(source_idx)  - set(detectable_idx.flatten()))
        all_bg_mass[event_n] = np.sum(eventWise.Energy[bg])**2 - np.sum(eventWise.Px[bg])**2 -\
                               np.sum(eventWise.Py[bg])**2 - np.sum(eventWise.Pz[bg])**2
    eventWise.append(DetectableBG_Mass = awkward.fromiter(all_bg_mass))


def get_mass_ratios(eventWise, jet_name, jet_idxs, append=False):
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
    if "DetectableTag_Mass" not in eventWise.columns:
        Components.add_mass(eventWise, "DetectableTag")
    if "DetectableBG_Mass" not in eventWise.columns:
        add_bg_mass(eventWise)
    eventWise.selected_index = None
    tags = eventWise.BQuarkIdx
    n_events = len(tags)
    # we expect 4 tags per event, but only assert this, don't build in the assumption otherwise
    n_tags = len(tags[0])
    assert n_tags == 4
    too_many_tags = set()  # keep track of which events have more than 4 tags
    # we assume the tagger behaves perfectly
    tag_mass_in = []
    bg_mass_in = []
    percent_found = np.zeros(n_events)
    for event_n, event_tags in enumerate(tags):
        eventWise.selected_index = event_n
        energy = eventWise.Energy
        px = eventWise.Px
        py = eventWise.Py
        pz = eventWise.Pz
        source_idx = eventWise.JetInputs_SourceIdx
        # we need to go though the
        tag_fragment = 1/len(event_tags)
        tag_mass_in.append([0 for _ in event_tags])
        bg_mass_in.append([0 for _ in event_tags])
        for jet_n, jet_tags in enumerate(getattr(eventWise, jet_name + "_ITags")):
            if jet_n not in jet_idxs[event_n]:
                continue  # jet not sutable
            for tag in jet_tags:
                # this one was found
                percent_found[event_n] += tag_fragment
                # the tag value is a particle idx
                tag_position = np.where(event_tags == tag)[0][0]
                if tag_position >= n_tags:
                    too_many_tags.add(event_n)
                # tag position should be 0 - 3 for the 4 b tags
                jet_inputs = getattr(eventWise, jet_name + "_InputIdx")[jet_n]
                # convert to source_idxs
                jet_inputs = jet_inputs[jet_inputs < len(source_idx)]
                jet_inputs = set(source_idx[jet_inputs])
                tag_in_jet = jet_inputs.intersection(eventWise.DetectableTag_Idx[tag_position])
                bg_in_jet = list(jet_inputs - tag_in_jet)
                tag_in_jet = list(tag_in_jet)
                tag_mass_in[event_n][tag_position] = np.sum(energy[tag_in_jet])**2 -\
                                                     np.sum(px[tag_in_jet])**2 -\
                                                     np.sum(py[tag_in_jet])**2 -\
                                                     np.sum(pz[tag_in_jet])**2
                bg_mass_in[event_n][tag_position] = np.sum(energy[bg_in_jet])**2 -\
                                                    np.sum(px[bg_in_jet])**2 -\
                                                    np.sum(py[bg_in_jet])**2 -\
                                                    np.sum(pz[bg_in_jet])**2
    eventWise.selected_index = None
    content = {}
    content[jet_name + "_SignalMassRatio"] = awkward.fromiter(tag_mass_in)/eventWise.DetectableTag_Mass
    content[jet_name + "_BGMassRatio"] = awkward.fromiter(bg_mass_in)/eventWise.DetectableBG_Mass
    content[jet_name + "_PercentFound"] = awkward.fromiter(percent_found) 
    print(f"More than 4 b quarks in {too_many_tags}. This is {len(too_many_tags)*100/n_events}% or events")
    if append:
        eventWise.append(**content)
    return content


def get_kinematic_distances(eventWise, jet_name, jet_idxs, append=False):
    if "DetectableTag_PT" not in eventWise.columns:
        Components.add_phi(eventWise, "DetectableTag")
        Components.add_PT(eventWise, "DetectableTag")
        Components.add_rapidity(eventWise, "DetectableTag")
    tags = eventWise.BQuarkIdx
    # we assume the tagger behaves perfectly
    rapidity_in_jet = []
    phi_in_jet = []
    pt_in_jet = []
    for event_n, event_tags in enumerate(tags):
        eventWise.selected_index = event_n
        energy = eventWise.Energy
        px = eventWise.Px
        py = eventWise.Py
        pz = eventWise.Pz
        source_idx = eventWise.JetInputs_SourceIdx
        rapidity_in_jet.append([np.nan for _ in event_tags])
        phi_in_jet.append([np.nan for _ in event_tags])
        pt_in_jet.append([np.nan for _ in event_tags])
        # we need to go though the
        for jet_n, jet_tags in enumerate(getattr(eventWise, jet_name + "_ITags")):
            if jet_n not in jet_idxs[event_n]:
                continue  # jet not sutable
            for tag in jet_tags:
                # the tag value is a particle idx
                tag_position = np.where(event_tags == tag)[0][0]
                # tag position should be 0 - 3 for the 4 b tags
                jet_inputs = getattr(eventWise, jet_name + "_InputIdx")[jet_n]
                # convert to source_idxs
                jet_inputs = jet_inputs[jet_inputs < len(source_idx)]
                jet_inputs = set(source_idx[jet_inputs])
                tag_in_jet = list(jet_inputs.intersection(eventWise.DetectableTag_Idx[tag_position]))
                phi, pt = Components.pxpy_to_phipt(np.sum(px[tag_in_jet]), np.sum(py[tag_in_jet]))
                phi_in_jet[event_n][tag_position] = phi
                pt_in_jet[event_n][tag_position] = pt
                rapidity_in_jet[event_n][tag_position] = Components.ptpze_to_rapidity(pt, np.sum(pz[tag_in_jet]),
                                                                                      np.sum(energy[tag_in_jet]))
    eventWise.selected_index = None
    content = {}
    rapidity_distance = awkward.fromiter(rapidity_in_jet) - eventWise.DetectableTag_Rapidity
    pt_distance = awkward.fromiter(pt_in_jet) - eventWise.DetectableTag_PT
    phi_distance = awkward.fromiter(phi_in_jet) - eventWise.DetectableTag_Phi
    content[jet_name + "_RapidiyDistance"] = rapidity_distance
    content[jet_name + "_PTDistance"] = pt_distance
    content[jet_name + "_PhiDistance"] = phi_distance
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
                                        if sum(children == -1) > min_ntracks))
        jet_idxs.append(long_enough)
    return awkward.fromiter(jet_idxs)


def append_scores(eventWise):
    if isinstance(eventWise, str):
        eventWise = Components.EventWise.from_file(eventWise)
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
        print(f"\n{100*i/num_names}%\t{name}\n" + " "*10, flush=True)

        # if we reach here the jet still needs a score
        try:
            best_width, best_fraction = JetQuality.quality_width_fracton(eventWise, name, 40)
        except (ValueError, RuntimeError):  # didn't make enough masses
            best_width = best_fraction = np.nan
        new_hyperparameters[name + "_QualityWidth"] = best_width
        new_hyperparameters[name + "_QualityFraction"] = best_fraction
        # now the mc truth based scores
        TrueTag.add_inheritance(eventWise, name, batch_length=np.inf)
        jet_idxs = filter_jets(eventWise, name)
        mass_content = get_mass_ratios(eventWise, name, jet_idxs, False)
        mass_averages = {key.replace('_', '_Ave'): np.nanmean(values.flatten())
                         for key, values in mass_content.items()}
        new_hyperparameters.update(mass_content)
        new_contents.update(mass_averages)
        kinematic_content = get_kinematic_distances(eventWise, name, jet_idxs, False)
        kinematic_averages = {key.replace('_', '_Ave'): np.nanmean(values.flatten())
                              for key, values in kinematic_content.items()}
        new_contents.update(kinematic_averages)
        new_contents.update(kinematic_content)
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
        variables_cols = sorted(variables_cols)
    # also record jet class, eventWise.svae_name and jet_name
    all_cols = ["jet_name", "jet_class", "eventWise_name"] + variables_cols + score_cols
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

