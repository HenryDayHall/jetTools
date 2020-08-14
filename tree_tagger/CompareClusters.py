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
                      "AveDistancePT", "AveDistancePhi", "AveDistanceRapidity"]
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


def filter_table(all_cols, variable_cols, score_cols, table):
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
    # drop any rows where all score_cols are nan
    score_nan = nan_mask[:, [all_cols.index(name) for name in score_cols]]
    all_nan = np.all(score_nan, axis=1)
    table = table[~all_nan]
    nan_mask = nan_mask[~all_nan]
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
    


def plot_grid(all_cols, variable_cols, score_cols, table):
    inverted_names = ["QualityWidth", "QualityFraction", "AveDistancePT", "AveDistancePhi",
                      "AveDistanceRapidity", "AveBGMassRatio"]
    n_variables = len(variable_cols)
    n_scores = len(score_cols)
    fig, ax_arr = plt.subplots(n_scores, n_variables, sharex='col', sharey='row')
    # give each of the clusters a random colour and marker shape
    colours = np.random.rand(len(table)*4).reshape((len(table), 4))
    #markers = np.random.choice(['v', 's', '*', 'D', 'P', 'X'], len(table))
    plotting_params = dict(c=colours) #, marker=markers)
    for col_n, variable_name in enumerate(variable_cols):
        values = table[:, all_cols.index(variable_name)]
        # this function will decided what kind of scale and create it
        x_positions, scale_positions, scale_labels = make_scale(values)
        for row_n, score_name in enumerate(score_cols):
            scores = table[:, all_cols.index(score_name)].tolist()
            ax = ax_arr[row_n, col_n]
            ax.scatter(x_positions, scores, **plotting_params)
            if score_name in inverted_names:
                ax.invert_yaxis()
            if row_n == n_scores-1:
                ax.set_xticks(scale_positions)
                ax.set_xticklabels(scale_labels, rotation=90)
                ax.set_xlabel(variable_name)
            else:
                ax.set_xticks([], [])
            if col_n == 0:
                ax.set_ylabel(score_name)
    return fig, ax_arr


def plot_scores(eventWise_paths):
    all_cols, variable_cols, score_cols, table = filter_table(*tabulate_scores(eventWise_paths))
    kinematic_scores = [name for name in score_cols if "Distance" in name]
    ratio_scores = [name for name in score_cols if "Ratio" in name]
    quality_scores = [name for name in score_cols if "Quality" in name]
    score_types = [kinematic_scores, ratio_scores, quality_scores]
    for scores in score_types:
        fig, ax_arr = plot_grid(all_cols, variable_cols, scores, table)
        fig.set_size_inches(len(variable_cols)*2, 8)
        fig.tight_layout()
        plt.show()
        input()
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
    try:
        start, stop = min(numbers), max(numbers)
    except ValueError:  # there may be no numbers present
        start = stop = 0
        numeric_positions = [0]
        step = 1
    else:
        step = (stop - start)/num_increments
        if stop - start > 2: # if it's large enough make the ticks be integer values
            step = int(np.ceil(step))
            numeric_positions = np.arange(start, stop, step, dtype=int)
        else:
            numeric_positions = np.arange(start, stop, step)
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
    
    if InputTools.yesNo_question("Score an eventWise? "):
        ew_name = InputTools.get_file_name("EventWise file? (existing) ")
        append_scores(ew_name.strip())
    elif InputTools.yesNo_question("Plot results? "):
        ew_paths = []
        path = True
        while path:
            path = InputTools.get_file_name(f"EventWise file {len(ew_paths)+1}? (empty to complete list) ").strip()
            ew_paths.append(path)
        ew_paths.pop()  # the last one is empty
        plot_scores(ew_paths)


