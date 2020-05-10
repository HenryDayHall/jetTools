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
from tree_tagger import Components, TrueTag, InputTools, FormJets, Constants, RescaleJets, FormShower
import sklearn.metrics
import sklearn.preprocessing
from matplotlib import pyplot as plt
import numpy as np
import scipy.stats
import bokeh, bokeh.palettes, bokeh.models, bokeh.plotting, bokeh.transform
import socket

def seek_clusters(records, jet_ids, dir_name="megaIgnore"):
    """
    Find a list of clusters in the eventWise files in a directory

    Parameters
    ----------
    records : Records
        the records object 
    dir_name :
        Default value = "megaIgnore")
    jet_ids :
        

    Returns
    -------

    """
    array = records.typed_array()
    # get a list of eventWise files
    file_names = [os.path.join(dir_name, name) for name in os.listdir(dir_name)
            if name.endswith(".awkd")]
    eventWises = []
    for name in file_names:
        try:
            ew = Components.EventWise.from_file(name)
            eventWises.append(ew)
        except KeyError:
            pass  # it's not actually an eventwise
        except Exception:
            st()
    # now looking for all occurances of the numbers
    matches = []
    for jid in jet_ids:
        found = []
        for ew in eventWises:
            jet_names = list(set([name.split('_', 1)[0] for name in ew.columns
                                  if "Jet" in name and name.split('_', 1)[0][-1].isdigit()]))
            jet_numbers = [int(''.join(filter(str.isdigit, name))) for name in jet_names]
            if jid not in jet_numbers:
                continue
            matching_name = jet_names[jet_numbers.index(jid)]
            if records.check_eventWise_match(ew, jid, matching_name):
                found.append((ew, matching_name))
            else:
                check =  records.check_eventWise_match(ew, jid, matching_name)
                jet_params = FormJets.get_jet_params(ew, matching_name, True)
                row = array[array[:, 0] == jid][0]
                print(f" Missmatch of matching name")
                print(f"recorded; {row}")
                print(f"found; {jet_params}")
        if len(found) == 0:
            print(f"Cannot find jet with id num {jid}")
        elif len(found) == 1:
            matches += found
        else:
            #raise RuntimeError(f"Found more than one match for {jid}")
            print(f"Found more than one match for {jid}")
            matches.append(found[0])
    return matches


def seek_best(records, number_required=3, dir_name="megaIgnore"):
    """
    

    Parameters
    ----------
    records :
        param number_required:  (Default value = 3)
    dir_name :
        Default value = "megaIgnore")
    number_required :
         (Default value = 3)

    Returns
    -------

    """
    array = records.typed_array()
    jet_classes = set(array[:, records.indices["jet_class"]])
    best_ids = []
    for jclass in jet_classes:
        best_ids += records.best(jet_class=jclass, num_items=number_required, return_cols=[0]).flatten().tolist()
    best = seek_clusters(records, best_ids)
    return best


def seek_shapeable(dir_name="megaIgnore", file_names=None, jet_pt_cut='default', tag_before_pt_cut=True):
    """
    Find a list of clusters in the eventWise files in a directory

    Parameters
    ----------
    dir_name :
        Default value = "megaIgnore")
    file_names :
        Default value = None)
    jet_pt_cut :
        Default value = 'default')
    tag_before_pt_cut :
        Default value = True)

    Returns
    -------

    """
    if jet_pt_cut == 'default':
        jet_pt_cut = Constants.min_jetpt
    if jet_pt_cut is None or tag_before_pt_cut:
        tag_name = '_Tags'
    else:
        tag_name = f'_{int(jet_pt_cut)}Tags'
    array = records.typed_array()
    if file_names is None:
        # get a list of eventWise files
        file_names = [os.path.join(dir_name, name) for name in os.listdir(dir_name)
                if name.endswith(".awkd")]
    eventWises = []
    for name in file_names:
        try:
            ew = Components.EventWise.from_file(name)
            eventWises.append(ew)
        except KeyError:
            pass  # it's not actually an eventwise
        except Exception:
            st()
    # now looking for all occurances of the numbers
    matches = []
    for ew in eventWises:
        print(ew.save_name)
        tag_names = [name for name in ew.columns if name.endswith(tag_name)]
        for name in tag_names:
            tags = getattr(ew, name, None)
            if tags is None:
                continue
            num_tagged = np.fromiter((sum([len(j) > 0 for j in evnt]) for evnt in tags), dtype=int)
            if np.any(num_tagged > 1):
                matches.append((ew, name.split('_', 1)[0]))
    return matches


def reindex_jets(dir_name="megaIgnore"):
    """
    go through the specified directory and renumber any jets that have the same ID

    Parameters
    ----------
    dir_name :
        Default value = "megaIgnore")

    Returns
    -------

    """
    taken_ids = []
    duplicates = dict()
    # get a list of eventWise files
    file_names = [os.path.join(dir_name, name) for name in os.listdir(dir_name)
                  if name.endswith(".awkd")]
    for ew_name in file_names:
        try:
            ew = Components.EventWise.from_file(ew_name)
        except KeyError:
            continue  # it's not actually an eventwise
        jet_names = get_jet_names(ew)
        duplicates[ew_name] = []
        for j_name in jet_names:
            jet_id = int(j_name.split("Jet", 1)[1])
            if jet_id in taken_ids:
                duplicates[ew_name].append(j_name)
            else:
                taken_ids.append(jet_id)
        if not duplicates[ew_name]:
            del duplicates[ew_name]
    gen_id = id_generator(taken_ids)
    # now work through the duplcates, adding them to the taken_ids
    for ew_name, dup_list in duplicates.items():
        print(f"Fixing {ew_name}")
        ew = Components.EventWise.from_file(ew_name)
        for j_name in dup_list:
            jet_id = int(j_name.split("Jet", 1)[1])
            new_id = next(gen_id)
            new_name = j_name.split("Jet", 1)[0] + "Jet" + str(new_id)
            ew.rename_prefix(j_name, new_name)
        ew.write()
            

def id_generator(used_ids):
    """
    

    Parameters
    ----------
    used_ids :
        

    Returns
    -------

    """
    current_id = 1
    while True:
        current_id += 1
        if current_id not in used_ids:
            yield current_id
            used_ids.append(current_id)


def get_jet_names(eventWise):
    """
    

    Parameters
    ----------
    eventWise :
        

    Returns
    -------

    """
    jet_names = {name.split('_', 1)[0]
                 for name in eventWise.columns
                 if "Jet" in name
                 and not name.startswith("JetInputs")}
    return sorted(jet_names)


def parameter_step(records, jet_class, ignore_parameteres=None):
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
    array = records.typed_array()
    # get the jets parameter list 
    names = FormJets.cluster_classes
    jet_parameters = list(names[jet_class].param_list.keys())
    all_parameters = jet_parameters + ['jet_class']
    parameter_idxs = [(name, records.indices[name]) for name in all_parameters]
    all_parameters, parameter_indices = zip(*sorted(parameter_idxs, key=lambda x: x[1]))
    print(f"All parameters; {all_parameters}")
    if ignore_parameteres is None:
        ignore_parameteres = []
    # sets of parameters to vary
    parameter_sets = {name: set(array[:, records.indices[name]]) for name in jet_parameters
                      if name not in ignore_parameteres}
    # for some parameter sets, fix the values
    parameter_sets['DeltaR'] = np.linspace(0.1, 1.5, 15)
    parameter_sets['ExponentMultiplier'] = np.linspace(-1., 1., 21)
    if 'AffinityCutoff' in parameter_sets:
        knn = [('knn', c) for c in np.arange(1, 6)]
        distance = [('distance', c) for c in np.linspace(0., 10., 11)]
        parameter_sets['AffinityCutoff'] = knn + distance + [None]
    # clip the array to the parameters themselves
    to_compare = [i for i, name in enumerate(all_parameters)
                  if name not in ignore_parameteres]
    array = array[:, parameter_indices][:, to_compare]
    # also only consider scored rows (the rest may not really exist)
    array = array[records.scored]
    steps_taken = 1
    while steps_taken < 20:
        current_best = records.best(jet_class=jet_class, num_items=steps_taken, return_cols=parameter_indices)[0]
        print(f"Searching round {current_best}")
        new_step = np.copy(current_best)
        # go through all the parameters looking for a substitution that hasn't been tried
        names = list(parameter_sets.keys())
        np.random.shuffle(names)
        for name in names:
            avalible = list(parameter_sets[name])
            np.random.shuffle(avalible)
            idx = all_parameters.index(name)
            for value in avalible:
                new_step[idx] = value
                matching = np.all(new_step[to_compare] == array,
                                  axis=1)
                if not np.any(matching):
                    parameters = {name: new_step[all_parameters.index(name)] for name in jet_parameters}
                    # check it's not an illegal combination
                    if parameters_valid(parameters, jet_class):
                        return parameters
            # reset this parameter
            new_step[idx] = current_best[idx]
        # if we get here, no luck in the first round
        # try the next best entry
        steps_taken += 1
    print("Cannot find free space round the 20 best points")


def parameters_valid(parameters, jet_class):
    """
    

    Parameters
    ----------
    parameters :
        param jet_class:
    jet_class :
        

    Returns
    -------

    """
    if parameters['AffinityType']  == 'linear' and parameters['Laplacien'] == 'symmetric':
        return False
    return True


def rand_score(eventWise, jet_name1, jet_name2):
    """
    

    Parameters
    ----------
    eventWise :
        param jet_name1:
    jet_name2 :
        
    jet_name1 :
        

    Returns
    -------

    """
    # two jets clustered from the same eventWise should have
    # the same JetInput_SourceIdx, 
    # if I change this in the future I need to update this function
    selection1 = getattr(eventWise, jet_name1+"_InputIdx")
    selection2 = getattr(eventWise, jet_name2+"_InputIdx")
    num_common_events = min(len(selection1), len(selection2))
    num_inputs_per_event = Components.apply_array_func(len, eventWise.JetInputs_SourceIdx[:num_common_events])
    scores = []
    for event_n, n_inputs in enumerate(num_inputs_per_event):
        labels1 = -np.ones(n_inputs)
        for i, jet in enumerate(selection1[event_n]):
            labels1[jet[jet < n_inputs]] = i
        assert -1 not in labels1
        labels2 = -np.ones(n_inputs)
        for i, jet in enumerate(selection2[event_n]):
            labels2[jet[jet < n_inputs]] = i
        assert -1 not in labels2
        score = sklearn.metrics.adjusted_rand_score(labels1, labels2)
        scores.append(score)
    return scores


def visulise_scores(scores, jet_name1, jet_name2, score_name="Rand score"):
    """
    

    Parameters
    ----------
    scores :
        param jet_name1:
    jet_name2 :
        param score_name: (Default value = "Rand score")
    jet_name1 :
        
    score_name :
         (Default value = "Rand score")

    Returns
    -------

    """
    plt.hist(scores, bins=40, density=True, histtype='stepfilled')
    mean = np.mean(scores)
    std = np.std(scores)
    plt.vlines([mean - 2*std, mean, mean + 2*std], 0, 1, colors=['orange', 'red', 'orange'])
    plt.xlabel(score_name)
    plt.ylabel("Density")
    plt.title(f"Similarity of {jet_name1} and {jet_name2} (for {len(scores)} events)")


def pseudovariable_differences(eventWise, jet_name1, jet_name2, var_name="Rapidity"):
    """
    

    Parameters
    ----------
    eventWise :
        param jet_name1:
    jet_name2 :
        param var_name: (Default value = "Rapidity")
    jet_name1 :
        
    var_name :
         (Default value = "Rapidity")

    Returns
    -------

    """
    eventWise.selected_index = None
    selection1 = getattr(eventWise, jet_name1+"_InputIdx")
    selection2 = getattr(eventWise, jet_name2+"_InputIdx")
    var1 = getattr(eventWise, jet_name1 + "_" + var_name)
    var2 = getattr(eventWise, jet_name2 + "_" + var_name)
    num_common_events = min(len(selection1), len(selection2))
    num_inputs_per_event = Components.apply_array_func(len, eventWise.JetInputs_SourceIdx[:num_common_events])
    pseudojet_vars1 = []
    pseudojet_vars2 = []
    num_unconnected = 0
    for event_n, n_inputs in enumerate(num_inputs_per_event):
        values1 = {}
        for i, jet in enumerate(selection1[event_n]):
            children = tuple(sorted(jet[jet < n_inputs]))
            values = sorted(var1[event_n, i, jet >= n_inputs])
            values1[children] = values
        values2 = {}
        for i, jet in enumerate(selection2[event_n]):
            children = tuple(sorted(jet[jet < n_inputs]))
            values = sorted(var2[event_n, i, jet >= n_inputs])
            values2[children] = values
        for key1 in values1.keys():
            if key1 in values2:
                pseudojet_vars1+=values2[key1]
                pseudojet_vars2+=values2[key1]
            else:
                num_unconnected += 1
    pseudojet_vars1 = np.array(pseudojet_vars1)
    pseudojet_vars2 = np.array(pseudojet_vars2)
    return pseudojet_vars1, pseudojet_vars2, num_unconnected


def fit_to_tags(eventWise, jet_name, tags_in_jets, event_n=None, use_quarks=True):
    """
    

    Parameters
    ----------
    eventWise :
        param jet_name:
    event_n :
        Default value = None)
    tags_in_jets :
        
    jet_name :
        

    Returns
    -------

    """
    if event_n is None:
        assert eventWise.selected_index is not None
    else:
        eventWise.selected_index = event_n
    tag_idx = list(eventWise.TagIndex)
    if use_quarks:
        quark_idx = TrueTag.tags_to_quarks(eventWise, tag_idx)
        tags_in_jets = awkward.fromiter([[quark_idx[tag_idx.index(tag)] for tag in jet]
                                         for jet in tags_in_jets])
        tag_idx = quark_idx
    inputidx_name = jet_name + "_InputIdx"
    rootinputidx_name = jet_name+"_RootInputIdx"
    jet_e = eventWise.match_indices(jet_name+"_Energy", inputidx_name, rootinputidx_name).flatten()
    coords = {name: i for i, name in enumerate(Constants.coordinate_order)}
    n_columns = len(coords)
    if len(tag_idx) == 0 or len(jet_e) == 0:
        empty = np.array([]).reshape((-1, n_columns))
        no_tags = np.full(len(tag_idx), False)
        return no_tags, empty, empty
    jet_px = eventWise.match_indices(jet_name+"_Px", inputidx_name, rootinputidx_name).flatten()
    jet_py = eventWise.match_indices(jet_name+"_Py", inputidx_name, rootinputidx_name).flatten()
    jet_pz = eventWise.match_indices(jet_name+"_Pz", inputidx_name, rootinputidx_name).flatten()
    tag_e = eventWise.Energy[tag_idx]
    tag_px = eventWise.Px[tag_idx]
    tag_py = eventWise.Py[tag_idx]
    tag_pz = eventWise.Pz[tag_idx]
    # calculate what fraction of jet momentum the tag actually receves
    assigned_momentum = np.zeros((len(tag_idx), n_columns), dtype=float)
    found_tags = np.full(len(tag_idx), False)
    for jidx, tags in enumerate(tags_in_jets):
        if len(tags) == 0:
            continue
        mask = np.fromiter((tag_idx.index(t) for t in tags), dtype=int)
        found_tags[mask] = True
        assigned_momentum[mask, coords["Energy"]] = jet_e[jidx] * tag_e[mask] / np.sum(tag_e[mask])
        assigned_momentum[mask, coords["Px"]] = jet_px[jidx] * tag_px[mask] / np.sum(tag_px[mask])
        assigned_momentum[mask, coords["Py"]] = jet_py[jidx] * tag_py[mask] / np.sum(tag_py[mask])
        assigned_momentum[mask, coords["Pz"]] = jet_pz[jidx] * tag_pz[mask] / np.sum(tag_pz[mask])
    # transform this to pt, rapidity, phi
    assigned_momentum[:, coords["Phi"]], assigned_momentum[:, coords["PT"]] = Components.pxpy_to_phipt(assigned_momentum[:, coords["Px"]], assigned_momentum[:, coords["Py"]])
    assigned_momentum[:, coords["Rapidity"]] = Components.ptpze_to_rapidity(assigned_momentum[:, coords["PT"]], assigned_momentum[:, coords["Pz"]], assigned_momentum[:, coords["Energy"]])
    # nans are the result of putting 0 pt into the rapidity calculation
    assigned_momentum[np.isnan(assigned_momentum)] = 0.
    # now get the actual coords
    tag_momentum = np.vstack([getattr(eventWise, name)[tag_idx] for name in Constants.coordinate_order]).transpose()
    return found_tags, tag_momentum, assigned_momentum


def count_b_heritage(eventWise, jet_name, jet_idxs):
    assert eventWise.selected_index is not None
    b_idxs = np.where(np.abs(eventWise.MCPID) == 5)[0]
    b_decendants = FormShower.decendant_idxs(eventWise, *b_idxs)
    input_idxs = eventWise.JetInputs_SourceIdx
    binput_idxs = set(b_decendants.intersection(input_idxs))
    # filter for the InputIdxs that are leaves
    is_leaf = getattr(eventWise, jet_name+"_Child1")[jet_idxs].flatten() == -1
    selected_InputIdxs = getattr(eventWise, jet_name+"_InputIdx")[jet_idxs].flatten()[is_leaf]
    bjetinput_idxs = set(input_idxs[selected_InputIdxs])
    true_positives = len(bjetinput_idxs.intersection(binput_idxs))
    n_positives = len(b_decendants.intersection(input_idxs))
    false_positives = len(bjetinput_idxs - binput_idxs)
    n_negatives = len(set(input_idxs) - binput_idxs)
    try:
        tpr = true_positives / n_positives
    except ZeroDivisionError:
        # probably happens
        tpr = 1.  # no positives avalible...
    try:
        fpr = false_positives / n_negatives
    except ZeroDivisionError:
        # shouldn't happen if there are any tracks
        #if len(input_idxs):
        #    print(f"Warning, no non b-decendent inputs found in event {eventWise.selected_index}"
        #          " this is suspect."
        #          f" All {len(input_idxs)} appear to be b-decndants")
        # looks like it does happen...
        fpr = 0.
    return tpr, fpr


# Add the percentage of b heritage here??
def fit_all_to_tags(eventWise, jet_name, silent=False, jet_pt_cut='default', min_tracks=None, max_angle=None, tag_before_pt_cut=True):
    """
    

    Parameters
    ----------
    eventWise :
        param jet_name:
    silent :
        Default value = False)
    jet_pt_cut :
        Default value = 'default')
    min_tracks :
        Default value = None)
    max_angle :
        Default value = None)
    tag_before_pt_cut :
        Default value = True)
    jet_name :
        

    Returns
    -------

    """
    if jet_pt_cut == 'default':
        jet_pt_cut = Constants.min_jetpt
    if min_tracks is None:
        min_tracks = Constants.min_ntracks
    if max_angle is None:
        max_angle = Constants.max_tagangle
    eventWise.selected_index = None
    if tag_before_pt_cut:
        tag_name = jet_name + '_Tags'
        h_content, content = TrueTag.add_tags(eventWise, jet_name, max_angle, batch_length=np.inf, jet_pt_cut=None, min_tracks=min_tracks, silent=True, append=False)
    else:
        tag_name = f'{jet_name}_{int(jet_pt_cut)}Tags'
        h_content, content = TrueTag.add_tags(eventWise, jet_name, max_angle, batch_length=np.inf, jet_pt_cut=jet_pt_cut, min_tracks=min_tracks, silent=True, append=False)
    tags_in_jets = content[tag_name]
    eventWise.selected_index = None
    n_events = len(getattr(eventWise, jet_name + "_Energy"))
    # name the vaiables to be cut on
    inputidx_name = jet_name + "_InputIdx"
    rootinputidx_name = jet_name+"_RootInputIdx"
    tag_coords = []
    jet_coords = []
    n_jets_formed = []
    tpr, fpr = [], []
    percent_tags_found = []
    for event_n in range(n_events):
        if event_n % 10 == 0 and not silent:
            print(f"{100*event_n/n_events}%", end='\r', flush=True)
        eventWise.selected_index = event_n
        jet_track_pts = getattr(eventWise, jet_name + "_PT")
        # get the valiables ot cut on
        jet_pt = eventWise.match_indices(jet_name+"_PT", inputidx_name, rootinputidx_name).flatten()
        # note this actually counts num pesudojets, but for more than 2 that is sufficient
        num_tracks = Components.apply_array_func(len, jet_track_pts, depth=Components.EventWise.EVENT_DEPTH).flatten()
        jet_idxs = np.where(np.logical_and(jet_pt > jet_pt_cut, num_tracks > min_tracks))[0]
        n_jets = len(jet_idxs)
        fpr_here, tpr_here = count_b_heritage(eventWise, jet_name, jet_idxs)
        fpr.append(fpr_here)
        tpr.append(tpr_here)
        tags_found, tag_c, jet_c = fit_to_tags(eventWise, jet_name, tags_in_jets[event_n])
        percent_tags_found.append(sum(tags_found)/len(tags_found))
        if not np.any(tags_found):
            continue
        # cut any tags that arn't found
        tag_c = tag_c[tags_found]
        jet_c = jet_c[tags_found]
        # if the jet pt cut hapens afterwards do it now
        if jet_pt_cut is not None and not tag_before_pt_cut:
            pt_index = Constants.coordinate_order.index("PT")
            tag_c, jet_c = zip(*[[j, t] for j, t in zip(jet_c, tag_c)
                                 if j[pt_index] > jet_pt_cut and t[pt_index] > jet_pt_cut])
        tag_coords.append(tag_c)
        jet_coords.append(jet_c)
        n_jets_formed.append(n_jets)
    #tag_coords = np.vstack(tag_coords)
    #jet_coords = np.vstack(jet_coords)
    return tag_coords, jet_coords, n_jets_formed, fpr, tpr, percent_tags_found, h_content, content


def score_component_rank(tag_coords, jet_coords):
    """
    

    Parameters
    ----------
    tag_coords :
        param jet_coords:
    jet_coords :
        

    Returns
    -------

    """
    indices = [Constants.coordinate_order.index(name) for name in ["PT", "Rapidity", "Phi"]]
    dims = len(indices)
    scores = np.zeros(dims)
    uncerts = np.zeros(dims)
    for i, j in enumerate(indices):
        scores[i], uncerts[i] = scipy.stats.spearmanr(tag_coords[:, j], jet_coords[:, j])
    return scores, uncerts


def invarientMass2_distance(tag_coords, jet_coords, rescale_poly=None):
    """
    

    Parameters
    ----------
    tag_coords :
        param jet_coords:
    rescale_poly :
        Default value = None)
    jet_coords :
        

    Returns
    -------

    """
    epxpypz_idx = [Constants.coordinate_order.index(name) for name in ["Energy", "Px", "Py", "Pz"]]
    if rescale_poly is not None:
        rescale_factors = RescaleJets.polyval2d(jet_coords[:, Constants.coordinate_order.index("PT")],
                                                jet_coords[:, Constants.coordinate_order.index("Rapidity")],
                                                rescale_poly)
        jet_coords[:, epxpypz_idx] *= np.tile(rescale_factors, (len(epxpypz_idx), 1)).transpose()
    vec_distance2 = (tag_coords[:, epxpypz_idx] - jet_coords[:, epxpypz_idx])**2
    s = vec_distance2[:, 0] - np.sum(vec_distance2[:, 1:], axis=1)
    return np.abs(s)


def distance_to_higgs_mass(jet_coords_by_event):  # no need the jets per event...
    epxpypz_idx = [Constants.coordinate_order.index(name) for name in ["Energy", "Px", "Py", "Pz"]]
    all_jet_momentum = np.array([np.sum(coords[:, epxpypz_idx], axis=0) for coords in jet_coords_by_event])
    jet_mass = np.sqrt(all_jet_momentum[:, 0]**2 - np.sum(all_jet_momentum[:, 1:]**2, axis=1))
    distance_to_higgs = np.abs(jet_mass - 125.)
    return distance_to_higgs


def get_catigories(records, content):
    """
    

    Parameters
    ----------
    records :
        param content:
    content :
        

    Returns
    -------

    """
    catigories = {}
    for name, i in records.indices.items():
        if name in records.evaluation_columns:
            continue
        try:
            possible = list(set(content[:, i]))
        except TypeError:
            possible = []
            for x in content[:, i]:
                if x not in possible:
                    possible.append(x)
        if len(possible) > 1:
            try:
                possible = sorted(possible)
            except TypeError:  #  a conversion to str allows sorting anything really
                possible.sort(key=str)
            catigories[name] = possible
    return catigories


def print_remaining(content, records, columns):
    """
    

    Parameters
    ----------
    content :
        param records:
    columns :
        
    records :
        

    Returns
    -------

    """
    indices = [records.indices[n] for n in columns]
    here = [list(row[indices]) for row in content]
    print(columns)
    print(np.array(here))
    print(f"Num remaining {len(here)}")


def soft_generic_equality(a, b):
    """
    

    Parameters
    ----------
    a :
        param b:
    b :
        

    Returns
    -------

    """
    try:
        # floats, ints and arrays of these should work here
        return np.allclose(a, b)
    except TypeError:
        try:  # strings and bools should work here
            return a == b
        except Exception:
            # for whatever reason this is not possible
            if len(a) == len(b): 
                # by this point it's not a string
                # maybe a tuple of objects, try each one seperatly
                sections = [soft_generic_equality(aa, bb) for aa, bb in zip(a, b)]
                return np.all(sections)
            return False  # diferent lengths


def select_improvements(records, min_jets=0.5,
        metrics=["score(PT)", "score(Rapidity)", "symmetric_diff(Phi)"]):
    """
    

    Parameters
    ----------
    records :
        param min_jets:  (Default value = 0.5)
    metrics :
        Default value = ["score(PT)")
    min_jets :
         (Default value = 0.5)
    "score(Rapidity)" :
        
    "symmetric_diff(Phi)"] :
        

    Returns
    -------

    """
    metric_cols = [records.indices[metric] for metric in metrics]
    invert_factor = np.fromiter((1 - 2*("symmetric_diff" in name) for name in metrics),
                                dtype=float)
    n_jets_col = records.indices['njets']
    array = records.typed_array()
    n_jets = array[:, n_jets_col].astype(float)
    sufficient_jets = n_jets > min_jets
    array = array[np.logical_and(records.scored, sufficient_jets)]
    metric_data = (array[:, metric_cols]*invert_factor).T
    improved_selection = np.full(len(array), False, dtype=bool)
    for metric in metrics:
        best_home = records.best(metric, jet_class="HomeJet", start_mask=sufficient_jets,
                                 return_cols=metric_cols)
        best_home *= invert_factor
        improved = np.logical_and.reduce([col > best for col, best in
                                          zip(metric_data, best_home)])
        improved_selection = np.logical_or(improved_selection, improved)
    return array[improved_selection]


def parameter_comparison(records, c_name="percentfound", cuts=True):
    """
    

    Parameters
    ----------
    records :
    cuts :
        ( Default value = True)
    c_name :
         (Default value = "percentfound")

    Returns
    -------

    """
    array = records.typed_array()[records.scored]
    col_names = ["s_distance", "bdecendant_tpr", "bdecendant_fpr", "distance_to_HiggsMass"]
    # filter anything that scored too close to zero in pt or rapidity
    # these are soem kind of bug
    small = 0.001
    for name in col_names:
        if "score" in name:
            array = array[array[:, records.indices[name]] > small]
    # now we have a scored valid selection
    y_cols = {name: array[:, records.indices[name]].astype(float) for name in col_names
              if name !='TagAngle'}
    sufficient_jets = array[:, records.indices["njets"]].astype(float) > 0.5
    if cuts:
        angles = array[:, records.indices["symmetric_diff(Phi)"]].astype(float)
        good_angle = angles < 1.
        #good_distance = y_cols["s_distance"] < 100000
        mask = np.logical_and(good_angle, sufficient_jets)
        #mask = np.logical_and(good_distance, mask)
        array = array[mask]
        y_cols = {name: y_cols[name][mask] for name in y_cols}
        sufficient_jets = sufficient_jets[mask]
    c_col = array[:, records.indices[c_name]].astype(float)
    # make the data dict
    data_dict = {}
    data_name = {}
    name_data = {}
    max_len = 15
    # some work on the names is needed for for hover to work
    for name, i in records.indices.items():
        if 'uncert' in name or 'std' in name:
            continue
        data_dict[str(i)] = np.array([str(x)[:max_len] for x in array[:, i]])
        data_name[str(i)] = name # somr of the names have probem characters in them
        name_data[name] = str(i) # need to be able to invert this
    use_names = ["jet_class", "DeltaR", "Invarient", "ExponentMultiplier",
                 "TagAngle", "AffinityCutoff", "Laplacien",
                 "NumEigenvectors", "AffinityType", "WithLaplacienScaling"]
    hover = bokeh.models.HoverTool(tooltips=[(data_name[i], "@" + i) for i in data_dict
                                             if data_name[i] in use_names])
    data_dict['colour'] = c_col
    data_dict['alpha'] = [0.6 if suf else 0.2 for suf in sufficient_jets]
    # now we are done altering the data dict, make a copy
    orignal_data_dict = {name: np.copy(data_dict[name]) for name in data_dict}
    # fix the colours
    mapper = bokeh.models.LinearColorMapper(palette=bokeh.palettes.Plasma10,
                                            low=min(c_col), high=max(c_col))
    # fix the markers
    marker_name = 'jet_class'
    marker_key = str(records.indices[marker_name])
    marker_shapes = ['triangle', 'hex', 'circle_x', 'square']
    marker_matches = sorted(set(data_dict[marker_key]))
    markers = bokeh.transform.factor_mark(marker_key, marker_shapes[:len(marker_matches)],
                                          marker_matches)
    plots = []
    sources = {}
    original_positions = {}
    ignore_cols = list(records.evaluation_columns) + ["jet_id", "match_error"]
    parameter_order = [name for name in sorted(records.indices.keys()) if name not in ignore_cols if len(set(array[:, records.indices[name]])) > 1]
    mask_dict = {}
    for name in parameter_order:
        print(name + "~"*5)
        # the data_dict contains strings, we need real values
        parameter_values = array[:, records.indices[name]]
        not_none = [p for p in parameter_values if p is not None]
        catigorical = hasattr(not_none[0], '__iter__')
        if catigorical:
            scale, positions, label_dict, label_catigories = make_ordinal_scale(parameter_values)
            print(f"label_dict = {label_dict}, scale={scale}, set(positions) = {set(positions)}")
            labels = [label_dict[key] for key in scale]  # to fix the order
            boxes = bokeh.models.widgets.CheckboxGroup(labels=labels, active=list(range(len(labels))))
        else:
            scale, positions, label_dict = make_float_scale(parameter_values)
            bottom, top = np.nanmin(not_none), np.nanmax(not_none) 
            has_slider = bottom != top
            if has_slider:
                slider = bokeh.models.RangeSlider(title=name, start=bottom, end=top,
                                                 value=(bottom, top), step=(top-bottom)/20,
                                                 orientation='vertical')
        original_positions[name] = np.copy(positions)
        plots.append([])
        sources[name] = {}
        for col_name, y_col in y_cols.items():
            # first the intresting jets
            data_dict['x'] = positions
            data_dict['y'] = y_col
            source = bokeh.models.ColumnDataSource(data=data_dict)
            sources[name][col_name] = source
            p = bokeh.plotting.figure(tools=[hover, "crosshair", "pan", "reset", "save", "wheel_zoom"], title=name)
            # p.xaxis.axis_label_text_font_size = "30pt" not working
            # p.xaxis.ticker = positions
            p.xaxis.ticker = scale
            p.xaxis.major_label_overrides = label_dict
            p.xaxis.major_label_orientation = "vertical"
            p.yaxis.axis_label = col_name 
            if "symmetric_diff" in col_name or "distance" in col_name:
                p.y_range.flipped = True
            y_lims = chose_ylims(y_col, invert=p.y_range.flipped)
            p.y_range = bokeh.models.Range1d(*y_lims)
            p.scatter('x', 'y', size=10, source=source,
                     fill_color=bokeh.transform.transform('colour', mapper),
                     fill_alpha='alpha', line_alpha='alpha',
                     marker=markers, legend_group=marker_key)
            p.legend.click_policy = "hide"
            p.legend.location = "bottom_left"
            plots[-1].append(p)
        # add a colour bar to the last plot only
        colour_bar = bokeh.models.ColorBar(color_mapper=mapper, location=(0,0),
                                           title=c_name)
        p.add_layout(colour_bar, 'right')
        # now make the selector features
        if catigorical:
            def update(attrname, old, new,
                       name=name, labels=labels, label_catigories=label_catigories):
                keep_catigories = [label_catigories[label] for label in np.array(labels)[new]]
                original_values = array[:, records.indices[name]]
                new_mask = np.fromiter((value in keep_catigories for value in original_values),
                                       dtype=bool)
                mask_dict[name] = new_mask
                mask = np.vstack(tuple(mask_dict.values()))
                mask = np.all(np.vstack(tuple(mask_dict.values())), axis=0)
                new_dict = {name: orignal_data_dict[name][mask] for name in orignal_data_dict}
                for change_name in parameter_order:
                    for col_name, y_col in y_cols.items():
                        new_dict['x'] = original_positions[change_name][mask]
                        new_dict['y'] = y_col[mask]
                        sources[change_name][col_name].data = new_dict
            boxes.on_change('active', update)
            plots[-1].append(boxes)
        elif has_slider:
            def update(attrname, old, new, name=name, top=top, bottom=bottom):
                original_values = array[:, records.indices[name]]
                epsilon = (top-bottom)/50
                new_mask = np.full_like(original_values, True, dtype=bool)
                if new[0] + epsilon > bottom:
                    new_mask = np.logical_and(new_mask, original_values>new[0])
                if new[1] - epsilon < top:
                    new_mask = np.logical_and(new_mask, original_values<new[1])
                mask_dict[name] = new_mask
                mask = np.vstack(tuple(mask_dict.values()))
                mask = np.all(np.vstack(tuple(mask_dict.values())), axis=0)
                new_dict = {name: orignal_data_dict[name][mask] for name in orignal_data_dict}
                for change_name in parameter_order:
                    for col_name, y_col in y_cols.items():
                        new_dict['x'] = original_positions[change_name][mask]
                        new_dict['y'] = y_col[mask]
                        sources[change_name][col_name].data = new_dict
            slider.on_change('value', update)
            plots[-1].append(slider)
    all_p = bokeh.layouts.gridplot(plots, plot_width=400, plot_height=400)
    return all_p


def filter_column_dict(column_dict, filters):
    mask = np.all(np.vstack(filters), axis=1)
    new_dict = {name: column_dict[name][mask] for name in column_dict}
    return new_dict


def make_float_scale(col_content):
    catigories = set(col_content)
    print(f"Unordered catigories {catigories}")
    has_none = None in catigories
    # None prevents propper sorting
    catigories.discard(None)
    catigories = sorted(catigories)
    if has_none:
        catigories += [None]
    print(f"Ordered catigories {catigories}")
    scale = np.array(catigories)
    real = np.fromiter((x for x in catigories if x is not None and np.isfinite(x)),
                       dtype=float)
    if len(real) > 1:
        ave_gap = np.mean(real[1:] - real[:-1])
    else:
        ave_gap = 1.
    scale[scale == -np.inf] = np.min(real) - ave_gap
    scale[scale == np.inf] = np.max(real) + ave_gap
    scale[scale == None] = np.min(real) - 2*ave_gap
    scale = scale.astype(float)
    positions = np.fromiter((scale[catigories.index(x)] for x in col_content),
                            dtype=float)
    # now check if the scale is too tight and if so drop a few values
    if len(scale) > 2:
        length = np.max(scale) - np.min(scale)
        min_gap = length/30
        new_scale = [scale[0]]
        new_catigories = [catigories[0]]
        last_point = scale[0]
        for s, c in zip(scale[1:], catigories[1:]):
            if s - new_scale[-1] > min_gap:
                new_scale.append(s)
                new_catigories.append(c)
        scale = np.array(new_scale)
        catigories = new_catigories
    # trim the labels
    label_dict = {tick: str(cat)[:5] for cat, tick in zip(catigories, scale)}
    return scale, positions, label_dict


def make_ordinal_scale(col_content):
    catigories = set(col_content)
    print(f"Unordered catigories {catigories}")
    has_none = None in catigories
    # None prevents propper sorting
    catigories.discard(None)
    catigories = sorted(catigories)
    if has_none:
        catigories += [None]
    print(f"Ordered catigories {catigories}")
    # it is either a string or a list,
    # the axis spacing will be maufactured
    scale = list(range(len(catigories)))
    positions = np.fromiter((scale[catigories.index(x)] for x in col_content),
                            dtype=float)
    label_catigories = {}
    label_dict = {}
    for pos, cat in zip(scale, catigories):
        if isinstance(cat, tuple):
            s = ', '.join((cat[0], str(cat[1])[:4]))
        else:
            s = str(cat)
        if s in label_catigories:  # don't add it
            exisiting_catigory = label_catigories[s]
            existing_position = scale[catigories.index(existing_cat)]
            positions[positions == pos] = existing_position
        else:
            label_catigories[s] = cat
            label_dict[pos] = s
    return scale, positions, label_dict, label_catigories


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


def cumulative_score(records, typed_array=None):
    """
    

    Parameters
    ----------
    records :
        param typed_array:  (Default value = None)
    typed_array :
         (Default value = None)

    Returns
    -------

    """
    if typed_array is None:
        typed_array = records.typed_array()
    col_names = ["score(PT)", "score(Rapidity)", "symmetric_diff(Phi)"]
    cols = {name: typed_array[:, records.indices[name]].astype(float) for name in col_names}
    cols_widths = {name: np.nanstd(col[col>0.01]) for name, col in cols.items()}
    cumulative = sum((col*(1-2*("symmetric_diff" in name)))/cols_widths[name]
                     for name, col in cols.items())
    return cumulative


def group_discreet(records, only_scored=True):
    """
    group the records by their discreet parameters

    Parameters
    ----------
    records :
        param only_scored:  (Default value = True)
    only_scored :
         (Default value = True)

    Returns
    -------

    """
    array = records.typed_array()
    if only_scored:
        array = array[records.scored]
    pure_numeric = ['DeltaR', 'ExponentMultiplier']
    numeric_tuples = ['AffinityCutoff']
    numeric_tuples = [(name, records.indices[name]) for name in numeric_tuples]
    not_discreet = pure_numeric + numeric_tuples + list(records.evaluation_columns) + ['jet_id']
    discreet_attributes = [(name, records.indices[name]) for name in 
                           sorted(set(records.indices.keys()) - set(not_discreet))]
    groups = {}
    for row in array:
        tuple_start = [f"{name}_None" if row[idx] is None else f"{name}_{row[idx][0]}"
                       for name, idx in numeric_tuples]
        discreet = [f"{name}_{row[idx]}" for name, idx in discreet_attributes]
        key = ','.join(discreet + tuple_start)
        groups.setdefault(key, []).append(row.tolist())
    return groups


def thin_clusters(records, min_mean_jets=0.5, proximity=0.10):
    """
    Identify some clusters to remove in order to save space

    Parameters
    ----------
    records :
        param min_mean_jets:  (Default value = 0.5)
    proximity :
        Default value = 0.10)
    min_mean_jets :
         (Default value = 0.5)

    Returns
    -------

    """
    removed_name = "removed_records.csv"
    removed_records = Records(removed_name)
    # first remove anything that fails a hard cut
    array = records.typed_array()
    mask = array[records.scored, records.indices['njets']] < min_mean_jets
    to_remove = array[records.scored][mask, 0]
    print(f"Removing {len(to_remove)} based on mean jets")
    remove_clusters(records, to_remove)
    print("clusteres removed from awkd files")
    removed_records.transfer(records, to_remove)
    print("cluster records moved to removed_records.csv")
    # now sift through the records looking for things that
    # share all discreet attributes and 
    # are closer than proximity
    pure_numeric_indices = [records.indices[name] for name in ('DeltaR', 'ExponentMultiplier')]
    numeric_tuples_indices = [records.indices['AffinityCutoff']]
    groups = group_discreet(records)
    to_remove = []
    # within each group
    for key in groups:
        group = np.array(groups[key])
        if len(group) < 3:
            continue
        # be sure to preseve the one with the best score
        cumulative = cumulative_score(records, group)
        best_idx = np.argmax(cumulative)
        # now filter on numeric values
        for col in pure_numeric_indices:
            min_step = proximity * (np.max(group[:, col]) - np.min(group[:, col]))
            order = np.argsort(group[:, col])
            best_idx = np.where(order == best_idx)[0][0]
            group = group[order]
            # start by working up from best_idx
            i = best_idx + 1
            while i < len(group):
                if group[i, col] - group[i-1, col] < min_step:
                    to_remove.append(group[i, 0])
                    group = np.delete(group, i, 0)
                i += 1
            # now go down, adjusting the position of the best idx everytime something is deleted
            i = best_idx - 1
            while i >= 0:
                if group[i+1, col] - group[i, col] < min_step:
                    to_remove.append(group[i, 0])
                    group = np.delete(group, i, 0)
                    best_idx -= 1
                i -= 1
        # now filter on numeric tuples
        for col in numeric_tuples_indices:
            values = np.fromiter((np.nan if item is None else item[1] for item in group[:, col]), dtype=float)
            found_none = False
            min_step = proximity * (np.max(values) - np.min(values))
            order = np.argsort(values)
            best_idx = np.where(order == best_idx)[0][0]
            group = group[order]
            values = values[order].tolist()
            # start by working up from best_idx
            i = best_idx + 1
            while i < len(group):
                if np.nan in [values[i], values[i-1]]:
                    if found_none:
                        to_remove.append(group[i, 0])
                        group = np.delete(group, i, 0)
                        del values[i]
                    else:
                        found_none = True
                        continue
                if values[i] - values[i-1] < min_step:
                    to_remove.append(group[i, 0])
                    group = np.delete(group, i, 0)
                    del values[i]
                i += 1
            # now go down, adjusting the position of the best idx everytime something is deleted
            i = best_idx - 1
            while i >= 0:
                if np.nan in [values[i], values[i-1]]:
                    if found_none:
                        to_remove.append(group[i, 0])
                        group = np.delete(group, i, 0)
                        del values[i]
                    else:
                        found_none = True
                        continue
                if values[i+1] - values[i] < min_step:
                    to_remove.append(group[i, 0])
                    group = np.delete(group, i, 0)
                    del values[i]
                    best_idx -= 1
                i -= 1
    print(f"Removing {len(to_remove)} based on proximity")
    remove_clusters(records, to_remove)
    print("clusteres removed from awkd files")
    removed_records.transfer(records, to_remove)
    print("cluster records moved to removed_records.csv")


def remove_clusters(records, jet_ids):
    """
    

    Parameters
    ----------
    records :
        param jet_ids:
    jet_ids :
        

    Returns
    -------

    """
    # probably need to deal with one eventwise at a time
    # and not seek too many jets int he first place
    batch_size = 50
    by_eventWise = {}
    for i in range(0, len(jet_ids), batch_size):
        batch_ids = jet_ids[i: i + batch_size]
        for eventWise, jet_name in seek_clusters(records, batch_ids):
            path = os.path.join(eventWise.dir_name, eventWise.save_name)
            by_eventWise.setdefault(path, []).append(jet_name)
    for path in by_eventWise:
        print(f"Removing {len(by_eventWise[path])} jets from {path}")
        eventWise = Components.EventWise.from_file(path)
        for jet_name in by_eventWise[path]:
            eventWise.remove_prefix(jet_name)
        eventWise.write()


def consolidate_clusters(length=2000, dir_name="megaIgnore", max_size=300):
    """
    Sort clusters by hyperparameter groups and
    put them into new eventWise awkd files

    Parameters
    ----------
    length :
        Default value = 2000)
    dir_name :
        Default value = "megaIgnore")
    max_size :
        Default value = 300)

    Returns
    -------

    """
    # start by learning what's in the directory
    records_name = "tmp{}_records.csv"
    i=0
    while os.path.exists(records_name.format(i)):
        i += 1
    records_name = records_name.format(i)
    records = Records(records_name)
    pottential_names = [os.path.join(dir_name, name) for name in os.listdir(dir_name)
                        if name.endswith(".awkd")]
    ew_names = []
    for ew_name in pottential_names:
        try:
            ew = Components.EventWise.from_file(ew_name)
            ew_names.append(ew_name)
        except Exception:
            print(f"Couldn't read {ew_name}")
            continue
        print(ew_name)
        ew.selected_index = None
        if len(ew.JetInputs_Energy) == length:
            records.scan(ew)
        del ew
    # now all jets in the sample should be located
    groups = group_discreet(records, only_scored=False)
    finalised = []
    priorties = [(name, records.indices[name]) for name in 
                 ["jet_class", "Laplacien", "WithLaplacienScaling", "AffinityType"]]
    for name, col in priorties:
        if not groups:
            break  # everything is finalised
        gathered = {}  # new gathering baskets
        for key, group in groups.items():
            try:
                assignment = str(group[0][col])
            except IndexError:
                group
                col
            if assignment in gathered and len(gathered[assignment]) + len(group) > max_size:
                finalised.append(gathered.pop(assignment))
                gathered[assignment] = group
            elif assignment in gathered:
                gathered[assignment] += group
            else:
                gathered[assignment] = group
        # now if there are still things left in gathered assign them back to groups
        if gathered:
            groups = gathered
        else:
            break
    # if we end the loop with things in groups move them to finalised
    finalised += list(groups.values())
    print(f"Found {len(finalised)} groups")
    # make a directory for the new eventwise objects
    new_dir = os.path.join(dir_name, "consolidated")
    os.mkdir(new_dir)
    consolodated_names = set()
    new_names = []
    # walk over the groups saving them
    for group in finalised:
        group = np.array(group)
        consistant_components = [str(group[0, int(idx)]).capitalize() for name, idx in priorties
                                 if len(set(group[:, int(idx)])) == 1]
        save_name_form = f"Group{{}}{''.join(consistant_components)}_{int(length/1000)}k.awkd"
        save_name = save_name_form.format('')
        i=1
        while save_name in new_names:
            save_name = save_name_form.format(i)
            i += 1
        new_names.append(save_name)
        print(f"Writing group {save_name}, length {len(group)}")
        jet_ids = group[:, records.indices['jet_id']]
        cluster = seek_clusters(records, [jet_ids[0]])
        print(f"Adding non-jet colummns")
        first_eventWise = cluster[0][0]
        non_jet_colunms = [name for name in first_eventWise.columns
                           if "Jet" not in name or name.startswith("JetInputs")]
        non_jet_hcolunms = [name for name in first_eventWise.hyperparameter_columns
                            if "Jet" not in name or name.startswith("JetInputs")]
        non_jet_content = {name: getattr(first_eventWise, name) for name in non_jet_colunms + non_jet_hcolunms}
        new_eventWise = Components.EventWise(new_dir, save_name, contents=non_jet_content,
                                             columns=non_jet_colunms, hyperparameter_columns=non_jet_hcolunms)
        del cluster  # get rid of the ew to save memory
        print("Adding jet content")
        # probably need to deal with one eventwise at a time
        # and not seek too many jets int he first place
        batch_size = 10
        by_eventWise = {}
        for i in range(0, len(jet_ids), batch_size):
            batch_ids = jet_ids[i: i + batch_size]
            for eventWise, jet_name in seek_clusters(records, batch_ids):
                path = os.path.join(eventWise.dir_name, eventWise.save_name)
                by_eventWise.setdefault(path, []).append(jet_name)
        print("Located jet content")
        for path in by_eventWise:
            print(f"Copying {len(by_eventWise[path])} jets from {path}", end='')
            eventWise = Components.EventWise.from_file(path)
            # if there are a lot of jets here break this into chunks
            for i in range(0, len(by_eventWise[path]), batch_size):
                for jet_name in by_eventWise[path][i:i+batch_size]:
                    consolodated_names.add(eventWise.save_name)
                    jet_content = {name: getattr(eventWise, name) for name in eventWise.columns if name.startswith(jet_name)}
                    new_eventWise.append(**jet_content)
                    jet_hcontent = {name: getattr(eventWise, name) for name in eventWise.hyperparameter_columns if name.startswith(jet_name)}
                    new_eventWise.append_hyperparameters(**jet_hcontent)
        print("Added all content, saving...")
        new_eventWise.write()
        print("Saved. removing from ram")
        del new_eventWise
    with open(os.path.join(new_dir, "included.txt"), 'w') as included:
        included.write(', '.join(consolodated_names))
            

class Records:
    """ """
    delimiter = '\t'
    evaluation_columns = ("score(PT)", "symmetric_diff(PT)", 
                          "score(Rapidity)", "symmetric_diff(Rapidity)", 
                          "score(Phi)", "symmetric_diff(Phi)", 
                          "s_distance", 
                          "njets", "percentfound", 
                          "bdecendant_tpr", "bdecendant_fpr",
                          "distance_to_HiggsMass")
    ignore_h_parameters = ['RescaleEnergy']
    def __init__(self, file_path):
        self.file_path = file_path
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r') as existing:
                reader = csv.reader(existing, delimiter=self.delimiter)
                header = next(reader)
                assert header[1] == 'jet_class'
                self.param_names = header[2:]
                self.indices = {name: i+2 for i, name in enumerate(self.param_names)}
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
            self.indices = {}
        after_50k = 'cluster' in socket.gethostname()
        if after_50k:
            relevent_jetids = [j for j in self.jet_ids if j > 50000]
            self.next_uid = np.max(relevent_jetids, initial=0) + 50000
        else:
            relevent_jetids = [j for j in self.jet_ids if j < 50000]
            self.next_uid = np.max(relevent_jetids, initial=0) + 1
        self.indices['jet_id'] = 0
        self.indices['jet_class'] = 1

    def write(self):
        """ """
        with open(self.file_path, 'w') as overwrite:
            writer = csv.writer(overwrite, delimiter=self.delimiter)
            all_rows = [['', 'jet_class'] + self.param_names] + self.content
            writer.writerows(all_rows)

    def typed_array(self):
        """
        Convert the contents to an array of apropreate type,
           fill blanks with default

        Parameters
        ----------

        Returns
        -------

        """
        jet_classes = FormJets.cluster_classes
        typed_content = []
        for row in self.content:
            typed_row = [None for _ in self.indices]
            jet_class = row[self.indices['jet_class']]
            for param_name, i in self.indices.items():
                try:
                    entry = row[i]
                except IndexError:
                    entry = ''
                if entry == '':
                    # set to the default
                    try:
                        typed = jet_classes[jet_class].param_list[param_name]
                    except KeyError:
                        typed = None
                elif entry == 'nan':
                    typed = None
                elif entry == 'inf':
                    typed = np.inf
                else:
                    try:
                        typed = ast.literal_eval(entry)
                    except ValueError:
                        # it's probably a string
                        typed = entry
                typed_row[i] = typed
            typed_content.append(typed_row)
        # got to be an awkward array because numpy hates mixed types
        return np.array(typed_content, dtype=object)

    @property
    def jet_ids(self):
        """ """
        ids = [int(row[0]) for row in self.content]
        return ids

    @property
    def scored(self):
        """ """
        if 'njets' not in self.param_names:
            return np.full(len(self.content), False)
        scored = [row[self.indices["score(PT)"]] not in ('', None, 'nan')
                  for row in self.content]
        return np.array(scored)

    def _add_param(self, *new_params):
        """
        

        Parameters
        ----------
        *new_params :
            

        Returns
        -------

        """
        new_params = [n for n in new_params if n not in self.param_names]
        self.param_names += new_params
        self.indices = {name: i+2 for i, name in enumerate(self.param_names)}
        self.indices['jet_id'] = 0
        self.indices['jet_class'] = 1
        new_blanks = ['' for _ in new_params]
        self.content = [row + new_blanks for row in self.content]

    def append(self, jet_class, param_dict, existing_idx=None, write_now=True):
        """
        gives the new jet a unique ID and returns that value

        Parameters
        ----------
        jet_class :
            param param_dict:
        existing_idx :
            Default value = None)
        write_now :
            Default value = True)
        param_dict :
            

        Returns
        -------

        """
        if existing_idx is None:
            chosen_id = self.next_uid
        else:
            assert existing_idx not in self.jet_ids
            chosen_id = existing_idx
        chosen_id = int(chosen_id)
        new_row = [str(chosen_id), jet_class]
        new_params = list(set(param_dict.keys()) - set(self.param_names))
        self._add_param(*new_params)
        for name in self.param_names:
            new_row.append(str(param_dict.get(name, '')))
        # write to disk
        if write_now:
            with open(self.file_path, 'a') as existing:
                writer = csv.writer(existing, delimiter=self.delimiter)
                writer.writerow(new_row)
        # update content in memeory
        self.content.append(new_row)
        self.next_uid += 1
        self.uid_length = len(str(self.next_uid))
        return chosen_id

    def remove(self, jet_id):
        """
        

        Parameters
        ----------
        jet_id :
            

        Returns
        -------

        """
        idx = self.jet_ids.index(jet_id)
        removed = self.content.pop(idx)
        removed = {name: removed[idx] for name, idx in self.indices.items()}
        return removed

    def transfer(self, donor_records, jet_ids):
        """
        

        Parameters
        ----------
        donor_records :
            param jet_ids:
        jet_ids :
            

        Returns
        -------

        """
        self._add_param(*donor_records.indices.keys())
        column_order = sorted(self.indices, key=self.indices.__getitem__)
        for jid in jet_ids:
            donated = donor_records.remove(jid)
            new_line = [donated[name] for name in column_order]
            self.content.append(new_line)
        self.write()
        donor_records.write()
 
    def check_eventWise_match(self, eventWise, jet_id, jet_name):
        """
        

        Parameters
        ----------
        eventWise :
            param jet_id:
        jet_name :
            
        jet_id :
            

        Returns
        -------

        """
        ignore_params = ['TagAngle']  # this one seems to have issues
        eventWise.selected_index = None
        jet_params = FormJets.get_jet_params(eventWise, jet_name, add_defaults=True)
        for p_name in ignore_params:
            jet_params.pop(p_name, None)
        content = self.typed_array()
        row = content[content[:, 0] == jet_id][0]
        for p_name in jet_params:
            if p_name in self.indices:
                if not soft_generic_equality(row[self.indices[p_name]], jet_params[p_name]):
                    return False
        return True

    def scan(self, eventWise):
        """
        

        Parameters
        ----------
        eventWise :
            

        Returns
        -------

        """
        eventWise.selected_index = None
        jet_names = get_jet_names(eventWise)
        existing = {}  # dicts like  "jet_name": int(row_idx)
        added = {}
        starting_ids = self.jet_ids
        num_events = len(eventWise.JetInputs_Energy)
        content = self.typed_array()
        scored = self.scored
        self._add_param("match_error")
        for name in jet_names:
            try:
                num_start = next(i for i, l in enumerate(name) if l.isdigit())
            except StopIteration:
                print(f"{name} does not have an id number, may not be a jet")
                continue
            jet_params = FormJets.get_jet_params(eventWise, name, add_defaults=True)
            for param_name in self.ignore_h_parameters:
                jet_params.pop(param_name, None)
            jet_class = name[:num_start]
            id_num = int(name[num_start:])
            if id_num in starting_ids:
                idx = starting_ids.index(id_num)
                row = content[idx]
                # verify the hyperparameters
                match = True  # set match here incase jet has no params
                for p_name in jet_params:
                    if p_name not in self.indices:
                        # this parameter wasn't recorded, add it
                        self._add_param(p_name)
                        # the row length will have changed
                        self.content[idx][self.indices[p_name]] = jet_params[p_name]
                        content = self.typed_array()
                        continue  # no sense in checking it
                    value = row[self.indices[p_name]]
                    if value is None or value is '':
                        # it probably didn't get recorded before
                        self.content[idx][self.indices[p_name]] = jet_params[p_name]
                        continue  # no sense in checking now
                    match = soft_generic_equality(value, jet_params[p_name])
                    if not match:
                        break
                if match:
                    # check if it's actually been clustered
                    any_col = next(c for c in eventWise.columns if c.startswith(name))
                    num_found = len(getattr(eventWise, any_col))
                    if num_found == num_events:  # perfect
                        existing[name] = idx
                    elif num_found < num_events:  # incomplete
                        self._add_param("incomplete")
                        self.content[idx][self.indices["incomplete"]] = True
                    elif num_found > num_events:  # wtf
                        raise ValueError(f"Jet {name} has more calculated events than there are jet input events")
                else:
                    # check if it's actually been clustered
                    any_col = next(c for c in eventWise.columns if c.startswith(name))
                    num_found = len(getattr(eventWise, any_col))
                    # check if it's scored and if the jet class matches,
                    if not scored[idx] and row[1] == jet_class and num_found==num_events:
                        # it probably just got recorded wrong, take it over
                        for p_name in self.indices:
                            self.content[idx][self.indices[p_name]] = jet_params.get(p_name, '')
                        match = True
                    # else leave this as a match error
                self.content[idx][self.indices["match_error"]] = not match
            else:  # this ID not found in jets
                self.append(jet_class, jet_params, existing_idx=id_num, write_now=False)
                added[name] = len(self.content) - 1
        self.write()
        return existing, added

    def score(self, target, rescale_for_s_distance=False):
        """
        

        Parameters
        ----------
        target :
            param rescale_for_s_distance:  (Default value = False)
        rescale_for_s_distance :
             (Default value = False)

        Returns
        -------

        """
        if isinstance(target, str):
            if target.endswith('.awkd'):
                target = Components.EventWise.from_file(target)
                self._score_eventWise(target, rescale_for_s_distance)
            elif os.path.isdir(target):
                all_scored = []
                not_scored = []
                for name in os.listdir(target):
                    if not name.endswith('.awkd'):
                        continue
                    path = os.path.join(target, name)
                    try:
                        print(f"Scoring {name}")
                        eventWise = Components.EventWise.from_file(path)
                    except Exception:
                        print(f"Don't recognise {name} as an eventWise")
                        not_scored.append(name)
                        continue  # this one is not an eventwise
                    if hasattr(eventWise, "JetInputs_Energy"):
                        all_scored.append(name)
                        self._score_eventWise(eventWise, rescale_for_s_distance)
                    else:
                        print(f"{name} does not have required components")
                        not_scored.append(name)
                print(f"Scored the files {all_scored}. Did not score {not_scored}")
            else:
                raise NotImplementedError
        else:  # assume its eventwise
            self._score_eventWise(target, rescale_for_s_distance)

    def _score_eventWise(self, eventWise, rescale_for_s_distance=True):
        """
        

        Parameters
        ----------
        eventWise :
            param rescale_for_s_distance:  (Default value = True)
        rescale_for_s_distance :
             (Default value = True)

        Returns
        -------

        """
        print("Scanning eventWise")
        existing, added = self.scan(eventWise)
        all_jets = {**existing, **added}
        num_names = len(all_jets)
        print(f"Found  {num_names}")
        print("Making a continue file, delete it to halt the evauation")
        open("continue", 'w').close()
        jet_ids = self.jet_ids
        scored = {jid: s for jid, s in zip(jet_ids, self.scored)}
        self._add_param(*self.evaluation_columns)
        # as we go we will create tags which can be saved
        # but appending everything at once speeds the process
        content = {}
        h_content = {}
        for i, name in enumerate(all_jets):
            if i % 2 == 0:
                print(f"{100*i/num_names}%", end='\r', flush=True)
                if not os.path.exists('continue'):
                    break
                if i % 30 == 0:
                    self.write()
                    # delete and reread the eventWise, this should free up some ram
                    # because the eventWise is lazy loading
                    path = os.path.join(eventWise.dir_name, eventWise.save_name)
                    del eventWise
                    eventWise = Components.EventWise.from_file(path)
            row = self.content[all_jets[name]]
            if not scored[int(row[0])]:
                coords = {name: i for i, name in enumerate(Constants.coordinate_order)}
                (tag_coords, jet_coords,
                        n_jets_formed,
                        fpr, tpr,
                        percent_tags_found,
                        h_content_here, content_here) = fit_all_to_tags(eventWise, name,
                                                                        silent=True)
                # construct the rescaling factors if required
                RescaleJets.energy_poly(eventWise, name)
                content = {**content, **content_here}
                h_content = {**h_content, **h_content_here}
                # these require no futher processing
                simple_mean = {"njets": n_jets_formed, "percentfound": percent_tags_found,
                               "bdecendant_tpr": tpr, "bdecendant_fpr": fpr}
                for col_name, variable in simple_mean.items():
                    row[self.indices[col_name]] = np.mean(variable)
                # check there are actually matched b jets to process
                if len(tag_coords) == 0:
                    # if not drop a zero in
                    assert np.sum(percent_tags_found) == 0
                    for col_name in self.evaluation_columns:
                        if col_name in simple_mean:
                            pass  # we already delt with it
                        elif 'score' in col_name:
                            row[self.indices[col_name]] = 0.
                        elif (col_name in ['s_distance', 'distance_to_HiggsMass']
                              or 'diff' in col_name):
                            row[self.indices[col_name]] = None
                        else:
                            raise NotImplementedError(f"Don't have process for dealing with {col_name} wen no tags have been found")
                    continue
                # if we reached this point there are jets 
                # before stacking the coords calculate the distance to the higgs mass
                dist_to_hmass = distance_to_higgs_mass(jet_coords)
                # now we no longer care which event each jet came from
                tag_coords = np.vstack(tag_coords)
                jet_coords = np.vstack(jet_coords)
                if len(jet_coords) > 1:
                    scores, uncerts = score_component_rank(tag_coords, jet_coords)
                    syd = 2*np.abs(tag_coords - jet_coords)/(np.abs(tag_coords) + np.abs(jet_coords))
                    if rescale_for_s_distance:
                        rescale_poly = getattr(eventWise, name + "_RescaleEnergy")
                    else:
                        rescale_poly = None
                    s = invarientMass2_distance(tag_coords, jet_coords, rescale_poly)
                else:
                    large_num = 100000
                    scores = np.zeros(len(coords))
                    uncerts = np.zeros(len(coords))
                    syd = np.ones((1, len(coords)))*large_num
                    s = [large_num]
                    n_jets_formed = [0]
                row[self.indices["score(PT)"]] = scores[0]
                row[self.indices["score(Rapidity)"]] = scores[1]
                row[self.indices["score(Phi)"]] = scores[2]
                row[self.indices["s_distance"]] = np.mean(s)
                row[self.indices["distance_to_HiggsMass"]] = np.mean(dist_to_hmass)
                # but the symetric difernce for phi should be angular
                try:
                    row[self.indices["symmetric_diff(PT)"]] = np.mean(syd[:, coords["PT"]])
                    row[self.indices["symmetric_diff(Rapidity)"]] = np.mean(syd[:, coords["Rapidity"]])
                    row[self.indices["symmetric_diff(Phi)"]] = np.mean(syd[:, coords["Phi"]])
                except Exception:
                    st()
                if np.nan in row:
                    st()
                    row
        eventWise.append(**content)
        eventWise.append_hyperparameters(**h_content)
        self.write()

    def best(self, metric='cumulative', jet_class=None, invert=None,
             start_mask=None, return_cols=None, num_items=1):
        """
        

        Parameters
        ----------
        metric :
            Default value = 'cumulative')
        jet_class :
            Default value = None)
        invert :
            Default value = None)
        start_mask :
            Default value = None)
        return_cols :
            Default value = None)
        num_items :
            Default value = 1)

        Returns
        -------

        """
        mask = self.scored
        if start_mask is not None:
            mask = np.logical_and(mask, start_mask)
        content = self.typed_array()[mask]
        if metric == 'cumulative':
            evaluate = cumulative_score(self, content)
        else:
            evaluate = content[:, self.indices[metric]]
        if jet_class is not None:
            jet_mask = content[:, 1] == jet_class
            content = content[jet_mask]
            evaluate = evaluate[jet_mask]
        if invert is None:
            invert = "symmetric_diff" in metric
        sign = 1 - 2*invert
        sorted_idx = np.argsort(sign*evaluate)
        best_in_mask = sorted_idx[-num_items:]
        if return_cols is None:
            return np.array([content[best_in_mask, 0], evaluate[best_in_mask]]).T
        return content[best_in_mask][:, return_cols]


if __name__ == '__main__':
    records_name = InputTools.get_file_name("Records file? (new or existing) ")
    records = Records(records_name)
    ew_name = InputTools.get_file_name("EventWise file? (existing) ")
    records.score(ew_name.strip())
    
    #ew_names = [name for name in os.listdir("megaIgnore") if name.endswith('.awkd')]
    #comparison1(records)

# serve this with
# bokeh serve --show tree_tagger/CompareClusters.py
if 'bk_script' in __name__:
    records_name = InputTools.get_file_name("Records file? (new or existing) ")
    records = Records(records_name)
    plots = parameter_comparison(records, cuts=True)
    bokeh.plotting.curdoc().add_root(plots)

