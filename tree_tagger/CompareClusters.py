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
from tree_tagger import Components, TrueTag, InputTools, FormJets
import sklearn.metrics
import sklearn.preprocessing
from matplotlib import pyplot as plt
import numpy as np
import scipy.stats
import bokeh, bokeh.palettes, bokeh.models, bokeh.plotting, bokeh.transform
import socket

def seek_clusters(records, jet_ids, dir_name="megaIgnore"):
    """ Find a list of clusters in the eventWise files in a directory """
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
    array = records.typed_array()
    jet_classes = set(array[:, records.indices["jet_class"]])
    best_ids = []
    for jclass in jet_classes:
        best_ids += records.best(jet_class=jclass, num_items=number_required, return_cols=[0]).flatten().tolist()
    best = seek_clusters(records, best_ids)
    return best


def rand_score(eventWise, jet_name1, jet_name2):
    """
    

    Parameters
    ----------
    eventWise :
        
    jet_name1 :
        
    jet_name2 :
        

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
        
    jet_name1 :
        
    jet_name2 :
        
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
        
    jet_name1 :
        
    jet_name2 :
        
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


def fit_to_tags(eventWise, jet_name, event_n=None, tag_pids=None, jet_pt_cut=30., min_tracks=2):
    """
    

    Parameters
    ----------
    eventWise :
        
    jet_name :
        
    event_n :
         (Default value = None)
    tag_pids :
         (Default value = None)
    jet_pt_cut :
         (Default value = 30.)
    min_tracks :
         (Default value = 2)

    Returns
    -------

    """
    if event_n is None:
        assert eventWise.selected_index is not None
    else:
        eventWise.selected_index = event_n
    inputidx_name = jet_name + "_InputIdx"
    rootinputidx_name = jet_name+"_RootInputIdx"
    jet_pt = eventWise.match_indices(jet_name+"_PT", inputidx_name, rootinputidx_name).flatten()
    num_tracks = Components.apply_array_func(len, getattr(eventWise, jet_name+"_PT")).flatten()
    if jet_pt_cut is None:
        mask = np.ones_like(jet_pt)
    else:
        mask = jet_pt>jet_pt_cut
    if min_tracks is not None:
        mask = np.logical_and(mask, num_tracks>(min_tracks-0.1))
    if not np.any(mask):
        empty = np.array([]).reshape((-1, 3))
        return empty, empty, 0
    n_jets = np.sum(mask)
    jet_pt = jet_pt[mask]
    jet_e = eventWise.match_indices(jet_name+"_Energy", inputidx_name, rootinputidx_name).flatten()[mask]
    jet_px = eventWise.match_indices(jet_name+"_Px", inputidx_name, rootinputidx_name).flatten()[mask]
    jet_py = eventWise.match_indices(jet_name+"_Py", inputidx_name, rootinputidx_name).flatten()[mask]
    jet_pz = eventWise.match_indices(jet_name+"_Pz", inputidx_name, rootinputidx_name).flatten()[mask]
    tag_idx = TrueTag.tag_particle_indices(eventWise, tag_pids=tag_pids)
    if len(tag_idx) == 0:
        empty = np.array([]).reshape((-1, 3))
        return empty, empty, n_jets
    tag_e = eventWise.Energy[tag_idx]
    tag_px = eventWise.Px[tag_idx]
    tag_py = eventWise.Py[tag_idx]
    tag_pz = eventWise.Pz[tag_idx]
    # starting with the highest CoM2 tag working to the lowest
    # assign tags to jets and recalculate the distance as needed
    matched_jet = np.zeros_like(tag_e, dtype=int) - 1
    jet_4vec = np.vstack((jet_e, jet_px, jet_py, jet_pz)).T
    tag_4vec = np.vstack((tag_e, tag_px, tag_py, tag_pz)).T
    current_4vec_ofset = -np.copy(jet_4vec)
    tag_4vec2 = tag_4vec**2
    s_tag = tag_4vec2[:, 0] - np.sum(tag_4vec2[:, 1:], axis=1)
    for t_idx in np.argsort(s_tag):
        this_4 = tag_4vec[t_idx]
        dist2 = (current_4vec_ofset + this_4)**2
        # not sure this reeealy makes sense # TODO use a standard distance
        allocate_to = np.argmin(dist2[:, 0] - np.sum(dist2[:, 1:], axis=1))
        matched_jet[t_idx] = allocate_to
        current_4vec_ofset[allocate_to] += this_4
    # now calculate what fraction of jet momentum the tag actually receves
    chosen_jets = list(set(matched_jet))
    p_fragment = np.zeros_like(matched_jet, dtype=float)
    for jet_idx in chosen_jets:
        tag_idx_here = np.where(matched_jet == jet_idx)[0]
        p_fragment[tag_idx_here] = s_tag[tag_idx_here]/np.sum(s_tag[tag_idx_here])
    # now get the actual coords
    # tag_idx is wrong var
    tag_coords = np.vstack((eventWise.PT[tag_idx],
                            eventWise.Rapidity[tag_idx],
                            eventWise.Phi[tag_idx])).transpose()
    jet_phi = eventWise.match_indices(jet_name+"_Phi", inputidx_name, rootinputidx_name).flatten()[mask]
    jet_phi = jet_phi[matched_jet]
    jet_4_coords = jet_4vec[matched_jet]*p_fragment.reshape(-1, 1)
    jet_pt = np.sqrt(jet_4_coords[:, 1]**2 + jet_4_coords[:, 2]**2)
    jet_rapidity = Components.ptpze_to_rapidity(jet_pt, jet_4_coords[:, 3], jet_4_coords[:, 0])
    jet_coords = np.vstack((jet_pt, jet_rapidity, jet_phi)).transpose()
    return tag_coords, jet_coords, n_jets
    

def fit_all_to_tags(eventWise, jet_name, silent=False):
    """
    

    Parameters
    ----------
    eventWise :
        
    jet_name :
        
    silent :
         (Default value = False)

    Returns
    -------

    """
    eventWise.selected_index = None
    n_events = len(getattr(eventWise, jet_name + "_Energy"))
    tag_pids = np.genfromtxt('tree_tagger/contains_b_quark.csv', dtype=int)
    tag_coords = []
    jet_coords = []
    n_jets_formed = []
    for event_n in range(n_events):
        if event_n % 10 == 0 and not silent:
            print(f"{100*event_n/n_events}%", end='\r', flush=True)
        eventWise.selected_index = event_n
        n_jets = len(getattr(eventWise, jet_name + "_Energy"))
        if n_jets > 0:
            tag_c, jet_c, n_jets = fit_to_tags(eventWise, jet_name, tag_pids=tag_pids)
            tag_coords.append(tag_c)
            jet_coords.append(jet_c)
        n_jets_formed.append(n_jets)
    #tag_coords = np.vstack(tag_coords)
    #jet_coords = np.vstack(jet_coords)
    return tag_coords, jet_coords, n_jets_formed


def score_rank(tag_coords, jet_coords):
    """
    

    Parameters
    ----------
    tag_coords :
        
    jet_coords :
        

    Returns
    -------

    """
    dims = 3
    scores = np.zeros(dims)
    uncerts = np.zeros(dims)
    for i in range(dims):
        scores[i], uncerts[i] = scipy.stats.spearmanr(tag_coords[:, i], jet_coords[:, i])
    return scores, uncerts


def get_catigories(records, content):
    """
    

    Parameters
    ----------
    records :
        
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
        
    records :
        
    columns :
        

    Returns
    -------

    """
    indices = [records.indices[n] for n in columns]
    here = [list(row[indices]) for row in content]
    print(columns)
    print(np.array(here))
    print(f"Num remaining {len(here)}")


def comparison_grid1(records):
    """
    

    Parameters
    ----------
    records :
        

    Returns
    -------

    """
    # we only want to look at the content that has been scored
    content = records.typed_array()[records.scored]
    # for homejet and fastjet we will just give best values
    mask = [c not in ("HomeJet", "FastJet", "SpectralMeanJet") for c in content[:, 1]]
    content = content[mask]
    # now filter for a rasonable number of jets
    min_jets = float(input("Give min mean jets; "))
    max_jets = float(input("Give max mean jets; "))
    mask = [False if c is None else (c > min_jets) and (c < max_jets) for c in content[:, records.indices["mean_njets"]]]
    content = content[mask]

    catigories = get_catigories(records, content)

    # select a colour axis and an x axis
    print(f"Avalible catigories are; {catigories.keys()}")
    x_name = InputTools.list_complete("Select an x axis; ", catigories.keys()).strip()
    x_values = catigories[x_name]
    c_name = InputTools.list_complete("Select a colour axis; ", catigories.keys()).strip()
    c_values = {v: i for i, v in enumerate(catigories[c_name])}
    s_name = InputTools.list_complete("Select a shape axis; ", catigories.keys()).strip()
    s_values = {v: i for i, v in enumerate(catigories[s_name])}
    columns = list(catigories.keys())
    print_remaining(content, records, columns)
    selections = f"jet class = Spectral\n{min_jets} < mean num jets < {max_jets}\n"
    # need to filter the rest down 
    for name in catigories.keys():
        if name in (x_name, c_name, s_name):
            continue
        columns.remove(name)
        i = records.indices[name]
        if name not in catigories:
            selections += f"{name} == {content[0, i]}\n"
            continue
        values = catigories[name]
        if isinstance(values[0], str):
            print(f"filtering {name}, values are; {values}")
            choice = InputTools.list_complete("Which one to keep? ", values).strip()
        else:
            print(f"filtering {name}, values are; {list(enumerate(values))}")
            choice = input("Which index to keep? ")
            if choice != '':
                choice = values[int(choice)]
        if choice == '':
            selections += f"{name} = any\n"
            continue
        selections += f"{name} = {choice}\n"
        content = content[np.array(content[:, i].tolist()) == choice]
        # update becuase some sections may be ruled out
        catigories = get_catigories(records, content)
        print_remaining(content, records, columns)
    
    fig, axes = plt.subplots(3, 3, sharex=True)
    dimensions = ["PT", "Rapidity", "Phi"]
    xs = np.array(content[:, records.indices[x_name]].tolist())
    n_colours = len(c_values)
    colour_map = matplotlib.cm.get_cmap('viridis')
    colour_positons = [colour_map(c) for c in np.linspace(0., 1., n_colours)]
    colour_coords = np.array([colour_positons[c_values[v]] for v in content[:, records.indices[c_name]]])
    colour_ticks = [str(n) for n in c_values]
    shape = np.array([s_values[s] for s in content[:, records.indices[s_name]]])
    markers = ['v', '^', '1', 's', '*', '+', 'd', 'P', '8', 'X']
    for dim, ax_trip in zip(dimensions, axes):
        ax0, ax1, ax2 = ax_trip
        ax0.set_axis_off()
        if dim == dimensions[0]:
            ax0.text(0, 0, selections)
        ax1.set_xlabel(x_name)
        ax1.set_ylabel(f"rank score {dim}")
        ax1.invert_yaxis()
        metric = f"score({dim})"
        ys = np.array(content[:, records.indices[metric]].tolist())
        for s_name in s_values:
            mask = shape == s_values[s_name]
            ax1.scatter(xs[mask], ys[mask], marker=markers[s_values[s_name]], c=colour_coords[mask], label=str(s_name))
        best_id, best = records.best(metric, "HomeJet")
        ax1.axhline(best, *ax1.get_xlim() , label="best HomeJet", c='b')
        ax1.text(0.5, best, "best traditional algorithm", c='b')
        ax2.set_xlabel(x_name)
        ax2.set_ylabel("Symmetrised % diff")
        ys = np.array(content[:, records.indices[f"symmetric_diff({dim})"]].tolist())
        for s_name in s_values:
            mask = shape == s_values[s_name]
            ax2.scatter(xs[mask], ys[mask], marker=markers[s_values[s_name]], c=colour_coords[mask], label=str(s_name))
        ax2.legend()
        mapable = plt.cm.ScalarMappable(cmap=colour_map)
        mapable.set_array([])
        cbar = fig.colorbar(mapable, ax=ax2, ticks=np.linspace(0, 1., len(colour_ticks)))
        cbar.set_label(c_name)
        cbar.ax.set_yticklabels(colour_ticks)
    plt.savefig("Test.png")
    plt.show()


def comparison1(records):
    """
    

    Parameters
    ----------
    records :
        

    Returns
    -------

    """
    plt.rcParams.update({'font.size': 22})
    # we only want to look at the content that has been scored
    content = records.typed_array()[records.scored]
    # for homejet and fastjet we will just give best values
    mask = [c not in ("HomeJet", "FastJet", "SpectralMeanJet") for c in content[:, 1]]
    content = content[mask]
    # now filter for a rasonable number of jets
    min_jets = float(input("Give min mean jets; "))
    max_jets = float(input("Give max mean jets; "))
    mask = [False if c is None else (c > min_jets) and (c < max_jets) for c in content[:, records.indices["mean_njets"]]]
    content = content[mask]

    catigories = get_catigories(records, content)

    # select a colour axis and an x axis
    print(f"Avalible catigories are; {catigories.keys()}")
    x_name = InputTools.list_complete("Select an x axis; ", catigories.keys()).strip()
    x_values = catigories[x_name]
    c_name = InputTools.list_complete("Select a colour axis; ", catigories.keys()).strip()
    c_values = {v: i for i, v in enumerate(catigories[c_name])}
    s_name = InputTools.list_complete("Select a shape axis; ", catigories.keys()).strip()
    s_values = {v: i for i, v in enumerate(catigories[s_name])}
    columns = list(catigories.keys())
    print_remaining(content, records, columns)
    selections = f"jet class = Spectral\n{min_jets} < mean num jets < {max_jets}\n"
    # need to filter the rest down 
    for name in catigories.keys():
        if name in (x_name, c_name, s_name):
            continue
        columns.remove(name)
        i = records.indices[name]
        if name not in catigories:
            selections += f"{name} == {content[0, i]}\n"
            continue
        values = catigories[name]
        if isinstance(values[0], str):
            print(f"filtering {name}, values are; {values}")
            choice = InputTools.list_complete("Which one to keep? ", values).strip()
        else:
            print(f"filtering {name}, values are; {list(enumerate(values))}")
            choice = input("Which index to keep? ")
            if choice != '':
                choice = values[int(choice)]
        if choice == '':
            selections += f"{name} = any\n"
            continue
        selections += f"{name} = {choice}\n"
        content = content[np.array(content[:, i].tolist()) == choice]
        # update becuase some sections may be ruled out
        catigories = get_catigories(records, content)
        print_remaining(content, records, columns)
    
    dimensions = ["PT", "Rapidity", "Phi"]
    xs = np.array(content[:, records.indices[x_name]].tolist())
    n_colours = len(c_values)
    colour_map = matplotlib.cm.get_cmap('viridis')
    colour_positons = [colour_map(c) for c in np.linspace(0., 1., n_colours)]
    colour_coords = np.array([colour_positons[c_values[v]] for v in content[:, records.indices[c_name]]])
    colour_ticks = [str(n) for n in c_values]
    shape = np.array([s_values[s] for s in content[:, records.indices[s_name]]])
    markers = ['v', '^', '1', 's', '*', '+', 'd', 'P', '8', 'X']
    for dim in dimensions:
        if dim == dimensions[0]:
            fig, ax0 = plt.subplots()
            ax0.set_axis_off()
            ax0.text(0, 0, selections)
            plt.show()
        fig, ax1 = plt.subplots()
        ax1.set_xlabel(x_name)
        ax1.set_ylabel(f"rank score {dim}")
        ax1.invert_yaxis()
        metric = f"score({dim})"
        ys = np.array(content[:, records.indices[metric]].tolist())
        for s_name in s_values:
            mask = shape == s_values[s_name]
            ax1.scatter(xs[mask], ys[mask], marker=markers[s_values[s_name]], s=100, c=colour_coords[mask], label=str(s_name))
        best_id, best = records.best(metric, "HomeJet")
        ax1.axhline(best, *ax1.get_xlim() , label="best HomeJet", c='b')
        ax1.text(0.5, best, "best traditional algorithm", c='b')
        ax1.legend()
        mapable = plt.cm.ScalarMappable(cmap=colour_map)
        mapable.set_array([])
        cbar = fig.colorbar(mapable, ax=ax1, ticks=np.linspace(0, 1., len(colour_ticks)))
        cbar.ax.set_yticklabels(colour_ticks)
        cbar.set_label(c_name)
        plt.show()
        fig, ax2 = plt.subplots()
        ax2.set_xlabel(x_name)
        ax2.set_ylabel("Symmetrised % diff")
        ys = np.array(content[:, records.indices[f"symmetric_diff({dim})"]].tolist())
        for s_name in s_values:
            mask = shape == s_values[s_name]
            ax2.scatter(xs[mask], ys[mask], marker=markers[s_values[s_name]], s=100, c=colour_coords[mask], label=str(s_name))
        ax2.legend()
        mapable = plt.cm.ScalarMappable(cmap=colour_map)
        mapable.set_array([])
        cbar = fig.colorbar(mapable, ax=ax2, ticks=np.linspace(0, 1., len(colour_ticks)))
        cbar.ax.set_yticklabels(colour_ticks)
        cbar.set_label(c_name)
        plt.show()


def comparison_grid2(records, rapidity=True, pt=True, phi=True):
    """
    

    Parameters
    ----------
    records :
        
    rapidity :
         (Default value = True)
    pt :
         (Default value = True)
    phi :
         (Default value = True)

    Returns
    -------

    """
    # we only want to look at the content that has been scored
    content = records.typed_array()[records.scored]
    mask = np.abs(content[:, records.indices["DeltaR"]] - 0.4) < 0.001
    num_eig = content[mask, records.indices["NumEigenvectors"]]
    exp_values = 2*content[mask, records.indices["ExponentMultiplier"]]
    scorePT = content[mask, records.indices["score(PT)"]]
    scoreRapidity = content[mask, records.indices["score(Rapidity)"]]
    sydPT = content[mask, records.indices["symmetric_diff(PT)"]]
    sydRapidity = content[mask, records.indices["symmetric_diff(Rapidity)"]]
    sydPhi = content[mask, records.indices["symmetric_diff(Phi)"]]
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
    ax1.set_xlabel("exponent")
    ax1.set_ylabel(f"rank score")
    xs = exp_values
    if len(set(num_eig)) > 1:
        is_inf = np.inf == num_eig
        max_eig = np.nanmax(num_eig[~is_inf])
        colour_map = matplotlib.cm.get_cmap('viridis')
        colours = colour_map(num_eig/max_eig + np.any(is_inf))
        colours = [tuple(c) for c in colours]
        colour_ticks = ["No spectral"] + [str(c+1) for c in range(int(max_eig))]
        if np.any(is_inf):
            colour_ticks += ["Max"]
            num_eig[is_inf] = max_eig + 1
    else:
        colours = None
    rap_marker = 'v'
    pt_marker = '^'
    phi_marker = 'o'
    if pt:
        ax1.scatter(xs, scorePT,
                     marker=pt_marker, c=colours, label="PT")
    if rapidity:
        ax1.scatter(xs, scoreRapidity,
                     marker=rap_marker, c=colours, label="Rapidity")
    ax2.set_xlabel("exponent")
    ax2.set_ylabel("Symmetrised % diff")
    if pt:
        ax2.scatter(xs, sydPT*100,
                    marker=pt_marker, c=colours, label="PT")
    if rapidity:
        ax2.scatter(xs, sydRapidity*100,
                    marker=rap_marker, c=colours, label="Rapidity")
    if phi:
        ax2.scatter(xs, sydPhi*100,
                    marker=phi_marker, c=colours, label="Phi")
    ax2.legend()
    ax2.set_ylim(0, max(100, np.max(np.concatenate((sydPT, sydRapidity, sydPhi)))))
    if colours is not None:
        norm = matplotlib.colors.Normalize(vmin=0, vmax=max_eig)
        mapable = matplotlib.cm.ScalarMappable(norm=norm, cmap=colour_map)
        mapable.set_array([])
        cbar = fig.colorbar(mapable, ax=ax2, ticks=np.linspace(0, max_eig, len(colour_ticks)))
        cbar.ax.set_yticklabels(colour_ticks)
    plt.show()


def calculated_grid(records, jet_name=None):
    """
    

    Parameters
    ----------
    records :
        
    jet_name :
         (Default value = None)

    Returns
    -------

    """
    names = {"FastJet": FormJets.Traditional, "HomeJet": FormJets.Traditional,
             "SpectralJet": FormJets.Spectral, "SpectralMeanJet": FormJets.SpectralMean}
    if jet_name is None:
        jet_name = InputTools.list_complete("Which jet? ", names.keys()).strip()
    default_params = names[jet_name].param_list
    param_list = sorted(default_params.keys())
    # we only want to look at the content that has been scored
    content = records.typed_array()[records.scored]
    catigories = {name: set(content[:, records.indices[name]])
                  for name in param_list}
    print("Parameter: num_catigories")
    for name in catigories:
        print(f"{name}: {len(catigories[name])}")
    horizontal_param = InputTools.list_complete("Horizontal param? ", param_list).strip()
    vertical_param = InputTools.list_complete("Vertical param? ", param_list).strip()
    horizontal_bins = sorted(catigories[horizontal_param])
    if isinstance(horizontal_bins[0], str):
        def get_h_index(value):
            """
            

            Parameters
            ----------
            value :
                

            Returns
            -------

            """
            return horizontal_bins.index(value)
    else:
        horizontal_bins_a = np.array(horizontal_bins)
        def get_h_index(value):
            """
            

            Parameters
            ----------
            value :
                

            Returns
            -------

            """
            return np.argmin(np.abs(horizontal_bins_a - value))
    vertical_bins = sorted(catigories[vertical_param])
    if isinstance(vertical_bins[0], str):
        def get_v_index(value):
            """
            

            Parameters
            ----------
            value :
                

            Returns
            -------

            """
            return vertical_bins.index(value)
    else:
        vertical_bins_a = np.array(vertical_bins)
        def get_v_index(value):
            """
            

            Parameters
            ----------
            value :
                

            Returns
            -------

            """
            return np.argmin(np.abs(vertical_bins_a - value))
    grid = [[[] for _ in horizontal_bins] for _ in vertical_bins]
    h_column = records.indices[horizontal_param]
    v_column = records.indices[vertical_param]
    for row in content:
        id_num = row[0]
        v_index = get_v_index(row[v_column])
        h_index = get_h_index(row[h_column])
        grid[v_index][h_index].append(id_num)
    table = [[value] + [len(entry) for entry in row] for value, row in zip(vertical_bins, grid)]
    first_row = [["\\".join([vertical_param, horizontal_param])] + horizontal_bins]
    table = first_row + table
    str_table = tabulate.tabulate(table, headers="firstrow")
    print(str_table)
    if InputTools.yesNo_question("Again? "):
        calculated_grid(records)


def soft_generic_equality(a, b):
    """
    

    Parameters
    ----------
    a :
        
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
    metric_cols = [records.indices[metric] for metric in metrics]
    invert_factor = np.fromiter((1 - 2*("symmetric_diff" in name) for name in metrics),
                                dtype=float)
    n_jets_col = records.indices['mean_njets']
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


def parameter_comparison(records, c_name="mean_njets", cuts=True):
    array = records.typed_array()[records.scored]
    col_names = ["score(PT)", "score(Rapidity)", "symmetric_diff(Phi)"]
    # filter anything that scored too close to zero in pt or rapidity
    # these are soem kind of bug
    small = 0.001
    for name in col_names:
        if "score" in name:
            array = array[array[:, records.indices[name]]>small]
    # now we have a scored valid selection
    y_cols = {name: array[:, records.indices[name]].astype(float) for name in col_names}
    col_names = ["cumulative"] + col_names
    y_cols["cumulative"] = cumulative_score(records, typed_array=array)
    c_col = array[:, records.indices[c_name]].astype(float)
    sufficient_jets = c_col > 0.5
    #sufficient_jets = array[:, 1] == 'SpectralAfterJet'
    if cuts:
        array = array[sufficient_jets]
        c_col = c_col[sufficient_jets]
        y_cols = {name: y_cols[name][sufficient_jets] for name in y_cols}
        sufficient_jets = sufficient_jets[sufficient_jets]
    # make the data dict
    data_dict = {}
    data_name = {}
    max_len = 15
    for name, i in records.indices.items():
        if 'uncert' in name or 'std' in name:
            continue
        data_dict[str(i)] = np.array([str(x)[:max_len] for x in array[:, i]])
        data_name[str(i)] = name # somr of the names have probem characters in them
    hover = bokeh.models.HoverTool(tooltips=[(data_name[i], "@" + i) for i in data_dict])
    data_dict['colour'] = c_col
    data_dict['alpha'] = [1. if suf else 0.2 for suf in sufficient_jets]
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
    ignore_cols = list(records.evaluation_columns) + ["jet_id", "match_error"]
    for name, col in records.indices.items():
        if name in ignore_cols:
            continue
        col_content = array[:, col]
        catigories = set(array[:, col])
        # None prevents propper sorting
        catigories.discard(None)
        if len(catigories) == 0:
            continue
        catigories = sorted(catigories)
        catigories += [None]
        if hasattr(catigories[0], '__iter__'):
            # it is either a string or a list,
            # the axis spacing will be maufactured
            scale = list(range(len(catigories)))
            positions = np.fromiter((scale[catigories.index(x)] for x in col_content),
                                    dtype=float)
            label_dict = {tick: str(cat) for cat, tick in zip(catigories, scale)}
        else:
            scale = np.array(catigories)
            real = np.fromiter((x for x in catigories[:-1] if np.isfinite(x)),
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
        plots.append([])
        for col_name, y_col in y_cols.items():
            # first the intresting jets
            data_dict['x'] = positions
            data_dict['y'] = y_col
            if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(y_col)):
                st()
            source = bokeh.models.ColumnDataSource(data=data_dict)
            p = bokeh.plotting.figure(tools=[hover], title=name)
            # p.xaxis.axis_label_text_font_size = "30pt" not working
            # p.xaxis.ticker = positions
            p.xaxis.ticker = scale
            p.xaxis.major_label_overrides = label_dict
            p.xaxis.major_label_orientation = "vertical"
            p.yaxis.axis_label = col_name
            if "symmetric_diff" in col_name:
                p.y_range.flipped = True
            p.scatter('x', 'y', size=10, source=source,
                     fill_color=bokeh.transform.transform('colour', mapper),
                     fill_alpha='alpha', line_alpha='alpha',
                     marker=markers)
            plots[-1].append(p)
    all_p = bokeh.layouts.gridplot(plots, plot_width=400, plot_height=400)
    bokeh.io.show(all_p)
    return all_p


def cumulative_score(records, typed_array=None):
    if typed_array is None:
        typed_array = records.typed_array()
    col_names = ["score(PT)", "score(Rapidity)", "symmetric_diff(Phi)"]
    cols = {name: typed_array[:, records.indices[name]].astype(float) for name in col_names}
    cols_widths = {name: np.nanstd(col[col>0.01]) for name, col in cols.items()}
    cumulative = sum((col*(1-2*("symmetric_diff" in name)))/cols_widths[name]
                     for name, col in cols.items())
    return cumulative


class Records:
    """ """
    delimiter = '\t'
    evaluation_columns = ("score(PT)", "score_uncert(PT)", "symmetric_diff(PT)", "symdiff_std(PT)",
                          "score(Rapidity)", "score_uncert(Rapidity)", "symmetric_diff(Rapidity)", "symdiff_std(Rapidity)",
                          "score(Phi)", "score_uncert(Phi)", "symmetric_diff(Phi)", "symdiff_std(Phi)",
                          "mean_njets", "std_njets")
    def __init__(self, file_path):
        self.file_path = file_path
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r') as existing:
                reader = csv.reader(existing, delimiter=self.delimiter)
                header = next(reader)
                assert header[1] == 'jet_class'
                self.param_names = header[2:]
                self.indices = {name: i+2 for i, name in enumerate(self.param_names)}
                self.indices['jet_id'] = 0
                self.indices['jet_class'] = 1
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
        self.uid_length = len(str(self.next_uid))

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
        jet_classes = {"HomeJet": FormJets.Traditional,
                       "FastJet": FormJets.Traditional,
                       "SpectralJet": FormJets.Spectral,
                       "SpectralMeanJet": FormJets.SpectralMean}
        typed_content = []
        for row in self.content:
            id_num = int(row[0])
            jet_class = row[1]
            typed_content.append([id_num, jet_class])
            for param_name, entry in zip(self.param_names, row[2:]):
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
                typed_content[-1].append(typed)
        # got to be an awkward array because numpy hates mixed types
        return np.array(typed_content)

    @property
    def jet_ids(self):
        """ """
        ids = [int(row[0]) for row in self.content]
        return ids

    @property
    def scored(self):
        """ """
        if 'mean_njets' not in self.param_names:
            return np.full(len(self.content), False)
        scored = [row[self.indices["mean_njets"]] not in ('', None)
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
        new_blanks = ['' for _ in new_params]
        self.content = [row + new_blanks for row in self.content]

    def append(self, jet_class, param_dict, existing_idx=None, write_now=True):
        """
        gives the new jet a unique ID and returns that value

        Parameters
        ----------
        jet_class :
            
        param_dict :
            
        existing_idx :
             (Default value = None)
        write_now :
             (Default value = True)

        Returns
        -------

        """
        if existing_idx is None:
            chosen_id = self.next_uid
        else:
            assert existing_idx not in self.jet_ids
            chosen_id = existing_idx
        new_row = [f"{chosen_id:0{self.uid_length}d}", jet_class]
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
 
    def check_eventWise_match(self, eventWise, jet_id, jet_name):
        eventWise.selected_index = None
        jet_params = FormJets.get_jet_params(eventWise, jet_name, add_defaults=True)
        content = self.typed_array()
        row = content[content[:, 0] == jet_id][0]
        for p_name in jet_params:
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
        jet_names = {c.split('_', 1)[0] for c in eventWise.columns
                     if (not c.startswith('JetInputs')) and 'Jet' in c}
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

    def score(self, eventWise):
        """
        

        Parameters
        ----------
        eventWise :
            

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
                tag_coords, jet_coords, n_jets_formed = fit_all_to_tags(eventWise, name, silent=True)
                tag_coords = np.vstack(tag_coords)
                jet_coords = np.vstack(jet_coords)
                if len(jet_coords) > 0:
                    scores, uncerts = score_rank(tag_coords, jet_coords)
                    syd = 2*np.abs(tag_coords - jet_coords)/(np.abs(tag_coords) + np.abs(jet_coords))
                else:
                    scores = [0., 0., 0.]
                    uncerts = [0., 0., 0.]
                    syd = [[1.], [1.], [1.]]
                    n_jets_formed = [0]
                row[self.indices["mean_njets"]] = np.mean(n_jets_formed)
                row[self.indices["std_njets"]] = np.std(n_jets_formed)
                row[self.indices["score(PT)"]] = scores[0]
                row[self.indices["score_uncert(PT)"]] = uncerts[0]
                row[self.indices["score(Rapidity)"]] = scores[1]
                row[self.indices["score_uncert(Rapidity)"]] = uncerts[1]
                row[self.indices["score(Phi)"]] = scores[2]
                row[self.indices["score_uncert(Phi)"]] = uncerts[2]
                # but the symetric difernce for phi should be angular
                row[self.indices["symmetric_diff(PT)"]] = np.mean(syd[0])
                row[self.indices["symdiff_std(PT)"]] = np.std(syd[0])
                row[self.indices["symmetric_diff(Rapidity)"]] = np.mean(syd[1])
                row[self.indices["symdiff_std(Rapidity)"]] = np.std(syd[1])
                row[self.indices["symmetric_diff(Phi)"]] = np.mean(syd[2])
                row[self.indices["symdiff_std(Phi)"]] = np.std(syd[2])
                if np.nan in row:
                    st()
                    row
        self.write()

    def best(self, metric='cumulative', jet_class=None, invert=None,
             start_mask=None, return_cols=None, num_items=1):
        """
        

        Parameters
        ----------
        metric :
            
        jet_class :
             (Default value = None)
        invert :
             (Default value = None)

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
    records = Records("records.csv")
    comparison1(records)

