""" compare two jet clustering techniques """
import os
import pickle
import matplotlib
import awkward
from ipdb import set_trace as st
from tree_tagger import Components, TrueTag, InputTools
import sklearn.metrics
import sklearn.preprocessing
from matplotlib import pyplot as plt
import numpy as np
import scipy.stats

def rand_score(eventWise, jet_name1, jet_name2):
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
    plt.hist(scores, bins=40, density=True, histtype='stepfilled')
    mean = np.mean(scores)
    std = np.std(scores)
    plt.vlines([mean - 2*std, mean, mean + 2*std], 0, 1, colors=['orange', 'red', 'orange'])
    plt.xlabel(score_name)
    plt.ylabel("Density")
    plt.title(f"Similarity of {jet_name1} and {jet_name2} (for {len(scores)} events)")



def pseudovariable_differences(eventWise, jet_name1, jet_name2, var_name="Rapidity"):
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


def fit_to_tags(eventWise, jet_name, event_n=None, tag_pids=None, jet_pt_cut=30.):
    if event_n is None:
        assert eventWise.selected_index is not None
    else:
        eventWise.selected_index = event_n
    inputidx_name = jet_name + "_InputIdx"
    rootinputidx_name = jet_name+"_RootInputIdx"
    jet_pt = eventWise.match_indices(jet_name+"_PT", inputidx_name, rootinputidx_name).flatten()
    if jet_pt_cut is None:
        mask = np.ones_like(jet_pt)
    else:
        mask = jet_pt>jet_pt_cut
        if not np.any(mask):
            empty = np.array([]).reshape((-1, 3))
            return empty, empty
    jet_pt = jet_pt[mask]
    jet_rapidity = eventWise.match_indices(jet_name+"_Rapidity", inputidx_name, rootinputidx_name).flatten()[mask]
    jet_phi = eventWise.match_indices(jet_name+"_Phi", inputidx_name, rootinputidx_name).flatten()[mask]
    tag_idx = TrueTag.tag_particle_indices(eventWise, tag_pids=tag_pids)
    if len(tag_idx) == 0:
        empty = np.array([]).reshape((-1, 3))
        return empty, empty
    tag_rapidity = eventWise.Rapidity[tag_idx]
    tag_phi = eventWise.Phi[tag_idx]
    tag_pt = eventWise.PT[tag_idx]
    try: # if there is not more than one jet these methods fail
        # normalise the tag_pt and jet_pts, it's reasonable to think these could be corrected to match
        normed_jet_pt = sklearn.preprocessing.normalize(jet_pt.reshape(1, -1)).flatten()
    except ValueError:
        normed_jet_pt = np.ones(1)
    try:
        normed_tag_pt = sklearn.preprocessing.normalize(tag_pt.reshape(1, -1)).flatten()
    except ValueError:
        normed_tag_pt = np.ones(1)
    # divide tag and jet rapidity by the abs mean of both, effectivly collectivly normalising them
    abs_mean = np.mean((np.mean(np.abs(jet_rapidity)), np.mean(np.abs(tag_rapidity))))
    normed_jet_rapidity = jet_rapidity/abs_mean
    normed_tag_rapidity = tag_rapidity/abs_mean
    # divide the phi coordinates by pi for the same effect
    normed_jet_phi = 2*jet_phi/np.pi
    normed_tag_phi = 2*tag_phi/np.pi
    # find angular distances as they are invarient of choices
    phi_distance = np.vstack([[j_phi - t_phi for j_phi in normed_jet_phi]
                             for t_phi in normed_tag_phi])
    # remeber that these values are normalised
    phi_distance[phi_distance > 2] = 4 - phi_distance[phi_distance > 2]
    rapidity_distance = np.vstack([[j_rapidity - t_rapidity for j_rapidity in normed_jet_rapidity]
                                   for t_rapidity in normed_tag_rapidity])
    angle_dist2 = np.square(phi_distance) + np.square(rapidity_distance)
    # starting with the highest PT tag working to the lowest PT tag
    # assign tags to jets and recalculate the distance as needed
    matched_jet = np.zeros_like(tag_pt, dtype=int) - 1
    current_pt_offset_for_jet = -np.copy(normed_jet_pt)
    for tag_idx in np.argsort(tag_pt):
        this_pt = normed_tag_pt[tag_idx]
        new_pt_dist2 = np.square(current_pt_offset_for_jet + this_pt)
        allocate_to = np.argmin(new_pt_dist2 + angle_dist2[tag_idx])
        matched_jet[tag_idx] = allocate_to
        current_pt_offset_for_jet[allocate_to] += this_pt
    # now calculate what fraction of jet PT the tag actually receves
    chosen_jets = list(set(matched_jet))
    pt_fragment = np.zeros_like(matched_jet, dtype=float)
    for jet_idx in chosen_jets:
        tag_idx_here = np.where(matched_jet == jet_idx)[0]
        pt_fragment[tag_idx_here] = tag_pt[tag_idx_here]/np.sum(tag_pt[tag_idx_here])
    tag_coords = np.vstack((tag_pt, tag_rapidity, tag_phi)).transpose()
    jet_coords = np.vstack((jet_pt[matched_jet]*pt_fragment, jet_rapidity[matched_jet], jet_phi[matched_jet])).transpose()
    return tag_coords, jet_coords
    

def fit_all_to_tags(eventWise, jet_name, silent=False):
    eventWise.selected_index = None
    n_events = len(getattr(eventWise, jet_name + "_Energy"))
    tag_pids = np.genfromtxt('tree_tagger/contains_b_quark.csv', dtype=int)
    tag_coords = []
    jet_coords = []
    for event_n in range(n_events):
        if event_n % 10 == 0 and not silent:
            print(f"{100*event_n/n_events}%", end='\r', flush=True)
        eventWise.selected_index = event_n
        n_jets = len(getattr(eventWise, jet_name + "_Energy"))
        if n_jets > 0:
            tag_c, jet_c = fit_to_tags(eventWise, jet_name, tag_pids=tag_pids)
            tag_coords.append(tag_c)
            jet_coords.append(jet_c)
    tag_coords = np.vstack(tag_coords)
    jet_coords = np.vstack(jet_coords)
    return tag_coords, jet_coords


def score_rank(tag_coords, jet_coords):
    dims = 3
    scores = np.zeros(dims)
    uncerts = np.zeros(dims)
    for i in range(dims):
        scores[i], uncerts[i] = scipy.stats.spearmanr(tag_coords[:, i], jet_coords[:, i])
    return scores, uncerts


def unpack_name(name):
    if name.startswith("FastJet"):
        start = "FastJetDeltaR"
        name = name[len(start):]
        num_eig = np.nan
    elif name.startswith("HomeJet"):
        start = "HomeJetDeltaR"
        name = name[len(start):]
        num_eig = np.nan
    elif name.startswith("SpectralJet"):
        start = "SpectralJetNumEig"
        name = name[len(start):]
        num_eig, name = name.split("DeltaR")
        num_eig = int(num_eig)
    # find out how long the deltaR extends
    char_num = 0
    while name[char_num] == 'p' or name[char_num].isdigit():
        char_num += 1
    deltaR = float(name[:char_num].replace('p', '.'))
    exponent = name[char_num:]
    return num_eig, exponent, deltaR



def create_axes_data(eventWise, axes_data=None):
    if axes_data is None:
        axes_data = {}
    else:
        axes_data = {name: axes_data[name].tolist() for name in axes_data}
    jet_names = [name.split('_', 1)[0] for name in eventWise.columns
                 if "Jet" in name and "JetInput" not in name]
    jet_names = sorted(set(jet_names))
    num_names = len(jet_names)
    for i, name in enumerate(jet_names):
        if i % 2 == 0:
            print(f"{100*i/num_names}%", end='\r', flush=True)
        if i % 30 == 0:
            # delete and reread the eventWise, this should free up some ram
            # because the eventWise is lazy loading
            path = os.path.join(eventWise.dir_name, eventWise.save_name)
            del eventWise
            eventWise = Components.EventWise.from_file(path)
        num_eig, exponent, deltaR = unpack_name(name)
        if exponent not in axes_data:
            axes_data[exponent] = {"num eigenvectors": [], "deltaR":[], "scores":[], "uncert(score)":[],
                                   "symmetric difference":[], "std(symmetric difference)":[]}
        else:
            idx = [i for i, (dr, ne) in 
                   enumerate(zip(axes_data[exponent]["deltaR"], axes_data[exponent]["num eigenvectors"]))
                   if dr == deltaR and ne == num_eig]
            if len(idx) == 1:
                continue  # already done this
            elif len(idx) > 1:
                raise ValueError(f"Multiple occurances of {num_eig}, {deltaR}, {exponent}")
        num_eig, exponent, deltaR = unpack_name(name)
        axes_data[exponent]["num eigenvectors"].append(num_eig)
        axes_data[exponent]["deltaR"].append(deltaR)
        tag_coords, jet_coords = fit_all_to_tags(eventWise, name, silent=True)
        scores, uncerts = score_rank(tag_coords, jet_coords)
        axes_data[exponent]["scores"].append(scores)
        axes_data[exponent]["uncert(score)"].append(uncerts)
        syd = 2*np.abs(tag_coords - jet_coords)/(np.abs(tag_coords) + np.abs(jet_coords))
        # but the symetric difernce for phi should be angular
        syd[:, 2] = np.abs(Components.angular_distance(tag_coords[:, 2], jet_coords[:, 2]))/np.pi
        axes_data[exponent]["symmetric difference"].append(np.mean(syd, axis=0))
        axes_data[exponent]["std(symmetric difference)"].append(np.std(syd, axis=0))
    axes_data = {ax_name:{part: np.array(axes_data[ax_name][part]) for part in axes_data[ax_name]} for ax_name in axes_data}
    return axes_data 


def comparison_grid1(axes_data, rapidity=True, pt=True, phi=True):
    axis_names = sorted(axes_data.keys())
    axis_names1 = [n for n in axis_names if len(n) < 4]
    fig, axes = plt.subplots(len(axis_names1), 2, sharex=True)
    if len(axes.shape) < 2:
        axes = [axes]
    for i, (ax_pair, ax_name) in enumerate(zip(axes, axis_names1)):
        ax1, ax2 = ax_pair
        ax1.set_xlabel("$\\Delta R$")
        ax1.set_ylabel(f"{ax_name} rank score")
        xs = axes_data[ax_name]["deltaR"]
        num_eig = axes_data[ax_name]["num eigenvectors"]
        if len(set(num_eig)) > 1:
            max_eig = np.nanmax(num_eig)
            colour_map = matplotlib.cm.get_cmap('viridis')
            colours = colour_map(num_eig/max_eig)
            colours = [tuple(c) for c in colours]
            colour_ticks = ["No spectral"] + [str(c+1) for c in range(int(max_eig))]
        else:
            colours = None
        rap_marker = 'v'
        pt_marker = '^'
        phi_marker = 'o'
        if pt:
            ax1.scatter(xs, axes_data[ax_name]["scores"][:, 0],
                         marker=pt_marker, c=colours, label="PT")
        if rapidity:
            ax1.scatter(xs, axes_data[ax_name]["scores"][:, 1], 
                         marker=rap_marker, c=colours, label="Rapidity")
        ax2.set_xlabel("$\\Delta R$")
        ax2.set_ylabel("Symmetrised % diff")
        if pt:
            ax2.scatter(xs, axes_data[ax_name]["symmetric difference"][:, 0]*100,
                        marker=pt_marker, c=colours, label="PT")
        if rapidity:
            ax2.scatter(xs, axes_data[ax_name]["symmetric difference"][:, 1]*100,
                        marker=rap_marker, c=colours, label="Rapidity")
        if phi:
            ax2.scatter(xs, axes_data[ax_name]["symmetric difference"][:, 2]*100,
                        marker=phi_marker, c=colours, label="Phi")
        ax2.legend()
        ax2.set_ylim(0, max(100, np.max(axes_data[ax_name]["symmetric difference"][:, [pt, rapidity, phi]])))
        if colours is not None:
            norm = matplotlib.colors.Normalize(vmin=0, vmax=max_eig)
            mapable = matplotlib.cm.ScalarMappable(norm=norm, cmap=colour_map)
            mapable.set_array([])
            cbar = fig.colorbar(mapable, ax=ax2, ticks=np.linspace(0, max_eig, len(colour_ticks)))
            cbar.ax.set_yticklabels(colour_ticks)
    plt.show()
    return axes_data


def comparison_grid2(axes_data, rapidity=True, pt=True, phi=True):
    axis_names = sorted(axes_data.keys())
    exp_values = []
    num_eig = []
    score = []
    syd = []
    for name in axis_names:
        mask = np.abs(axes_data[name]["deltaR"] - 0.4) < 0.001
        eig_here = axes_data[name]["num eigenvectors"][mask]
        n_entries = len(eig_here)
        num_eig += eig_here.tolist()
        score.append(axes_data[name]["scores"][mask])
        syd.append(axes_data[name]["symmetric difference"][mask])
        sign = 0 if name.endswith("CA") else (-1 if name.startswith("A") else 1)
        num = ''.join([c for c in name if c.isdigit()])
        if num == '':
            num = 1.
        else:
            num = 0.1*float(num)
        exp_values += [2*sign*num]*n_entries
    score = np.vstack(score)
    syd = np.vstack(syd)
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
    ax1.set_xlabel("exponent")
    ax1.set_ylabel(f"rank score")
    xs = exp_values
    if len(set(num_eig)) > 1:
        max_eig = np.nanmax(num_eig)
        colour_map = matplotlib.cm.get_cmap('viridis')
        colours = colour_map(num_eig/max_eig)
        colours = [tuple(c) for c in colours]
        colour_ticks = ["No spectral"] + [str(c+1) for c in range(int(max_eig))]
    else:
        colours = None
    rap_marker = 'v'
    pt_marker = '^'
    phi_marker = 'o'
    if pt:
        ax1.scatter(xs, score[:, 0],
                     marker=pt_marker, c=colours, label="PT")
    if rapidity:
        ax1.scatter(xs, score[:, 1], 
                     marker=rap_marker, c=colours, label="Rapidity")
    ax2.set_xlabel("exponent")
    ax2.set_ylabel("Symmetrised % diff")
    if pt:
        ax2.scatter(xs, syd[:, 0]*100,
                    marker=pt_marker, c=colours, label="PT")
    if rapidity:
        ax2.scatter(xs, syd[:, 1]*100,
                    marker=rap_marker, c=colours, label="Rapidity")
    if phi:
        ax2.scatter(xs, syd[:, 2]*100,
                    marker=phi_marker, c=colours, label="Phi")
    ax2.legend()
    ax2.set_ylim(0, max(100, np.max(syd[:, [pt, rapidity, phi]])))
    if colours is not None:
        norm = matplotlib.colors.Normalize(vmin=0, vmax=max_eig)
        mapable = matplotlib.cm.ScalarMappable(norm=norm, cmap=colour_map)
        mapable.set_array([])
        cbar = fig.colorbar(mapable, ax=ax2, ticks=np.linspace(0, max_eig, len(colour_ticks)))
        cbar.ax.set_yticklabels(colour_ticks)
    plt.show()
    return axes_data


if __name__ == '__main__':
    path = InputTools.get_file_name("Where is the eventwise or axes_data? ", '.awkd')
    if path.endswith(".awkd"):
        path2 = path[:-5] + ".pkl"
        if os.path.exists(path2):
            with open(path2, 'rb') as pickle_file:
                axes_data = pickle.load(pickle_file)
        else:
            axes_data = None
        axes_data = create_axes_data(Components.EventWise.from_file(path), axes_data)
        with open(path2, 'wb') as pickle_file:
            pickle.dump(axes_data, pickle_file)
    else:
        with open(path, 'rb') as pickle_file:
            axes_data = pickle.load(pickle_file)
        if InputTools.yesNo_question("Add another? "):
            path = InputTools.get_file_name("Where is the eventwise or axes_data? ", '.awkd')
            with open(path, 'rb') as pickle_file:
                axes_data2 = pickle.load(pickle_file)
            for name in axes_data2:
                if (name not in axes_data) or (len(axes_data[name]) < len(axes_data2[name])):
                    axes_data[name] = axes_data2[name]
    go = True
    while go:
        rap = InputTools.yesNo_question("Rapidity? ")
        phi = InputTools.yesNo_question("Phi? ")
        pt = InputTools.yesNo_question("PT? ")
        comparison_grid1(axes_data=axes_data, pt=pt, rapidity=rap, phi=phi)
        comparison_grid2(axes_data=axes_data, pt=pt, rapidity=rap, phi=phi)
        go = InputTools.yesNo_question("Again? ")


