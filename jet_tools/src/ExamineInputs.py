""" File to examine the nature of the input data """
import awkward
from jet_tools.src import Components, FormShower, PDGNames
from matplotlib import pyplot as plt
import numpy as np
from ipdb import set_trace as st
from collections import Counter

def get_signal_masks(eventWise, observables=True):
    eventWise.selected_index = None
    signal_mask = (eventWise.PT*0).astype(bool)
    background_mask = (eventWise.PT*0).astype(bool)
    vector_sum = np.vectorize(np.sum)
    if observables:
        vector_len = np.vectorize(len)
        total = vector_len(eventWise.JetInputs_SourceIdx)
        signal_idxs = eventWise.DetectableTag_Leaves.flatten(axis=1)
        signal_mask[signal_idxs] = True
        background_mask[eventWise.JetInputs_SourceIdx] = True
        background_mask[signal_idxs] = False
    else:
        total = vector_sum(eventWise.Is_leaf)
        background_mask[eventWise.Is_leaf] = True
        for i, bidxs in enumerate(eventWise.BQuarkIdx):
            eventWise.selected_index = i
            signal_idx = list(FormShower.descendant_idxs(eventWise, *bidxs))
            signal_mask[i][signal_idx] = True
            background_mask[i][signal_idx] = False
        eventWise.selected_index = None
    signal = vector_sum(signal_mask)
    return signal, total, signal_mask.astype(bool), background_mask.astype(bool)


def plot_event_counts(eventWise, observables=True, ax_arr=None):
    if ax_arr is None:
        fig, ax_arr = plt.subplots(3, 1, sharex=True)
    else:
        fig = plt.gcf()
    fig.set_size_inches(6, 7)
    ax1, ax2, ax3 = ax_arr
    signal, total, signal_mask, background_mask = get_signal_masks(eventWise, observables)
    percentages = signal/total
    bin_heights, bins, patches = ax1.hist(total, bins=40, histtype='stepfilled',
                                         color='grey', label="Frequency of events by size")
    ax1.set_ylabel("Counts")
    if observables:
        xlabel = "Number of particles that pass cuts"
    else:
        xlabel = "Number of particles without cuts"
    #ax1.set_xlabel(xlabel)
    #ax2.set_xlabel(xlabel)
    ax3.set_xlabel(xlabel)
    # loop over the bins
    percent_signal = [[], [], []]
    signal_pt = [[], [], []]
    background_pt = [[], [], []] 
    all_signal_pt = eventWise.PT[signal_mask]
    all_background_pt = eventWise.PT[background_mask]
    for bin_start, bin_end in zip(bins[:-1], bins[1:]):
        in_bin = np.logical_and(total > bin_start, total <= bin_end)
        if not np.any(in_bin):
            percentages_here = [np.nan]
            sig_here = bg_here = [np.nan]
        else:
            percentages_here = percentages[in_bin]
            sig_here = all_signal_pt[in_bin].flatten()
            bg_here = all_background_pt[in_bin].flatten()
        percent_signal[0].append(np.mean(percentages_here))
        percent_signal[1].append(np.quantile(percentages_here, 0.25))
        percent_signal[2].append(np.quantile(percentages_here, 0.75))
        signal_pt[0].append(np.mean(sig_here))
        signal_pt[1].append(np.quantile(sig_here, 0.25))
        signal_pt[2].append(np.quantile(sig_here, 0.75))
        background_pt[0].append(np.mean(bg_here))
        background_pt[1].append(np.quantile(bg_here, 0.25))
        background_pt[2].append(np.quantile(bg_here, 0.75))
    percent_signal = [[percent_signal[i][0]] + percent_signal[i] for i in range(3)]
    signal_pt = [[signal_pt[i][0]] + signal_pt[i] for i in range(3)]
    background_pt = [[background_pt[i][0]] + background_pt[i] for i in range(3)]
    ax2.fill_between(bins, percent_signal[1], percent_signal[2], color='g', alpha=0.3)
    ax2.plot(bins, percent_signal[0], color='g', label="Mean $b$-shower fraction by size")
    ax2.set_ylabel("Fraction from $b$-shower")
    ax2.legend()
    ax3.fill_between(bins, signal_pt[1], signal_pt[2], color='r', alpha=0.3)
    ax3.plot(bins, signal_pt[0], color='r', label="$b$-shower particle")
    ax3.set_ylabel("Mean $p_T$")
    ax3.fill_between(bins, background_pt[1], background_pt[2], color='b', alpha=0.3)
    ax3.plot(bins, background_pt[0], color='b', label="Non $b$-shower particle")
    ax3.set_ylabel("Mean $p_T$")
    ax3.legend()
    fig.set_tight_layout(True)


def get_hard_structure(eventWise, required_pids=[25,35]):
    assert eventWise.selected_index is not None
    chain_pids = []
    chains = []
    parents_to_follow = []
    all_found = set()
    # start by identifying the chains of any required_idxs
    for pid in required_pids:
        locations = set(np.where(eventWise.MCPID == pid)[0])
        while locations:  # make another chain
            chain_pids.append(pid)
            new_chain, parents, _ = make_chain(eventWise, locations.pop())
            chains.append(new_chain)
            all_found.update(new_chain)
            locations = locations - set(new_chain)
            if all_found.isdisjoint(parents):
                parents_to_follow += parents.tolist()
    assert all_found == set([i for c in chains for i in c])
    # now make chains out of all the parents
    while parents_to_follow:
        parent = parents_to_follow.pop()
        if parent in all_found:
            continue
        new_chain, parents, pid = make_chain(eventWise, parent)
        chain_pids.append(pid)
        chains.append(new_chain)
        all_found.update(new_chain)
        parents_to_follow += parents.tolist()
    assert all_found == set([i for c in chains for i in c]), f"chains = {chains}"
    return chain_pids, chains


def get_structure_relations(eventWise, chain_pids, chains):
    leads_to = []
    comes_from = []
    for pid, chain in zip(chain_pids, chains):
        outward = set(eventWise.Children[chain].flatten())
        outward = [chain_pids[i] for i, c in enumerate(chains)
                   if not outward.isdisjoint(c)]
        if pid in outward:  # removes self refernces
            outward.remove(pid)
        leads_to.append(outward)
        inward = set(eventWise.Parents[chain].flatten())
        inward = [chain_pids[i] for i, c in enumerate(chains)
                  if not inward.isdisjoint(c)]
        if pid in inward:
            inward.remove(pid)
        comes_from.append(inward)
    ider = PDGNames.IDConverter()
    relations = []
    for pid, list_from, list_to in zip(chain_pids, comes_from, leads_to):
        string = ider[pid]
        if list_from:
            list_from = ','.join(sorted([ider[p] for p in list_from]))
            string += f", generated by {list_from}"
        else:
            string += " from beam"
        if list_to:
            list_to = ','.join(sorted([ider[p] for p in list_to]))
            string += f", it generates {list_to}"
        relations.append(string)
    return relations

    


def get_structure_emmisions(eventWise, chains):
    all_found = {p for c in chains for p in c}
    emmisions = []
    for chain in chains:
        out = set(eventWise.Children[chain].flatten()) - all_found
        emmisions.append(out)
    return emmisions


def mass_from_emmisions(eventWise, emmisions):
    visibles = eventWise.JetInputs_SourceIdx.tolist()
    from_emmision = [[] for _ in visibles]
    emmisions_energy = [[] for _ in visibles]
    for emission_n, chain_e in enumerate(emmisions):
        for particle in chain_e:
            decendants = FormShower.descendant_idxs(eventWise, particle)
            for d in decendants:
                if d not in visibles:
                    continue
                index = visibles.index(d)
                energy = eventWise.Energy[d]
                from_emmision[index].append(emission_n)
                emmisions_energy[index].append(energy)
    n_emissions = emission_n + 1
    mass2 = np.empty(n_emissions)
    visible_energy = eventWise.JetInputs_Energy
    visible_px = eventWise.JetInputs_Px
    visible_py = eventWise.JetInputs_Py
    visible_pz = eventWise.JetInputs_Pz
    for emission_n in range(n_emissions):
        fraction_of = []
        for sources, energies in zip(from_emmision, emmisions_energy):
            if emission_n not in sources:
                fraction_of.append(0)
                continue
            fraction_of.append(energies[sources.index(emission_n)]/
                               np.sum(energies))
        energy = np.sum(visible_energy*fraction_of)
        px = np.sum(visible_px*fraction_of)
        py = np.sum(visible_py*fraction_of)
        pz = np.sum(visible_pz*fraction_of)
        mass2[emission_n] = energy**2 - px**2 - py**2 - pz**2
    return np.sqrt(mass2)



def make_chain(eventWise, location):
    pid = eventWise.MCPID[location]
    chain = [location]
    while True:  # go up the chain
        parents = eventWise.Parents[chain[-1]]
        parent_pids = eventWise.MCPID[parents]
        if pid not in parent_pids:
            break
        for p in parents[parent_pids == pid]:
            chain.append(p)
    while True:  # go down the chain
        children = eventWise.Children[chain[0]]
        child_pids = eventWise.MCPID[children]
        if pid not in child_pids:
            break
        for c in children[child_pids == pid]:
            chain = [c] + chain
    # last set of parents will be parents to the chain
    return chain, parents, pid
            
        
def get_origins_of_visible(eventWise):
    if isinstance(eventWise, str):
        eventWise = Components.EventWise.from_file(eventWise)
    mass_origins = Counter()
    n_events = len(eventWise.JetInputs_SourceIdx)
    for event_n in range(n_events):
        if event_n %100 == 0:
            print(f"{event_n/n_events:.1%}", end='\r', flush=True)
        eventWise.selected_index = event_n
        chain_pids, chains = get_hard_structure(eventWise, required_pids=[35])
        relations = get_structure_relations(eventWise, chain_pids, chains)
        emmisions = get_structure_emmisions(eventWise, chains)
        masses = mass_from_emmisions(eventWise, emmisions)
        for name, mass in zip(relations, masses):
            mass_origins[name] += mass
    mass_origins = {name: mass/n_events for name, mass in mass_origins.items()}
    return mass_origins



