""" Jet quality measures as discribed in https://arxiv.org/pdf/0810.1304.pdf """
import numpy as np
from tree_tagger import MassPeaks
import scipy.optimize


def sorted_masses(eventWise, jet_name, mass_function='highest pt pair',
                  jet_pt_cut=20.):
    if mass_function == 'highest pt pair':
        all_masses, pairs, pair_masses = MassPeaks.all_PT_pairs(eventWise, jet_name, jet_pt_cut)
        idx = next(i for i, p in enumerate(pairs) if set(p) == {0, 1})
        masses = pair_masses[idx]
    else:
        masses = mass_function(eventWise, jet_name, jet_pt_cut)
    np.sort(masses)
    return masses


def quality_width(eventWise, jet_name, fraction=0.15, mass_function='highest pt pair',
                  jet_pt_cut=20.):
    eventWise.selected_index = None
    n_events = len(getattr(eventWise, jet_name+'_InputIdx'))
    target_counts = int(np.ceil(fraction*n_events))
    masses = sorted_masses(eventWise, jet_name, mass_function, jet_pt_cut)
    if target_counts > len(masses):
        msg = f"Cannot acheve a fraction of {fraction} with {len(masses)} masses from {n_events} events"
        raise RuntimeError(msg)
    widths = masses[:-target_counts] - masses[target_counts:]
    best_width = np.min(widths)
    return best_width


def quality_fraction(eventWise, jet_name, multiplier, mass_of_obj, mass_function='highest pt pair',
                     jet_pt_cut=20.):
    eventWise.selected_index = None
    n_events = len(getattr(eventWise, jet_name+'_InputIdx'))
    masses = sorted_masses(eventWise, jet_name, mass_function, jet_pt_cut)
    window = multiplier*mass_of_obj
    ends = np.searchsorted(masses, masses+window)
    counts = ends - np.arange(len(ends))
    fraction = n_events/np.max(counts)
    return fraction
    
