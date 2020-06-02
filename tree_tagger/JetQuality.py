""" Jet quality measures as discribed in https://arxiv.org/pdf/0810.1304.pdf"""
import numpy as np
from tree_tagger import MassPeaks, Constants
import scipy.optimize
from ipdb import set_trace as st


def sorted_masses(eventWise, jet_name, mass_function='highest pt pair',
                  jet_pt_cut=None, max_tag_angle=None):
    if max_angle is None:
        max_angle = Constants.max_tagangle
    if jet_pt_cut is None:
        jet_pt_cut = Constants.min_jetpt
    if mass_function == 'highest pt pair':
        all_masses, pairs, pair_masses = MassPeaks.all_PT_pairs(eventWise, jet_name, jet_pt_cut, max_tag_angle=max_tag_angle)
        idx = next(i for i, p in enumerate(pairs) if set(p) == {0, 1})
        masses = pair_masses[idx]
    else:
        masses = mass_function(eventWise, jet_name, jet_pt_cut)
    masses = np.sort(masses)
    return masses


def quality_width(eventWise, jet_name, fraction=0.15, mass_function='highest pt pair',
                  jet_pt_cut=None, max_tag_angle=None):
    eventWise.selected_index = None
    n_events = len(getattr(eventWise, jet_name+'_InputIdx'))
    target_counts = int(np.ceil(fraction*n_events))
    masses = sorted_masses(eventWise, jet_name, mass_function, jet_pt_cut, max_tag_angle=None)
    if target_counts > len(masses):
        msg = f"Cannot acheve a fraction of {fraction} with {len(masses)} masses from {n_events} events"
        raise RuntimeError(msg)
    widths = masses[target_counts:] - masses[:-target_counts]
    best_width = np.min(widths)
    return best_width


def quality_fraction(eventWise, jet_name, mass_of_obj, multiplier=125., mass_function='highest pt pair',
                     jet_pt_cut=None, max_tag_angle=None):
    eventWise.selected_index = None
    n_events = len(getattr(eventWise, jet_name+'_InputIdx'))
    masses = sorted_masses(eventWise, jet_name, mass_function, jet_pt_cut, max_tag_angle=max_tag_angle)
    if len(masses) == 0:
        msg = f"No masses from {n_events} events"
        raise RuntimeError(msg)
    window = multiplier*mass_of_obj
    ends = np.searchsorted(masses, masses+window)
    counts = ends - np.arange(len(ends))
    fraction = n_events/np.max(counts)
    return fraction


def quality_width_fracton(eventWise, jet_name, mass_of_obj, fraction=0.15, multiplier=125.,
                          mass_function='highest pt pair', jet_pt_cut=None, max_tag_angle=None):
    """ slightly faster to do both together """
    if max_angle is None:
        max_angle = Constants.max_tagangle
    eventWise.selected_index = None
    n_events = len(getattr(eventWise, jet_name+'_InputIdx'))
    masses = sorted_masses(eventWise, jet_name, mass_function, jet_pt_cut, max_tag_angle=max_tag_angle)
    if len(masses) == 0:
        msg = f"No masses from {n_events} events"
        raise RuntimeError(msg)
    # width
    target_counts = int(np.ceil(fraction*n_events))
    if target_counts > len(masses):
        target_counts = len(masses)
    widths = masses[target_counts-1:] - masses[:1-target_counts]
    best_width = np.min(widths)
    # fraction
    window = multiplier*mass_of_obj
    ends = np.searchsorted(masses, masses+window)
    counts = ends - np.arange(len(ends))
    try:
        fraction = n_events/np.max(counts)
    except ZeroDivisionError:
        fraction = n_events
    return best_width, fraction


