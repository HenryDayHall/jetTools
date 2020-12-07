""" Jet quality measures as discribed in https://arxiv.org/pdf/0810.1304.pdf"""
import numpy as np
from jet_tools.tree_tagger import MassPeaks, Constants, TrueTag
import scipy.optimize
from ipdb import set_trace as st


def sorted_masses(eventWise, jet_name, mass_function='correct allocation',
                  jet_pt_cut=None, max_tag_angle=None):
    """
    Get mass predictions for this jet.

    Parameters
    ----------
    eventWise : EventWise
        dataset containing the jets
    jet_name : str
        prefix of the jet's variables in the eventWise
    mass_function : str or callable
        function that predicts a particle masses from the events
         (Default value = 'highest pt pair')
    jet_pt_cut : float
        required minimum jet PT for the jet to be selected
        if None the value s taken from Constants.py
        (Default = None)
    max_tag_angle : float
        The maximum deltaR betweeen a tag and its jet
         (Default value = 0.8)

    Returns
    -------
    masses : numpy array of floats
        the sorted list of particle mass predictions

    """
    if mass_function == 'highest pt pair':
        all_masses, pairs, pair_masses = MassPeaks.all_PT_pairs(eventWise, jet_name, jet_pt_cut, max_tag_angle=max_tag_angle)
        idx = next(i for i, p in enumerate(pairs) if set(p) == {0, 1})
        masses = pair_masses[idx]
    elif mass_function == 'correct allocation':
        _, masses, _ = MassPeaks.all_h_combinations(eventWise, jet_name)
        masses = np.array(masses)
        # zero masses should be dropped as unreconstructed
        masses = masses[masses > 0.00001]
    else:
        masses = mass_function(eventWise, jet_name, jet_pt_cut)
    masses = np.sort(masses)
    return masses


def quality_width(eventWise, jet_name, fraction=0.15, mass_function='correct allocation',
                  jet_pt_cut=None, max_tag_angle=None):
    """
    The width of the smallest reconstruced mass windows that contains
    the specifed fraction of generated massive objects.
    This is divided by total generated objects.
    See page 5 of 0810.1304

    Parameters
    ----------
    eventWise : EventWise
        dataset containing the jets
    jet_name : str
        prefix of the jet's variables in the eventWise
    fraction : float
        number between 0 and 1 to specify number of mass
        reconstruction that should fall in window
         (Default value = 0.15)
    mass_function : str or callable
        function that predicts a particle masses from the events
         (Default value = 'highest pt pair')
    jet_pt_cut : float
        required minimum jet PT for the jet to be selected
        if None the value s taken from Constants.py
        (Default = None)
    max_tag_angle : float
        The maximum deltaR betweeen a tag and its jet
         (Default value = 0.8)

    Returns
    -------
    best_width : float
        the width of the required window

    """
    eventWise.selected_index = None
    n_events = len(getattr(eventWise, jet_name+'_InputIdx'))
    target_counts = int(np.ceil(fraction*n_events))
    masses = sorted_masses(eventWise, jet_name, mass_function, jet_pt_cut, max_tag_angle=None)
    if len(masses) < 2:
        msg = f"Not enough masses from {n_events} events"
        raise RuntimeError(msg)
    if target_counts < 2:
        msg = f"Number of masses is {len(masses)}, and this is insufficient for a window of {fraction}"
        raise RuntimeError(msg)
    if target_counts > len(masses):
        msg = f"Cannot acheve a fraction of {fraction} with {len(masses)} masses from {n_events} events"
        raise RuntimeError(msg)
    widths = masses[target_counts-1:] - masses[:1-target_counts]
    best_width = np.min(widths)/n_events
    return best_width


def quality_fraction(eventWise, jet_name, mass_of_obj, multiplier=125.,
                     mass_function='correct allocation',
                     jet_pt_cut=None, max_tag_angle=0.8):
    """
    A window proportional to the root of the mass of the object to be reconstructed
    is slid across the data. The maximum fraction of masses captured by the
    window divided by total generated objects is calculated and its inverse is returned.
    See page 6 of 0810.1304

    Parameters
    ----------
    eventWise : EventWise
        dataset containing the jets
    jet_name : str
        prefix of the jet's variables in the eventWise
    mass_of_obj : float
        The mass of the object to be reconstructed
    multiplier : float
        Value in sqrt(GeV) to multiply the root of the mass be to
        find th ewidth of the window to be used
         (Default value = 125.)
    mass_function : str or callable
        function that predicts a particle masses from the events
         (Default value = 'highest pt pair')
    jet_pt_cut : float
        required minimum jet PT for the jet to be selected
        if None the value s taken from Constants.py
        (Default = None)
    max_tag_angle : float
        The maximum deltaR betweeen a tag and its jet
         (Default value = 0.8)

    Returns
    -------
    fraction : float
        maximum fraction of events found in window

    """
    eventWise.selected_index = None
    n_events = len(getattr(eventWise, jet_name+'_InputIdx'))
    masses = sorted_masses(eventWise, jet_name, mass_function, jet_pt_cut, max_tag_angle=max_tag_angle)
    if len(masses) < 2:
        msg = f"Not enough masses from {n_events} events"
        raise RuntimeError(msg)
    window = multiplier*mass_of_obj
    ends = np.searchsorted(masses, masses+window)
    counts = ends - np.arange(len(ends))
    fraction = n_events/np.max(counts)
    return fraction


def quality_width_fracton(eventWise, jet_name, mass_of_obj, fraction=0.15, multiplier=125.,
                          mass_function='correct allocation', jet_pt_cut=None, max_tag_angle=None):
    """
    Equivalent to calling quality_width and quality_fraction.
    slightly faster to do both together - also don't raise exceptions
    for edge cases, just make a logical patch.
    The number of masses required by quality_width is confined between 2 and total masses.

    Parameters
    ----------
    eventWise : EventWise
        dataset containing the jets
    jet_name : str
        prefix of the jet's variables in the eventWise
    mass_of_obj : float
        The mass of the object to be reconstructed
    multiplier : float
        Value in sqrt(GeV) to multiply the root of the mass be to
        find th ewidth of the window to be used
         (Default value = 125.)
    mass_function : str or callable
        function that predicts a particle masses from the events
         (Default value = 'highest pt pair')
    jet_pt_cut : float
        required minimum jet PT for the jet to be selected
        if None the value s taken from Constants.py
        (Default = None)
    max_tag_angle : float
        The maximum deltaR betweeen a tag and its jet
         (Default value = 0.8)

    Returns
    -------
    best_width : float
        the width of the required window
    fraction : float
        maximum fraction of events found in window

    """
    if max_tag_angle is None:
        max_tag_angle = Constants.max_tagangle
    eventWise.selected_index = None
    n_events = len(getattr(eventWise, jet_name+'_InputIdx'))
    masses = sorted_masses(eventWise, jet_name, mass_function, jet_pt_cut,
                           max_tag_angle=max_tag_angle)
    if len(masses) < 2:
        msg = f"Not enough masses from {n_events} events"
        raise RuntimeError(msg)
    # width
    target_counts = int(np.ceil(fraction*n_events))
    # in thsi one autofix instead of raising exceptions
    target_counts = np.clip(target_counts, 2, len(masses))
    widths = masses[target_counts-1:] - masses[:1-target_counts]
    best_width = np.min(widths)/n_events
    # fraction
    window = multiplier*mass_of_obj
    ends = np.searchsorted(masses, masses+window)
    counts = ends - np.arange(len(ends))
    try:
        fraction = n_events/np.max(counts)
    except ZeroDivisionError:
        fraction = n_events
    return best_width, fraction


