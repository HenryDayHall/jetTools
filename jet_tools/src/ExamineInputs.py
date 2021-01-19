""" File to examine the nature of the input data """
from jet_tools.src import Components, FormShower
from matplotlib import pyplot as plt

def percent_b(eventWise, observables=True):
    assert eventWise.selected_index is not None
    if observables:
        total = len(eventWise.JetInputs_Energy)
        bs = len(eventWise.DetectableTag_Leaves.flatten())
    else:
        total = np.sum(eventWise.Is_Leaf)
        bs = len(FormShower.descendant_idxs(eventWise, *eventWise.BQuarkIdxs))
    return bs/total

def plot_event_counts(eventWise, observables=True, ax=None):
    if ax is None:
        ax = plt.gca()
