""" File to examine the nature of the input data """
import awkward
from jet_tools.src import Components, FormShower
from matplotlib import pyplot as plt
import numpy as np
from ipdb import set_trace as st

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





