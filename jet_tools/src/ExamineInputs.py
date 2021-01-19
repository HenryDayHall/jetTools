""" File to examine the nature of the input data """
from jet_tools.src import Components, FormShower
from matplotlib import pyplot as plt
import numpy as np

def get_signal_and_total(eventWise, observables=True):
    eventWise.selected_index = None
    if observables:
        vector_len = np.vectorize(len)
        total = vector_len(eventWise.JetInputs_Energy)
        vector_flat_len = np.vectorize(lambda x: len(x.flatten()))
        signal = vector_flat_len(eventWise.DetectableTag_Leaves)
    else:
        vector_sum = np.vectorize(np.sum)
        total = vector_sum(eventWise.Is_Leaf)
        signal = np.empty_like(total)
        for i, bidxs in enumerate(eventWise.BQuarkIdxs):
            eventWise.selected_index = i
            signal[i] = len(FormShower.descendant_idxs(eventWise, *bidxs))
    return signal, total


def plot_event_counts(eventWise, observables=True, ax_arr=None):
    if ax_arr is None:
        fig, ax_arr = plt.subplots(1, 2)
    ax1, ax2 = ax_arr
    signal, total = get_signal_and_total(eventWise, observables)
    percentages = signal/total
    bin_heights, bins, patches = ax1.hist(total, bins=40, histtype='stepfilled',
                                        density=True,
                                         color='grey', label="Frequency of events by size")
    if observables:
        ax.set_xlabel("Number of particles that pass cuts")
    else:
        ax.set_xlabel("Number of particles without cuts")
    # loop over the bins
    mean, lower_quartile, upper_quartile = [], [], []
    for bin_start, bin_end in zip(bins[:-1], bins[1:]):
        in_bin = np.logical_and(total > bin_start, total <= bin_end)
        if not np.any(in_bin):
            percentages_here = [np.nan]
        else:
            percentages_here = percentages[in_bin]
        mean.append(np.mean(percentages_here))
        lower_quartile.append(np.quantile(percentages_here, 0.25))
        upper_quartile.append(np.quantile(percentages_here, 0.75))
    mean = [mean[0]] + mean
    lower_quartile = [lower_quartile[0]] + lower_quartile
    upper_quartile = [upper_quartile[0]] + upper_quartile
    ax.fill_between(bins, lower_quartile, upper_quartile, color='g', alpha=0.3)
    ax.plot(bins, mean, color='g', label="Mean $b$-shower fraction by size")
    ax.set_ylabel("Fraction of particles in event from $b$-shower")





