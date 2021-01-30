""" Script to generate a summary based on one eventWise file """
from jet_tools import FormJets, CompareClusters, Components, InputTools,\
                        PlottingTools, AreaMeasures, MassPeaks
from matplotlib import pyplot as plt
#from ipdb import set_trace as st
import awkward
import numpy as np
import scipy.stats

# Rapidity/Phi offsets ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def plot_rapidity_phi_offset(jet_name, eventWise,
                             rapidity_in, phi_in, tag_rapidity_in, tag_phi_in,
                             ax_arr=None):
    if ax_arr is None:
        fig, ax_arr = plt.subplots(1, 3)
        fig.suptitle(jet_name)
    colour = eventWise.DetectableTag_Mass.flatten()
    alpha=0.2
    x_range = (-3.5, 3.5)
    y_range = (0, np.pi+0.1)
    # zeroth axis has the full jet kinematics
    ax = ax_arr[0]
    ax.set_title("Entire jets")
    rapidity_distance = (awkward.fromiter(rapidity_in)
                         - eventWise.DetectableTag_Rapidity).flatten()
    phi_distance = (awkward.fromiter(phi_in) - eventWise.DetectableTag_Phi).flatten()
    phi_distance = Components.raw_to_angular_distance(phi_distance)
    # print the mean distance
    ax.scatter([0], [0], s=5, c='k', marker='o')
    mean_distance = np.nanmean(np.sqrt(rapidity_distance**2 + phi_distance**2))
    # put a dot at the origin
    ax.text(0, y_range[1]*0.75, f"mean distance={mean_distance:.2f}",
            horizontalalignment='center')
    points = ax.scatter(rapidity_distance, phi_distance, c=colour, alpha=alpha)
    ax.set_xlabel("Jet rapidity offset")
    ax.set_ylabel("Jet $|\\phi|$ offset")
    ax.set_ylim(y_range)
    ax.set_xlim(x_range)
    # add a colourbar
    cbar = plt.colorbar(points, ax=ax)
    cbar.set_label("True mass of detectable signal")
    # the first axis has only the signal content of the jet
    ax = ax_arr[1]
    ax.set_title("Signal content of jets")
    tag_rapidity_distance = (awkward.fromiter(tag_rapidity_in)
                             - eventWise.DetectableTag_Rapidity).flatten()
    tag_phi_distance = (awkward.fromiter(tag_phi_in) - eventWise.DetectableTag_Phi).flatten()
    tag_phi_distance = Components.raw_to_angular_distance(tag_phi_distance)
    # print the mean distance
    tag_mean_distance = np.nanmean(np.sqrt(tag_rapidity_distance**2 + tag_phi_distance**2))
    ax.text(0, y_range[1]*0.75, f"mean distance={tag_mean_distance:.2f}",
            horizontalalignment='center')
    # put a dot at the origin
    ax.scatter([0], [0], s=5, c='k', marker='o')
    points = ax.scatter(tag_rapidity_distance, tag_phi_distance, c=colour, alpha=alpha)
    ax.set_xlabel("Jet rapidity offset")
    ax.set_ylabel("Jet $|\\phi|$ offset")
    ax.set_ylim(y_range)
    ax.set_xlim(x_range)
    # add a colourbar
    cbar = plt.colorbar(points, ax=ax)
    cbar.set_label("True mass of detectable signal")
    if len(ax_arr) > 2:
        # then in the last axis discribe the jet
        PlottingTools.discribe_jet(eventWise, jet_name, ax=ax_arr[2])
        fig.subplots_adjust(top=0.88,
                            bottom=0.11,
                            left=0.05,
                            right=1.0,
                            hspace=0.2,
                            wspace=0.255)
        fig.set_size_inches(12, 4.)
    return mean_distance, tag_mean_distance

# PT change ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def plot_PT(jet_name, eventWise, pt_in, tag_pt_in, ax_arr=None):
    if ax_arr is None:
        fig, ax_arr = plt.subplots(1, 3)
        fig.suptitle(jet_name)
    # flatten both
    pt_in = awkward.fromiter(pt_in).flatten()
    tag_pt_in = awkward.fromiter(tag_pt_in).flatten()
    pt_detectable = eventWise.DetectableTag_PT.flatten()
    # set up
    colour = eventWise.DetectableTag_Mass.flatten()
    alpha=0.2
    x_range = (0, np.nanmax(pt_detectable))
    y_range = (0, np.nanmax(pt_in))
    # zeroth axis has the full jet kinematics
    ax = ax_arr[0]
    ax.set_title("Entire jets")
    # put a diagonal line in
    ax.plot([0, max(x_range[1], y_range[1])], [0, max(x_range[1], y_range[1])],
            ls='--', alpha=0.5)
    # print the mean distance
    mean_distance = np.nanmean(np.abs(pt_in - pt_detectable))
    ax.text(x_range[1]*0.75, y_range[1]*0.75, f"mean distance={mean_distance:.2f}",
            horizontalalignment='center')
    points = ax.scatter(pt_detectable, pt_in, c=colour, alpha=alpha)
    ax.set_xlabel("Detectable signal PT")
    ax.set_xlabel("Jet PT")
    ax.set_ylim(y_range)
    ax.set_xlim(x_range)
    # add a colourbar
    cbar = plt.colorbar(points, ax=ax)
    cbar.set_label("True mass of detectable signal")
    # the first axis has only the signal content of the jet
    ax = ax_arr[1]
    ax.set_title("Signal content of jets")
    # put a diagonal line in
    ax.plot([0, max(x_range[1], y_range[1])], [0, max(x_range[1], y_range[1])],
            ls='--', alpha=0.5)
    tag_mean_distance = np.nanmean(np.abs(tag_pt_in - pt_detectable))
    ax.text(x_range[1]*0.75, y_range[1]*0.75, f"mean distance={tag_mean_distance:.2f}",
            horizontalalignment='center')
    points = ax.scatter(pt_detectable, tag_pt_in, c=colour, alpha=alpha)
    ax.set_xlabel("Detectable signal PT")
    ax.set_xlabel("Jet PT")
    ax.set_ylim(y_range)
    ax.set_xlim(x_range)
    # add a colourbar
    cbar = plt.colorbar(points, ax=ax)
    cbar.set_label("True mass of detectable signal")
    if len(ax_arr) > 2:
        # then in the last axis discribe the jet
        PlottingTools.discribe_jet(eventWise, jet_name, ax=ax_arr[2])
        fig.subplots_adjust(top=0.88,
                            bottom=0.11,
                            left=0.05,
                            right=1.0,
                            hspace=0.2,
                            wspace=0.255)
        fig.set_size_inches(12, 4.)
    return mean_distance, tag_mean_distance

# Jet width ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def plot_Width(jet_name, eventWise, ax_arr=None):
    if ax_arr is None:
        fig, ax_arr = plt.subplots(1, 3)
        fig.suptitle(jet_name)
    # set up
    # zeroth axis has the full jet kinematics
    ax0 = ax_arr[0]
    ax0.set_title("Entire jets")
    jet_widths, shower_widths, _ = AreaMeasures.plot_jet_v_shower(
                                   eventWise, jet_name, colour='PT',
                                   ax=ax0, signal_only=False)
    correlation, p_value = scipy.stats.spearmanr(jet_widths, shower_widths)
    spearmans_rank =  p_value
    # the first axis has only the signal content of the jet
    ax1 = ax_arr[1]
    ax1.set_title("Signal content of jets")
    jet_widths, shower_widths, _ = AreaMeasures.plot_jet_v_shower(
                                   eventWise, jet_name, colour='PT',
                                   ax=ax1, signal_only=True)
    correlation, p_value = scipy.stats.spearmanr(jet_widths, shower_widths)
    tag_spearmans_rank = p_value

    x_range = (0, max(ax0.get_xlim()[1], ax1.get_xlim()[1]))
    y_range = (0, max(ax0.get_ylim()[1], ax1.get_ylim()[1]))
    ax0.set_ylim(y_range)
    ax0.set_xlim(x_range)
    ax1.set_ylim(y_range)
    ax1.set_xlim(x_range)
    if len(ax_arr) > 2:
        # then in the last axis discribe the jet
        PlottingTools.discribe_jet(eventWise, jet_name, ax=ax_arr[2])
        fig.subplots_adjust(top=0.88,
                            bottom=0.11,
                            left=0.05,
                            right=1.0,
                            hspace=0.2,
                            wspace=0.255)
        fig.set_size_inches(12, 4.)
    return spearmans_rank, tag_spearmans_rank

# Jet multiplicity ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def plot_Multiplicity(jet_name, eventWise, ax_arr=None):
    if ax_arr is None:
        fig, ax_arr = plt.subplots(1, 2)
        fig.suptitle(jet_name)
    tag_multiplicity = [len(event.flatten()) for event in eventWise.DetectableTag_Roots]
    jet_multiplicity = getattr(eventWise, jet_name + "_SeperateJets").tolist()
    hist_args = dict(bins=6, range=(-0.5, 5.5))
    # plot the tag multiplicity underneat
    ax = ax_arr[0]
    ax.hist(tag_multiplicity, color='grey', alpha=0.5, label="Detectable signals",
            **hist_args)
    # then plot the jet multiplicity
    ax.hist(jet_multiplicity, color='red', histtype='step', label="Jet multiplicity",
            **hist_args)
    ax.legend()
    ax.set_xlabel("Seperate jets/detectable signals")
    ax.set_ylabel(f"Raw counts in {len(jet_multiplicity)} events")
    if len(ax_arr) > 1:
        # then in the last axis discribe the jet
        PlottingTools.discribe_jet(eventWise, jet_name, ax=ax_arr[1])
        fig.subplots_adjust(top=0.88,
                            bottom=0.11,
                            left=0.085,
                            right=1.0,
                            hspace=0.2,
                            wspace=0.255)
        fig.set_size_inches(9, 4.)
    return 1/np.nanmean(jet_multiplicity)

# Mass peaks ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def plot_MassPeaks(jet_name, eventWise, ax_arr=None):
    if ax_arr is None:
        fig, ax_arr = plt.subplots(3, 3)
        fig.suptitle(jet_name)
    MassPeaks.plot_correct_pairs(eventWise, [jet_name], show=False, plot_type='hist', signal_background='both', ax_array=ax_arr[0])
    MassPeaks.plot_correct_pairs(eventWise, [jet_name], show=False, plot_type='hist', signal_background='signal', ax_array=ax_arr[1])
    if len(ax_arr) > 2:
        PlottingTools.discribe_jet(eventWise, jet_name, font_size=10, ax=ax_arr[2, 1])
        PlottingTools.hide_axis(ax_arr[2, 0])
        PlottingTools.hide_axis(ax_arr[2, 2])
        fig.subplots_adjust(top=0.93,
                            bottom=0.005,
                            left=0.060,
                            right=0.97,
                            hspace=0.35,
                            wspace=0.255)
        fig.set_size_inches(12, 9)
    

# Summary table ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def plot_Summary(jet_names, criteria_names, values):
    fig, ax = plt.subplots()
    values /= np.mean(values, axis=0)
    values -= np.min(values)
    im = ax.imshow(values)
    ax.set_xticks(np.arange(len(criteria_names)))
    ax.set_yticks(np.arange(len(jet_names)))
    ax.set_xticklabels(criteria_names, rotation=45)
    ax.set_yticklabels(jet_names)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                     rotation_mode="anchor")
    # print the numbers
    for row in range(len(jet_names)):
        for col in range(len(criteria_names)):
            text = ax.text(col, row, f"{values[row, col]:.1f}",
                           ha='center', va='center', color='w')
    ax.set_title("Low values are best")
    fig.tight_layout()
    fig.set_size_inches(8, 9)


# start ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def plot_one(eventWise, jet_name=None):
    if isinstance(eventWise, str):
        eventWise = Components.EventWise.from_file(eventWise)
    if jet_name is None:
        jet_names = FormJets.get_jet_names(eventWise)
        jet_name = InputTools.list_complete("Chose a jet; ", jet_names).strip()
    jet_idxs = FormJets.filter_jets(eventWise, jet_name)
    [tag_mass2_in, all_mass2_in, bg_mass2_in, rapidity_in, phi_in, pt_in, mask,
        tag_rapidity_in, tag_phi_in, tag_pt_in, percent_found,
     seperate_jets] = CompareClusters.per_event_detectables(eventWise, jet_name, jet_idxs)
    plot_rapidity_phi_offset(jet_name, eventWise, rapidity_in, phi_in, tag_rapidity_in, tag_phi_in)
    plt.show()
    input()
    plot_PT(jet_name, eventWise, pt_in, tag_pt_in)
    plt.show()
    input()
    plot_Width(jet_name, eventWise)
    plt.show()
    input()
    plot_Multiplicity(jet_name, eventWise)
    plt.show()
    input()
    plot_MassPeaks(jet_name, eventWise)
    plt.show()
    input()


def plot_all(eventWise, prefix, jet_names=None):
    if isinstance(eventWise, str):
        eventWise = Components.EventWise.from_file(eventWise)
    if jet_names is None:
        jet_names = FormJets.get_jet_names(eventWise)
    criteria_names = ["$\\Delta R$ offset", "BG free $\\Delta R$ offset",
                      "$p_T$ offset", "BG free $p_T$ offset",
                      "width correlation", "BG free width correlation",
                      "1/average multiplicity",
                      "quality width", "quality fraction"]
    values = np.empty((len(jet_names), len(criteria_names)), dtype=float)

    for i, jet_name in enumerate(jet_names):
        print(f"{i/len(jet_names):.1%} {jet_name}")
        jet_idxs = FormJets.filter_jets(eventWise, jet_name)
        [tag_mass2_in, all_mass2_in, bg_mass2_in, rapidity_in, phi_in, pt_in, mask,
            tag_rapidity_in, tag_phi_in, tag_pt_in, percent_found,
         seperate_jets] = CompareClusters.per_event_detectables(eventWise, jet_name, jet_idxs)
        dr_off, sig_dr_off = plot_rapidity_phi_offset(jet_name, eventWise, rapidity_in, phi_in, tag_rapidity_in, tag_phi_in)
        values[i, criteria_names.index("$\\Delta R$ offset")] = dr_off
        values[i, criteria_names.index("BG free $\\Delta R$ offset")] = sig_dr_off
        plt.savefig(prefix + "_" + jet_name + "_deltaRoffset.png")
        plt.close()
        pt_off, sig_pt_off = plot_PT(jet_name, eventWise, pt_in, tag_pt_in)
        values[i, criteria_names.index("$p_T$ offset")] = pt_off
        values[i, criteria_names.index("BG free $p_T$ offset")] = sig_pt_off
        plt.savefig(prefix + "_" + jet_name + "_PToffset.png")
        plt.close()
        width, sig_width = plot_Width(jet_name, eventWise)
        values[i, criteria_names.index("width correlation")] = width
        values[i, criteria_names.index("BG free width correlation")] = sig_width
        plt.savefig(prefix + "_" + jet_name + "_Width.png")
        plt.close()
        mul = plot_Multiplicity(jet_name, eventWise)
        values[i, criteria_names.index("1/average multiplicity")] = mul
        plt.savefig(prefix + "_" + jet_name + "_Multiplicity.png")
        plt.close()
        plot_MassPeaks(jet_name, eventWise)
        plt.savefig(prefix + "_" + jet_name + "_MassPeaks.png")
        plt.close()
        # then jet quality measures
        values[i, criteria_names.index("quality width")] = getattr(eventWise,
                                                                   jet_name + "_QualityWidth")
        values[i, criteria_names.index("quality fraction")] = getattr(eventWise,
                                                                   jet_name + "_QualityFraction")
    # now plot and save the summary name
    plot_Summary(jet_names, criteria_names, values)
    plt.savefig(prefix + "_Summary.png")
    #st()
    plot_Summary(jet_names, criteria_names, values)
    plt.show()
    input()


if __name__ == '__main__':
    eventWise = "megaIgnore/best_v2.awkd"
    plot_all(eventWise, "images/summary/best_v2")
