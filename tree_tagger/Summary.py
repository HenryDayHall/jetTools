""" Script to generate a summary based on one eventWise file """
from tree_tagger import FormJets, CompareClusters, Components, InputTools, PlottingTools
from matplotlib import pyplot as plt
import awkward
import numpy as np

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
    # put a dot at the origin
    ax.scatter([0], [0], s=5, c='k', marker='o')
    points = ax.scatter(rapidity_distance, phi_distance, c=colour, alpha=alpha)
    ax.set_xlabel("Jet rapidity offset")
    ax.set_ylabel("Jet $|\\phi|$ offset")
    ax.set_ylim(y_range)
    ax.set_xlim(x_range)
    # add a colourbar
    cbar = plt.colorbar(points, ax=ax)
    cbar.set_label("True mass of signal object")
    # the first axis has only the signal content of the jet
    ax = ax_arr[1]
    ax.set_title("Signal content of jets")
    tag_rapidity_distance = (awkward.fromiter(tag_rapidity_in)
                             - eventWise.DetectableTag_Rapidity).flatten()
    tag_phi_distance = (awkward.fromiter(tag_phi_in) - eventWise.DetectableTag_Phi).flatten()
    tag_phi_distance = Components.raw_to_angular_distance(tag_phi_distance)
    # put a dot at the origin
    ax.scatter([0], [0], s=5, c='k', marker='o')
    points = ax.scatter(tag_rapidity_distance, tag_phi_distance, c=colour, alpha=alpha)
    ax.set_xlabel("Jet rapidity offset")
    ax.set_ylabel("Jet $|\\phi|$ offset")
    ax.set_ylim(y_range)
    ax.set_xlim(x_range)
    # add a colourbar
    cbar = plt.colorbar(points, ax=ax)
    cbar.set_label("True mass of signal object")
    if len(ax_arr) > 2:
        # then in the last axis discribe the jet
        PlottingTools.discribe_jet(eventWise, jet_name, ax=ax_arr[2])

# PT change ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Jet width ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Jet multiplicity ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Mass peaks ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Summary table ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# start ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def main(eventWise, jet_name=None):
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



if __name__ == '__main__':
    eventWise = "megaIgnore/best_v2.awkd"
    main(eventWise)
