from tree_tagger import Components, PDGNames, InputTools
from ipdb import set_trace as st
import collections
from matplotlib import pyplot as plt
import numpy as np
import awkward
import scipy.spatial


def get_low_pt_mask(eventWise, jet_name=None, low_pt=10.):
    # find the low PT area
    if jet_name is None:
        low_pt_mask = eventWise.PT[eventWise.Is_leaf] < low_pt
        jet_roots = None
    else:
        jet_roots = getattr(eventWise, jet_name + "_Parent") == -1
        low_pt_mask = getattr(eventWise, jet_name + "_PT")[jet_roots] < low_pt
    return low_pt_mask, jet_roots


def append_flat_IRC_variables(eventWise, jet_name=None, low_pt=10.):
    if jet_name is None:
        jet_str = ""
    else:
        jet_str = jet_name
    leaf_variables = ["PT", "Rapidity"]
    low_pt_mask, jet_roots = get_low_pt_mask(eventWise, jet_name, low_pt)
    # kinamtic
    new_content = {}
    for var in leaf_variables:
        if jet_name is None:
            values = getattr(eventWise, var)[eventWise.Is_leaf].flatten()
        else:
            values = getattr(eventWise, jet_name + "_" + var)[jet_roots].flatten()
        if var == "PT":
            var = "logPT"
            values = np.log(values[values > 0])
            low_pt_values = values[values < np.log(low_pt)]
            values = values.tolist()
        else:
            low_pt_values = values[low_pt_mask.flatten()].tolist()
        new_content[jet_str + "IRC_" +var] = values
        new_content[jet_str + "IRCLowPT_" +var] = low_pt_values
    eventWise.append(**new_content)


def awkward_to_2d(array, depth=1):
    for _ in range(depth):
        array = array.flatten()
    return np.array(array.tolist()).reshape((-1, 1))

def append_pairwise_IRC_variables(eventWise, jet_name=None, low_pt=10.):
    if jet_name is None:
        jet_str = ""
    else:
        jet_str = jet_name
    new_content = {}
    # find the low PT area
    low_pt_mask, jet_roots = get_low_pt_mask(eventWise, jet_name, low_pt)
    # PT
    if jet_name is None:
        values = [scipy.spatial.distance.pdist(event.reshape(-1, 1)) for event in
                  eventWise.PT[eventWise.Is_leaf]]
    else:
        values = [scipy.spatial.distance.pdist(awkward_to_2d(event)) for event in
                  getattr(eventWise, jet_name + "_PT")[jet_roots]]
    values = awkward.fromiter(values).flatten().tolist()
    if jet_name is None:
        low_pt_values = [scipy.spatial.distance.pdist(event.reshape(-1, 1)) for event in
                         eventWise.PT[eventWise.Is_leaf][low_pt_mask]]
    else:
        low_pt_values = [scipy.spatial.distance.pdist(awkward_to_2d(event)) for event in
                         getattr(eventWise, jet_name + "_PT")[jet_roots][low_pt_mask]]
    low_pt_values = awkward.fromiter(low_pt_values).flatten().tolist()
    new_content[jet_str + "IRCPariwise_PT"] = values
    new_content[jet_str + "IRCPariwiseLowPT_PT"] = low_pt_values
    # delta R
    if jet_name is None:
        rapidity = [scipy.spatial.distance.pdist(event.reshape(-1, 1)) for event in
                    eventWise.Rapidity[eventWise.Is_leaf]]
    else:
        rapidity = [scipy.spatial.distance.pdist(awkward_to_2d(event)) for event in
                    getattr(eventWise, jet_name + "_Rapidity")[jet_roots]]
    rapidity = awkward.fromiter(rapidity)
    if jet_name is None:
        low_pt_rapidity = [scipy.spatial.distance.pdist(event.reshape(-1, 1)) for event in
                           eventWise.Rapidity[eventWise.Is_leaf][low_pt_mask]]
    else:
        low_pt_rapidity = [scipy.spatial.distance.pdist(awkward_to_2d(event)) for event in
                           getattr(eventWise, jet_name + "_Rapidity")[jet_roots][low_pt_mask]]
    low_pt_rapidity = awkward.fromiter(low_pt_rapidity)
    if jet_name is None:
        phi = [scipy.spatial.distance.pdist(awkward_to_2d(event), metric=Components.angular_distance) for event in
               eventWise.Phi[eventWise.Is_leaf]]
    else:
        phi = [scipy.spatial.distance.pdist(awkward_to_2d(event), metric=Components.angular_distance) for event in
               getattr(eventWise, jet_name + "_Phi")[jet_roots]]
    phi = awkward.fromiter(phi)
    if jet_name is None:
        low_pt_phi = [scipy.spatial.distance.pdist(awkward_to_2d(event), metric=Components.angular_distance) for event in
                      eventWise.Phi[eventWise.Is_leaf][low_pt_mask] if len(event)]
    else:
        low_pt_phi = [scipy.spatial.distance.pdist(awkward_to_2d(event), metric=Components.angular_distance) for event in
                      getattr(eventWise, jet_name + "_Phi")[jet_roots][low_pt_mask]]
    low_pt_phi = awkward.fromiter(low_pt_phi)
    values = np.sqrt(awkward.fromiter(rapidity)**2 + awkward.fromiter(phi)**2)
    low_pt_values = np.sqrt(awkward.fromiter(low_pt_rapidity)**2 + awkward.fromiter(low_pt_phi)**2)
    low_pt_values = low_pt_values.flatten().tolist()
    values = values.flatten().tolist()
    new_content[jet_str + "IRCPariwise_DeltaR"] = values
    new_content[jet_str + "IRCPariwiseLowPT_DeltaR"] = low_pt_values
    eventWise.append(**new_content)



def plot_hist(variable_name, names, low_pt, values, low_pt_values, ax1, ax2):
    ax1.hist(values, histtype='step', label=names)
    ax1.set_xlabel(variable_name)
    ax1.set_ylabel("Frequency")
    ax2.hist(low_pt_values, histtype='step', label=names)
    ax2.set_xlabel(variable_name)
    ax2.set_ylabel(f"PT < {low_pt} Frequency")



# identical ordering
def plot_ordered_comparison(name, name1, name2, vals1, vals2, ax, cbar=True):
    order = np.arange(len(vals1))
    # must shuffle to avoid effects arising from plot order
    np.random.shuffle(order)
    points = ax.scatter(vals1[order], vals2[order], alpha=0.5, c=order)
    if cbar:
        cbar = plt.colorbar(points, ax=ax, label="Event no.")
    ax.set_xlabel(f"{name} in dataset {name1}")
    ax.set_ylabel(f"{name} in dataset {name2}")


def ordered_counts_comparison(eventWise1, eventWise2, ax=None, cbar=False):
    if ax is None:
        cbar = True
        ax = plt.gca()
    pt1 = np.fromiter((np.sum(leaves) for leaves in eventWise1.Is_leaf), dtype=float)
    pt2 = np.fromiter((np.sum(leaves) for leaves in eventWise2.Is_leaf), dtype=float)
    plot_comparison("counts", eventWise1.save_name[:-5], eventWise2.save_name[:-5], pt1, pt2, ax, cbar)


def ordered_pt_comparison(eventWise1, eventWise2, ax=None, cbar=False):
    if ax is None:
        cbar = True
        ax = plt.gca()
    pt1 = np.fromiter((np.mean(evt[leaves]) for evt, leaves in zip(eventWise1.PT, eventWise1.Is_leaf)), dtype=float)
    pt2 = np.fromiter((np.mean(evt[leaves]) for evt, leaves in zip(eventWise2.PT, eventWise2.Is_leaf)), dtype=float)
    plot_comparison("PT", eventWise1.save_name[:-5], eventWise2.save_name[:-5], pt1, pt2, ax, cbar)


def ordered_rapidity_comparison(eventWise1, eventWise2, ax=None, cbar=False):
    if ax is None:
        cbar = True
        ax = plt.gca()
    rap1 = np.fromiter((np.mean(evt[leaves]) for evt, leaves in zip(eventWise1.Rapidity, eventWise1.Is_leaf)), dtype=float)
    rap2 = np.fromiter((np.mean(evt[leaves]) for evt, leaves in zip(eventWise2.Rapidity, eventWise2.Is_leaf)), dtype=float)
    plot_comparison("Rapidity", eventWise1.save_name[:-5], eventWise2.save_name[:-5], rap1, rap2, ax, cbar)


def ordered_pid_comparison(eventWise1, eventWise2, ax=None, cbar=False):
    if ax is None:
        cbar = True
        ax = plt.gca()
    flat_pids1 = eventWise1.MCPID.flatten()
    flat_pids2 = eventWise2.MCPID.flatten()
    all_mcpids = sorted(set(flat_pids1).union(flat_pids2))
    # we only wish to consider th emost common particles
    counts = collections.Counter(flat_pids1)
    counts.update(flat_pids2)
    num_to_plot = 7
    to_plot, _ = zip(*counts.most_common(num_to_plot))
    converter = PDGNames.IDConverter()
    names = [converter[i] for i in to_plot]
    changed = [(event_n, x, np.sum(pids1 == i) - np.sum(pids2 == i))
               for event_n, (pids1, pids2) in enumerate(zip(eventWise1.MCPID, eventWise2.MCPID))
               for x, i in enumerate(to_plot)]
    np.random.shuffle(changed)
    event_ns, xs, ys = zip(*changed)
    points = ax.scatter(xs, ys, c=event_ns)
    ax.set_xticks(range(num_to_plot))
    ax.set_xticklabels(names, rotation=90)
    if cbar:
        cbar = plt.colorbar(points, ax=ax, label="Event no.")
    ax.set_ylabel("Change in counts")
    

def plot_ordered_grid(eventWise1, eventWise2):
    fig, axs = plt.subplots(2, 2)
    counts_comparison(eventWise1, eventWise2, axs[0, 0])
    pt_comparison(eventWise1, eventWise2, axs[0, 1])
    rapidity_comparison(eventWise1, eventWise2, axs[1, 0])
    pid_comparison(eventWise1, eventWise2, axs[1, 1], cbar=True)
    fig.set_size_inches(9, 8)
    fig.tight_layout()

if __name__ == '__main__':
    eventWises = []
    while True:
        name = InputTools.get_file_name(f"Eventwise {len(eventWises)+1} to compare; ", '.awkd').strip()
        if name:
            eventWises.append(Components.EventWise.from_file(name))
        else:
            break
    append_pairwise_IRC_variables(eventWises[0], "AntiKTp8")
    #plot_hists(eventWises)
    #plt.show()
    #input()

