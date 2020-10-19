from tree_tagger import Components, PDGNames, InputTools, FormJets
import multiprocessing
import time
import os
from ipdb import set_trace as st
import collections
from matplotlib import pyplot as plt
import numpy as np
import awkward
import scipy.spatial, scipy.stats


def calculate_ks_values(eventWises):
    jet_names = FormJets.get_jet_names(eventWise[0])
    ks_values = {}
    for name in jet_names:
        pass



def get_low_pt_mask(eventWise, jet_name=None, low_pt=10.):
    # find the low PT area
    if jet_name is None:
        low_pt_mask = eventWise.PT[eventWise.Is_leaf] < low_pt
        jet_roots = None
        root_pt = None
    else:
        jet_roots = getattr(eventWise, jet_name + "_Parent") == -1
        # no idea why this won't work
        #low_pt_mask = getattr(eventWise, jet_name + "_PT")[jet_roots] < low_pt
        jet_pt = getattr(eventWise, jet_name + "_PT")
        root_pt = awkward.fromiter([pt[root] for pt, root in zip(jet_pt, jet_roots)])
        low_pt_mask = root_pt < low_pt
    return low_pt_mask, jet_roots, root_pt


def append_flat_IRC_variables(eventWise, jet_name=None, low_pt=10., append=False):
    if jet_name is None:
        jet_str = ""
    else:
        jet_str = jet_name
    leaf_variables = ["PT", "Rapidity"]
    low_pt_mask, jet_roots, root_pt = get_low_pt_mask(eventWise, jet_name, low_pt)
    # kinamtic
    new_content = {}
    for var in leaf_variables:
        if jet_name is None:
            values = getattr(eventWise, var)[eventWise.Is_leaf].flatten()
        elif var == "PT":
            values = root_pt.flatten()
        else:
            values = awkward.fromiter([v[root] for v, root in
                                       zip(getattr(eventWise, jet_name+"_"+var), jet_roots)])
            values = values.flatten()
        if var == "PT":
            var = "logPT"
            values = np.log(values[values > 0])
            low_pt_values = values[values < np.log(low_pt)]
            values = values.tolist()
        else:
            low_pt_values = values[low_pt_mask.flatten()].tolist()
        new_content[jet_str + "IRC_" +var] = values
        new_content[jet_str + "IRCLowPT_" +var] = low_pt_values
    if append:
        eventWise.append(**new_content)
    else:
        return new_content


def awkward_to_2d(array, depth=1):
    for _ in range(depth):
        array = array.flatten()
    return np.array(array.tolist()).reshape((-1, 1))


def append_pairwise_IRC_variables(eventWise, jet_name=None, low_pt=10., append=False):
    if jet_name is None:
        jet_str = ""
    else:
        jet_str = jet_name
    new_content = {}
    # find the low PT area
    low_pt_mask, jet_roots, root_pt = get_low_pt_mask(eventWise, jet_name, low_pt)
    # PT
    if jet_name is None:
        values = [scipy.spatial.distance.pdist(event.reshape(-1, 1)) for event in
                  eventWise.PT[eventWise.Is_leaf]]
    else:
        values = [scipy.spatial.distance.pdist(awkward_to_2d(event)) for event in root_pt]
    values = awkward.fromiter(values).flatten().tolist()
    if jet_name is None:
        low_pt_values = [scipy.spatial.distance.pdist(event.reshape(-1, 1)) for event in
                         eventWise.PT[eventWise.Is_leaf][low_pt_mask]]
    else:
        low_pt_values = [scipy.spatial.distance.pdist(awkward_to_2d(event)) for event in
                         root_pt[low_pt_mask]]
    low_pt_values = awkward.fromiter(low_pt_values).flatten().tolist()
    new_content[jet_str + "IRCPariwise_PT"] = values
    new_content[jet_str + "IRCPariwiseLowPT_PT"] = low_pt_values
    # delta R
    if jet_name is None:
        rapidity = [scipy.spatial.distance.pdist(event.reshape(-1, 1)) for event in
                    eventWise.Rapidity[eventWise.Is_leaf]]
    else:
        jet_rapidity = getattr(eventWise, jet_name + "_Rapidity")
        rapidity = [scipy.spatial.distance.pdist(awkward_to_2d(rap[root])) for rap, root in
                    zip(jet_rapidity, jet_roots)]
    rapidity = awkward.fromiter(rapidity)
    if jet_name is None:
        low_pt_rapidity = [scipy.spatial.distance.pdist(event.reshape(-1, 1)) for event in
                           eventWise.Rapidity[eventWise.Is_leaf][low_pt_mask]]
    else:
        low_pt_rapidity = [scipy.spatial.distance.pdist(awkward_to_2d(rap[root][low]))
                           for rap, root, low in
                           zip(jet_rapidity, jet_roots, low_pt_mask)]
    low_pt_rapidity = awkward.fromiter(low_pt_rapidity)
    if jet_name is None:
        phi = [scipy.spatial.distance.pdist(awkward_to_2d(event), metric=Components.angular_distance)
               for event in
               eventWise.Phi[eventWise.Is_leaf]]
    else:
        jet_phi = getattr(eventWise, jet_name + "_Phi")
        phi = [scipy.spatial.distance.pdist(awkward_to_2d(ph[root]), metric=Components.angular_distance)
               for ph, root in
               zip(jet_phi, jet_roots)]
    phi = awkward.fromiter(phi)
    if jet_name is None:
        low_pt_phi = [scipy.spatial.distance.pdist(awkward_to_2d(event), metric=Components.angular_distance) for event in
                      eventWise.Phi[eventWise.Is_leaf][low_pt_mask] if len(event)]
    else:
        low_pt_phi = [scipy.spatial.distance.pdist(awkward_to_2d(ph[root][low]), metric=Components.angular_distance)
                      for ph, root, low in
                      zip(jet_phi, jet_roots, low_pt_mask)]
    low_pt_phi = awkward.fromiter(low_pt_phi)
    values = np.sqrt(awkward.fromiter(rapidity)**2 + awkward.fromiter(phi)**2)
    low_pt_values = np.sqrt(awkward.fromiter(low_pt_rapidity)**2 + awkward.fromiter(low_pt_phi)**2)
    low_pt_values = low_pt_values.flatten().tolist()
    values = values.flatten().tolist()
    new_content[jet_str + "IRCPariwise_DeltaR"] = values
    new_content[jet_str + "IRCPariwiseLowPT_DeltaR"] = low_pt_values
    if append:
        eventWise.append(**new_content)
    else:
        return new_content


def append_all(path, end_time, low_pt=10.):
    eventWise = Components.EventWise.from_file(path)
    new_content = {}
    jet_names = FormJets.get_jet_names(eventWise)
    n_jets = len(jet_names)
    start_columns = eventWise.columns
    for i, jet_name in enumerate(jet_names):
        # prevent overwites
        found = next((name for name in start_columns if name.startswith(jet_name + "IRC")),
                     False)
        if found:
            continue
        if (i+1) % 10 == 0:  # reload to preserve ram
            eventWise.append(**new_content)
            new_content = {}
            eventWise = Components.EventWise.from_file(path)
        if time.time() > end_time:
            break
        print(f'{i/n_jets:%}', end='\r', flush=True)
        new_c = append_pairwise_IRC_variables(eventWise, jet_name, low_pt)
        new_content.update(new_c)
        new_c = append_flat_IRC_variables(eventWise, jet_name, low_pt)
        new_content.update(new_c)
    eventWise.append(**new_content)
    print(f"\nDone {eventWise.save_name}\n", flush=True)


def multiprocess_append(eventWise_paths, end_time, leave_one_free=True):
    n_paths = len(eventWise_paths)
    # cap this out at 20, more seems to create a performance hit
    n_threads = np.min((multiprocessing.cpu_count()-leave_one_free, 20, n_paths))
    if n_threads < 1:
        n_threads = 1
    wait_time = 24*60*60 # in seconds
    # note that the longest wait will be n_cores time this time
    print("Running on {} threads".format(n_threads))
    job_list = []
    # now each segment makes a worker
    args = [(path, end_time) for path in eventWise_paths]
    # set up some initial jobs
    for _ in range(n_threads):
        job = multiprocessing.Process(target=append_all, args=args.pop())
        job.start()
        job_list.append(job)
    processed = 0
    for dataset_n in range(n_paths):
        job = job_list[dataset_n]
        job.join(wait_time)
        processed += 1
        # check if we shoudl stop
        if end_time - time.time() < wait_time/10:
            break
        if args:  # make a new job
            job = multiprocessing.Process(target=append_all, args=args.pop())
            job.start()
            job_list.append(job)
    # check they all stopped
    stalled = [job.is_alive() for job in job_list]
    if np.any(stalled):
        # stop everything
        for job in job_list:
            job.terminate()
            job.join()
        print(f"Problem in {sum(stalled)} out of {len(stalled)} threads")
        return False
    print("All processes ended")
    remaining_paths = eventWise_paths[:-len(args)]
    print(f"Num remaining jobs {len(remaining_paths)}")
    print(remaining_paths)
    return True


def plot_hists(eventWises, jet_name, low_pt=10.):
    eventWise_name = ["NLO" if 'nlo' in eventWise.save_name.lower() else "LO"
                      for eventWise in eventWises]
    content_prefix = jet_name + "IRC"
    contents = [name for name in eventWises[0].columns if name.startswith(content_prefix)]
    pairs = [(name, name.replace('LowPT_', '_')) for name in contents if
             'LowPT_' in name and name.replace('LowPT_', '_') in contents]
    assert len(pairs)*2 == len(contents)
    # make the axis
    fig, ax_arr = plt.subplots(len(pairs), 2)
    ax_list = ax_arr.tolist()
    for name, low_pt_name in pairs:
        variable_name = name[len(content_prefix):].replace('_', ' ').strip()
        values = [getattr(eventWise, name) for eventWise in eventWises]
        low_pt_values = [getattr(eventWise, low_pt_name) for eventWise in eventWises]
        ax1, ax2 = ax_list.pop()
        plot_hist(variable_name, eventWise_name, low_pt, values, low_pt_values, ax1, ax2)
    ax2.legend()
    fig.tight_layout()


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
    paths = []
    while True:
        name = InputTools.get_file_name(f"Eventwise {len(paths)+1} to compare; ", '.awkd').strip()
        if name:
            paths.append(name)
        else:
            break
    options = ["prepare", "plot"]
    chosen = InputTools.list_complete("What would you like to do? ", options).strip()
    if chosen == "prepare":
        duration = InputTools.get_time("How long to work for? (negative for infinite) ")
        if duration < 0:
            duration = np.inf
        end_time = time.time() + duration
        if len(paths) > 1:
            multiprocess_append(paths, end_time)
        else:
            append_all(paths[0], end_time)
    elif chosen == "plot":
        jet_names = FormJets.get_jet_names(eventWises[0])
        jet_name = InputTools.list_complete("Which jet? ", jet_names).strip()
        eventWises = [Components.EventWise.from_file(name) for path in paths]
        plot_hists(eventWises, jet_name)
        input()


