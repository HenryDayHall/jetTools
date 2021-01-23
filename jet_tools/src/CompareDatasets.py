from . import Components, PDGNames, InputTools, FormJets
import ast
import multiprocessing
import time
import os
from ipdb import set_trace as st
import collections
from matplotlib import pyplot as plt
import numpy as np
import awkward
import scipy.spatial, scipy.stats


def subsample_dissdiff(data1, data2, diff_function, n_subsamples=3):
    # start be making subsamples
    n_subsamples = int(np.min([len(data1)/5, len(data2)/5, n_subsamples]))
    sub_sum = 0
    for split_point in np.linspace(0, len(data1), n_subsamples, dtype=int)[1:-1]:
        sub_sum += diff_function(data1[:split_point], data1[split_point:])
    for split_point in np.linspace(0, len(data2), n_subsamples, dtype=int)[1:-1]:
        sub_sum += diff_function(data2[:split_point], data2[split_point:])
    dissdiff = 2*n_subsamples*diff_function(data1, data2)/sub_sum
    return dissdiff


def subsample_ks(data1, data2):
    diff_function = lambda d1, d2: scipy.stats.ks_2samp(d1, d2)[0]
    return subsample_dissdiff(data1, data2, diff_function), None


def subsample_es(data1, data2):
    diff_function = lambda d1, d2: scipy.stats.epps_singleton_2samp(d1, d2)[1]
    return subsample_dissdiff(data1, data2, diff_function), None

def subsample_js(data1, data2):
    diff_function = lambda d1, d2: jensen_shannon(d1, d2)[0]
    return subsample_dissdiff(data1, data2, diff_function), None


def probablility_vectors(data1, data2, bins=20):
    all_data = np.concatenate((data1, data2))
    data_range = (np.nanmin(all_data), np.nanmax(all_data))
    probs1, _ = np.histogram(data1, bins=bins, range=data_range)
    probs1 = probs1/np.sum(probs1)
    probs2, _ = np.histogram(data2, bins=bins, range=data_range)
    probs2 = probs2/np.sum(probs2)
    return probs1, probs2
    

def jensen_shannon(data1, data2):
    probs1, probs2 = probablility_vectors(data1, data2)
    return scipy.spatial.distance.jensenshannon(probs1, probs2), None


def kullback_Leibler(data1, data2):
    probs1, probs2 = probablility_vectors(data1, data2)
    return scipy.stats.entropy(probs1, probs2), None

metric_dict = {'js': jensen_shannon, 'kl': kullback_Leibler,
               'es': scipy.stats.epps_singleton_2samp,
               'ks': scipy.stats.ks_2samp, 'ksu': subsample_ks,
               'jsu': subsample_js}
metric_names = {'js': 'Jensen-Shannon', 'kl': 'Kullback-Leibler',
                'es': 'Epps-Singleton',
                'ks': "Kolmogrov-Smirnov"}

def calculate_dissdiff_values(path1, path2, save_name=None, metric_name='js'):
    eventWise1 = Components.EventWise.from_file(path1)
    jet_names = FormJets.get_jet_names(eventWise1)
    dissdiff_values = {}
    metric = metric_dict[metric_name]
    desired_endings = ["IRC_DeltaR", "IRC_JetMass", "IRC_relativePT"]
    for jet_name in jet_names:
        # reload to reduce data in ram
        eventWise1 = Components.EventWise.from_file(path1)
        eventWise2 = Components.EventWise.from_file(path2)
        # get contents
        contents = [jet_name + end for end in desired_endings
                    if jet_name + end in eventWise1.columns]
        for content in contents:
            data1 = getattr(eventWise1, content).flatten()
            data2 = getattr(eventWise2, content).flatten()
            # remove nans
            data1 = data1[np.isfinite(data1)]
            data2 = data2[np.isfinite(data2)]
            if min(len(data1), len(data2)) < 5:
                continue
            try:
                dissdiff_values[content] = metric(data1, data2)[0]
            except ZeroDivisionError:
                pass
    if save_name is None:
        name1 = eventWise1.save_name[:-5].replace('_', '')
        name2 = eventWise2.save_name[:-5].replace('_', '')
        save_name = f"IRC_meta_dicts/{metric_name}_{name1}_{name2}.txt"
    with open(save_name, 'w') as save_file:
        save_file.write(str(dissdiff_values))
    return dissdiff_values


def tabulate_dissdiff_dict(dissdiff_values, jet_classes=None, variable_types=None, existing_table=None):
    if jet_classes is None:
        jet_classes = []
        variable_types = []
        existing_table = []
    for key in dissdiff_values:
        # get the class name
        jet_class = ""
        variable_type = ""
        part = 0
        for c in key:
            if c.isdigit():
                part += 1
            elif part == 0:
                jet_class += c
            else:
                variable_type += c
        if jet_class not in jet_classes:
            jet_classes.append(jet_class)
            for i, var in enumerate(variable_types):
                existing_table[i].append([])
        if variable_type not in variable_types:
            variable_types.append(variable_type)
            existing_table.append([[] for _ in jet_classes])
        row = variable_types.index(variable_type)
        col = jet_classes.index(jet_class)
        value = np.nan if dissdiff_values[key] is None else dissdiff_values[key]
        existing_table[row][col].append(value)
    return jet_classes, variable_types, existing_table


def read_dissdiffs(list_dissdiff_values, output=None):
    jet_classes = []
    variable_types = []
    table = []
    for dissdiff_values in list_dissdiff_values:
        # if needed read in file
        if isinstance(dissdiff_values, str):
            with open(dissdiff_values, 'r') as save_file:
                line = save_file.readline()
            dissdiff_values = ast.literal_eval(line)
            assert isinstance(dissdiff_values, dict)
        if output is not None:
            output.append(dissdiff_values)
        # get the class name
        jet_classes, variable_types, table = tabulate_dissdiff_dict(dissdiff_values,
                                                                    jet_classes,
                                                                    variable_types,
                                                                    table)
    return jet_classes, variable_types, table


def plot_dissdiff_values(list_dissdiff_values, score_name=None):
    if score_name is None:
        for key in metric_names:
            if '_' + key in list_dissdiff_values[0]:
                score_name = metric_names[key]
                break
        else:
            score_name = "Score"
        if '_normed' in list_dissdiff_values[0]:
            score_name += " normed"
        else:
            score_name += " unnormed"
    jet_classes, variable_types, table = read_dissdiffs(list_dissdiff_values)
    #fig, ax_arr = plt.subplots(len(variable_types), 1)
    fig, ax_arr = plt.subplots(1, 1)
    variable_types = [n for n in variable_types if "Mass" in n]
    if len(variable_types) == 1:
        ax_arr = [ax_arr]
    fig.suptitle(f"{score_name} between LO and NLO data")
    for var_type, ax in zip(variable_types, ax_arr):
        row = variable_types.index(var_type)
        class_values = table[row]
        bins=130
        st()
        ax.hist(class_values, label=jet_classes, bins=bins, density=True, histtype='step')
        ax.set_ylabel("Frequency")
        #ax.set_xlabel(f"{score_name} for {var_type} change")
        ax.set_xlabel(f"{score_name} for jet mass change")
    ax.legend()
    fig.subplots_adjust(top=0.95, bottom=0.08, left=0.08, right=0.95, hspace=0.4, wspace=0.21)


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


def append_flat_IRC_variables(eventWise, jet_name=None, append=False):
    if jet_name is None:
        return  # cannot caculate mass for the event
    else:
        jet_str = jet_name
    # jet mass
    new_content = {}
    eventWise.selected_index = None
    jet_roots = getattr(eventWise, jet_name + "_Parent") == -1
    pts = awkward.fromiter([p[r] for p, r in 
                            zip(getattr(eventWise, jet_name + "_PT"), jet_roots)]).flatten()
    pzs = awkward.fromiter([p[r] for p, r in 
                            zip(getattr(eventWise, jet_name + "_Pz"), jet_roots)]).flatten()
    es = awkward.fromiter([e[r] for e, r in 
                           zip(getattr(eventWise, jet_name + "_Energy"), jet_roots)]).flatten()
    values = np.sqrt(es**2 - pts**2 - pzs**2)
    new_content[jet_str + "IRC_JetMass"] = values
    if append:
        eventWise.append(**new_content)
    else:
        return new_content


def awkward_to_2d(array, depth=1):
    for _ in range(depth):
        array = array.flatten()
    return np.array(array.tolist()).reshape((-1, 1))


def append_pairwise_IRC_variables(eventWise, jet_name=None, append=False):
    jet_roots = getattr(eventWise, jet_name + "_Parent") == -1
    if jet_name is None:
        jet_str = ""
    else:
        jet_str = jet_name
    new_content = {}
    # PT
    if jet_name is None:
        pts = [scipy.spatial.distance.pdist(event.reshape(-1, 1), metric=min)
               for event in eventWise.PT[eventWise.Is_leaf]]
        angles = [1-scipy.spatial.distance.pdist(np.stack((px, py)).T, metric='cosine')
                  for px, py in
                  zip(eventWise.Px[eventWise.Is_leaf], eventWise.Py[eventWise.Is_leaf])]
        values = awkward.fromiter([angle*pt for angle, pt in zip(angles, pts)])
    else:
        jet_pxs = getattr(eventWise, jet_name + "_Px")
        jet_pys = getattr(eventWise, jet_name + "_Py")
        jet_pts = getattr(eventWise, jet_name + "_PT")
        pts = [scipy.spatial.distance.pdist(event.flatten().reshape((-1, 1)), metric=min)
               for event in jet_pts]
        angles = [1-scipy.spatial.distance.pdist(np.stack((px.flatten(), py.flatten())).T,
                                                 metric='cosine')
                  for px, py in zip(jet_pxs, jet_pys)]
        values = awkward.fromiter([angle*pt for angle, pt in zip(angles, pts)])
    values = awkward.fromiter(values).flatten().tolist()
    new_content[jet_str + "IRC_relativePT"] = values
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
        phi = [scipy.spatial.distance.pdist(awkward_to_2d(event), metric=Components.angular_distance)
               for event in
               eventWise.Phi[eventWise.Is_leaf]]
    else:
        jet_phi = getattr(eventWise, jet_name + "_Phi")
        phi = [scipy.spatial.distance.pdist(awkward_to_2d(ph[root]), metric=Components.angular_distance)
               for ph, root in
               zip(jet_phi, jet_roots)]
    phi = awkward.fromiter(phi)
    values = np.sqrt(awkward.fromiter(rapidity)**2 + awkward.fromiter(phi)**2)
    values = values.flatten().tolist()
    new_content[jet_str + "IRC_DeltaR"] = values
    if append:
        eventWise.append(**new_content)
    else:
        return new_content


def append_all(path, end_time, overwrite=False):
    eventWise = Components.EventWise.from_file(path)
    new_content = {}
    jet_names = FormJets.get_jet_names(eventWise)
    n_jets = len(jet_names)
    start_columns = eventWise.columns
    for i, jet_name in enumerate(jet_names):
        # prevent overwites
        found = next((name for name in start_columns if name.startswith(jet_name + "IRC")),
                     False)
        if found and not overwrite:
            continue
        if (i+1) % 10 == 0:  # reload to preserve ram
            new_content = {key: awkward.fromiter(value) for key, value in new_content.items()}
            eventWise.append(**new_content)
            new_content = {}
            eventWise = Components.EventWise.from_file(path)
        if time.time() > end_time:
            break
        print(f'{i/n_jets:%}', end='\r', flush=True)
        #new_c = append_pairwise_IRC_variables(eventWise, jet_name)
        #new_content.update(new_c)
        new_c = append_flat_IRC_variables(eventWise, jet_name)
        new_content.update(new_c)
    new_content = {key: awkward.fromiter(value) for key, value in new_content.items()}
    eventWise.append(**new_content)
    print(f"\nDone {eventWise.save_name}\n", flush=True)


def multiprocess_append(eventWise_paths, end_time, overwrite=False, leave_one_free=True):
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
    args = [(path, end_time, overwrite) for path in eventWise_paths]
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


def plot_catagory(eventWises, jet_name, ax_arr=None):
    eventWise_name = ["NLO" if 'nlo' in eventWise.save_name.lower() else "LO"
                      for eventWise in eventWises]
    desired_endings = ["IRC_JetMass"]
    # make the axis
    if ax_arr is None:
        fig, ax_arr = plt.subplots(len(desired_endings), 1)
        set_spaceing = True
    else:
        ax_arr[0].set_title(jet_name)
        set_spaceing = False
    try:
        ax_list = ax_arr.tolist()
    except AttributeError:
        ax_list = [ax_arr]
    for end in desired_endings:
        name = jet_name + end
        variable_name = end[4:]
        values = [getattr(eventWise, name).flatten().tolist() for eventWise in eventWises]
        if end == "IRC_relativePT":
            values = np.log(np.abs(awkward.fromiter(values)))
            variable_name = "log(|relativePT|)"
        ax = ax_list.pop()
        bins = 30
        if "mass" in variable_name.lower():
            all_values = np.concatenate(values)
            min_non_zero = np.nanmin(all_values[all_values>0])
            max_val = np.nanmax(all_values)
            min_non_zero = max(max_val/2000, min_non_zero)
            bins = np.logspace(np.log10(min_non_zero), np.log10(max_val), 30)
        plot_hist(variable_name, eventWise_name, values, ax, bins=bins)
    ax.legend()
    if set_spaceing:
        fig.suptitle(jet_name)
        fig.set_size_inches(6, 8)
        fig.subplots_adjust(top=0.911,
                bottom=0.071,
                left=0.171,
                right=0.976,
                hspace=0.5,
                wspace=0.5)


def plot_hist(variable_name, names, values, ax, bins=None):
    ax.hist(values, histtype='step', label=names, bins=bins, density=True)
    ax.set_xlabel(variable_name)
    ax.set_ylabel("Normed Frequency")


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
    options = ["prepare", "plot", "dissdiff"]
    chosen = InputTools.list_complete("What would you like to do? ", options).strip()
    if chosen == "prepare":
        duration = InputTools.get_time("How long to work for? (negative for infinite) ")
        if duration < 0:
            duration = np.inf
        end_time = time.time() + duration
        overwrite = InputTools.yesNo_question("Overwrite existing? ")
        if len(paths) > 1:
            multiprocess_append(paths, end_time, overwrite)
        else:
            append_all(paths[0], end_time, overwrite)
    elif chosen == "plot":
        eventWises = [Components.EventWise.from_file(path) for path in paths]
        jet_names = FormJets.get_jet_names(eventWises[0])
        jet_name = InputTools.list_complete("Which jet? ", jet_names).strip()
        plot_catagory(eventWises, jet_name)
        plt.show()
        input()
    elif chosen == "dissdiff":
        dissdiff_name = InputTools.get_file_name("Name a file to save the dissdiff scores in; ",
                                           '.txt').strip()
        metric = InputTools.list_complete("Which metric? ", metric_dict).strip()
        dissdiff_values = calculate_dissdiff_values(paths[0], paths[1], dissdiff_name,
                                                    metric)
        #plot_dissdiff_values([dissdiff_values])
        #input()





