""" compare two jet clustering techniques """
import tabulate
import awkward
import ast
import csv
import os
import pickle
import matplotlib
import awkward
from ipdb import set_trace as st
from tree_tagger import Components, TrueTag, InputTools, FormJets
import sklearn.metrics
import sklearn.preprocessing
from matplotlib import pyplot as plt
import numpy as np
import scipy.stats

def rand_score(eventWise, jet_name1, jet_name2):
    # two jets clustered from the same eventWise should have
    # the same JetInput_SourceIdx, 
    # if I change this in the future I need to update this function
    selection1 = getattr(eventWise, jet_name1+"_InputIdx")
    selection2 = getattr(eventWise, jet_name2+"_InputIdx")
    num_common_events = min(len(selection1), len(selection2))
    num_inputs_per_event = Components.apply_array_func(len, eventWise.JetInputs_SourceIdx[:num_common_events])
    scores = []
    for event_n, n_inputs in enumerate(num_inputs_per_event):
        labels1 = -np.ones(n_inputs)
        for i, jet in enumerate(selection1[event_n]):
            labels1[jet[jet < n_inputs]] = i
        assert -1 not in labels1
        labels2 = -np.ones(n_inputs)
        for i, jet in enumerate(selection2[event_n]):
            labels2[jet[jet < n_inputs]] = i
        assert -1 not in labels2
        score = sklearn.metrics.adjusted_rand_score(labels1, labels2)
        scores.append(score)
    return scores


def visulise_scores(scores, jet_name1, jet_name2, score_name="Rand score"):
    plt.hist(scores, bins=40, density=True, histtype='stepfilled')
    mean = np.mean(scores)
    std = np.std(scores)
    plt.vlines([mean - 2*std, mean, mean + 2*std], 0, 1, colors=['orange', 'red', 'orange'])
    plt.xlabel(score_name)
    plt.ylabel("Density")
    plt.title(f"Similarity of {jet_name1} and {jet_name2} (for {len(scores)} events)")



def pseudovariable_differences(eventWise, jet_name1, jet_name2, var_name="Rapidity"):
    eventWise.selected_index = None
    selection1 = getattr(eventWise, jet_name1+"_InputIdx")
    selection2 = getattr(eventWise, jet_name2+"_InputIdx")
    var1 = getattr(eventWise, jet_name1 + "_" + var_name)
    var2 = getattr(eventWise, jet_name2 + "_" + var_name)
    num_common_events = min(len(selection1), len(selection2))
    num_inputs_per_event = Components.apply_array_func(len, eventWise.JetInputs_SourceIdx[:num_common_events])
    pseudojet_vars1 = []
    pseudojet_vars2 = []
    num_unconnected = 0
    for event_n, n_inputs in enumerate(num_inputs_per_event):
        values1 = {}
        for i, jet in enumerate(selection1[event_n]):
            children = tuple(sorted(jet[jet < n_inputs]))
            values = sorted(var1[event_n, i, jet >= n_inputs])
            values1[children] = values
        values2 = {}
        for i, jet in enumerate(selection2[event_n]):
            children = tuple(sorted(jet[jet < n_inputs]))
            values = sorted(var2[event_n, i, jet >= n_inputs])
            values2[children] = values
        for key1 in values1.keys():
            if key1 in values2:
                pseudojet_vars1+=values2[key1]
                pseudojet_vars2+=values2[key1]
            else:
                num_unconnected += 1
    pseudojet_vars1 = np.array(pseudojet_vars1)
    pseudojet_vars2 = np.array(pseudojet_vars2)
    return pseudojet_vars1, pseudojet_vars2, num_unconnected


def fit_to_tags(eventWise, jet_name, event_n=None, tag_pids=None, jet_pt_cut=30.):
    if event_n is None:
        assert eventWise.selected_index is not None
    else:
        eventWise.selected_index = event_n
    inputidx_name = jet_name + "_InputIdx"
    rootinputidx_name = jet_name+"_RootInputIdx"
    jet_pt = eventWise.match_indices(jet_name+"_PT", inputidx_name, rootinputidx_name).flatten()
    if jet_pt_cut is None:
        mask = np.ones_like(jet_pt)
    else:
        mask = jet_pt>jet_pt_cut
        if not np.any(mask):
            empty = np.array([]).reshape((-1, 3))
            return empty, empty
    jet_pt = jet_pt[mask]
    jet_rapidity = eventWise.match_indices(jet_name+"_Rapidity", inputidx_name, rootinputidx_name).flatten()[mask]
    jet_phi = eventWise.match_indices(jet_name+"_Phi", inputidx_name, rootinputidx_name).flatten()[mask]
    tag_idx = TrueTag.tag_particle_indices(eventWise, tag_pids=tag_pids)
    if len(tag_idx) == 0:
        empty = np.array([]).reshape((-1, 3))
        return empty, empty
    tag_rapidity = eventWise.Rapidity[tag_idx]
    tag_phi = eventWise.Phi[tag_idx]
    tag_pt = eventWise.PT[tag_idx]
    try: # if there is not more than one jet these methods fail
        # normalise the tag_pt and jet_pts, it's reasonable to think these could be corrected to match
        normed_jet_pt = sklearn.preprocessing.normalize(jet_pt.reshape(1, -1)).flatten()
    except ValueError:
        normed_jet_pt = np.ones(1)
    try:
        normed_tag_pt = sklearn.preprocessing.normalize(tag_pt.reshape(1, -1)).flatten()
    except ValueError:
        normed_tag_pt = np.ones(1)
    # divide tag and jet rapidity by the abs mean of both, effectivly collectivly normalising them
    abs_mean = np.mean((np.mean(np.abs(jet_rapidity)), np.mean(np.abs(tag_rapidity))))
    normed_jet_rapidity = jet_rapidity/abs_mean
    normed_tag_rapidity = tag_rapidity/abs_mean
    # divide the phi coordinates by pi for the same effect
    normed_jet_phi = 2*jet_phi/np.pi
    normed_tag_phi = 2*tag_phi/np.pi
    # find angular distances as they are invarient of choices
    phi_distance = np.vstack([[j_phi - t_phi for j_phi in normed_jet_phi]
                             for t_phi in normed_tag_phi])
    # remeber that these values are normalised
    phi_distance[phi_distance > 2] = 4 - phi_distance[phi_distance > 2]
    rapidity_distance = np.vstack([[j_rapidity - t_rapidity for j_rapidity in normed_jet_rapidity]
                                   for t_rapidity in normed_tag_rapidity])
    angle_dist2 = np.square(phi_distance) + np.square(rapidity_distance)
    # starting with the highest PT tag working to the lowest PT tag
    # assign tags to jets and recalculate the distance as needed
    matched_jet = np.zeros_like(tag_pt, dtype=int) - 1
    current_pt_offset_for_jet = -np.copy(normed_jet_pt)
    for tag_idx in np.argsort(tag_pt):
        this_pt = normed_tag_pt[tag_idx]
        new_pt_dist2 = np.square(current_pt_offset_for_jet + this_pt)
        allocate_to = np.argmin(new_pt_dist2 + angle_dist2[tag_idx])
        matched_jet[tag_idx] = allocate_to
        current_pt_offset_for_jet[allocate_to] += this_pt
    # now calculate what fraction of jet PT the tag actually receves
    chosen_jets = list(set(matched_jet))
    pt_fragment = np.zeros_like(matched_jet, dtype=float)
    for jet_idx in chosen_jets:
        tag_idx_here = np.where(matched_jet == jet_idx)[0]
        pt_fragment[tag_idx_here] = tag_pt[tag_idx_here]/np.sum(tag_pt[tag_idx_here])
    tag_coords = np.vstack((tag_pt, tag_rapidity, tag_phi)).transpose()
    jet_coords = np.vstack((jet_pt[matched_jet]*pt_fragment, jet_rapidity[matched_jet], jet_phi[matched_jet])).transpose()
    return tag_coords, jet_coords
    

def fit_all_to_tags(eventWise, jet_name, silent=False):
    eventWise.selected_index = None
    n_events = len(getattr(eventWise, jet_name + "_Energy"))
    tag_pids = np.genfromtxt('tree_tagger/contains_b_quark.csv', dtype=int)
    tag_coords = []
    jet_coords = []
    n_jets_formed = []
    for event_n in range(n_events):
        if event_n % 10 == 0 and not silent:
            print(f"{100*event_n/n_events}%", end='\r', flush=True)
        eventWise.selected_index = event_n
        n_jets = len(getattr(eventWise, jet_name + "_Energy"))
        n_jets_formed.append(n_jets)
        if n_jets > 0:
            tag_c, jet_c = fit_to_tags(eventWise, jet_name, tag_pids=tag_pids)
            tag_coords.append(tag_c)
            jet_coords.append(jet_c)
    tag_coords = np.vstack(tag_coords)
    jet_coords = np.vstack(jet_coords)
    return tag_coords, jet_coords, n_jets_formed


def score_rank(tag_coords, jet_coords):
    dims = 3
    scores = np.zeros(dims)
    uncerts = np.zeros(dims)
    for i in range(dims):
        scores[i], uncerts[i] = scipy.stats.spearmanr(tag_coords[:, i], jet_coords[:, i])
    return scores, uncerts


def comparison_grid1(records, rapidity=True, pt=True, phi=True):
    # find everthing with an exponent multiplier nearly at one of there three
    epsilon = 0.001
    # we only want to look at the content that has been scored
    content = records.typed_array()[records.scored]
    exp_mul = content[:, content.indices["ExponentMultiplier"]]
    KTmask = np.abs(exp_mul - 1) < epsilon
    CAmask = np.abs(exp_mul) < epsilon
    AKTmask = np.abs(exp_mul + 1) < epsilon
    axis_dict = {"AKT": AKTmask, "CA": CAmask, "KT": KTmask}
    
    fig, axes = plt.subplots(len(axis_dict), 2, sharex=True)
    if len(axes.shape) < 2:
        axes = [axes]
    for ax_name, ax_pair in zip(axis_dict, axes):
        mask = axis_dict[ax_name]
        ax1, ax2 = ax_pair
        ax1.set_xlabel("$\\Delta R$")
        ax1.set_ylabel(f"{ax_name} rank score")
        xs = content[mask, records.indices["DeltaR"]]
        num_eig = content[mask, records.indices["NumEigenvectors"]]
        if len(set(num_eig)) > 1:
            max_eig = np.nanmax(num_eig)
            colour_map = matplotlib.cm.get_cmap('viridis')
            colours = colour_map(num_eig/max_eig)
            colours = [tuple(c) for c in colours]
            colour_ticks = ["No spectral"] + [str(c+1) for c in range(int(max_eig))]
        else:
            colours = None
        rap_marker = 'v'
        pt_marker = '^'
        phi_marker = 'o'
        if pt:
            ax1.scatter(xs, content[mask, records.indices["score(PT)"]],
                         marker=pt_marker, c=colours, label="PT")
        if rapidity:
            ax1.scatter(xs, content[mask, records.indices["score(Rapidity)"]],
                         marker=rap_marker, c=colours, label="Rapidity")
        ax2.set_xlabel("$\\Delta R$")
        ax2.set_ylabel("Symmetrised % diff")
        if pt:
            ax2.scatter(xs, content[mask, records.indices["symmetric_diff(PT)"]]*100, 
                        marker=pt_marker, c=colours, label="PT")
        if rapidity:
            ax2.scatter(xs, content[mask, records.indices["symmetric_diff(Rapidity)"]]*100, 
                        marker=rap_marker, c=colours, label="Rapidity")
        if phi:
            ax2.scatter(xs, content[mask, records.indices["symmetric_diff(Phi)"]]*100, 
                        marker=phi_marker, c=colours, label="Phi")
        ax2.legend()
        ax2.set_ylim(0, max(100, ax2.get_ylim()[1]))
        if colours is not None:
            norm = matplotlib.colors.Normalize(vmin=0, vmax=max_eig)
            mapable = matplotlib.cm.ScalarMappable(norm=norm, cmap=colour_map)
            mapable.set_array([])
            cbar = fig.colorbar(mapable, ax=ax2, ticks=np.linspace(0, max_eig, len(colour_ticks)))
            cbar.ax.set_yticklabels(colour_ticks)
    plt.show()
    return axes_data


def comparison_grid2(records, rapidity=True, pt=True, phi=True):
    # we only want to look at the content that has been scored
    content = records.typed_array()[records.scored]
    mask = np.abs(content[:, records.indices["DeltaR"]] - 0.4) < 0.001
    num_eig = content[mask, records.indices["NumEigenvectors"]]
    exp_values = 2*content[mask, records.indices["ExponentMultiplier"]]
    scorePT = content[mask, records.indices["score(PT)"]]
    scoreRapidity = content[mask, records.indices["score(Rapidity)"]]
    sydPT = content[mask, records.indices["symmetric_diff(PT)"]]
    sydRapidity = content[mask, records.indices["symmetric_diff(Rapidity)"]]
    sydPhi = content[mask, records.indices["symmetric_diff(Phi)"]]
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
    ax1.set_xlabel("exponent")
    ax1.set_ylabel(f"rank score")
    xs = exp_values
    if len(set(num_eig)) > 1:
        is_inf = np.inf == num_eig
        max_eig = np.nanmax(num_eig[~is_inf])
        colour_map = matplotlib.cm.get_cmap('viridis')
        colours = colour_map(num_eig/max_eig + np.any(is_inf))
        colours = [tuple(c) for c in colours]
        colour_ticks = ["No spectral"] + [str(c+1) for c in range(int(max_eig))]
        if np.any(is_inf):
            colour_ticks += ["Max"]
            num_eig[is_inf] = max_eig + 1
    else:
        colours = None
    rap_marker = 'v'
    pt_marker = '^'
    phi_marker = 'o'
    if pt:
        ax1.scatter(xs, scorePT,
                     marker=pt_marker, c=colours, label="PT")
    if rapidity:
        ax1.scatter(xs, scoreRapidity,
                     marker=rap_marker, c=colours, label="Rapidity")
    ax2.set_xlabel("exponent")
    ax2.set_ylabel("Symmetrised % diff")
    if pt:
        ax2.scatter(xs, sydPT*100,
                    marker=pt_marker, c=colours, label="PT")
    if rapidity:
        ax2.scatter(xs, sydRapidity*100,
                    marker=rap_marker, c=colours, label="Rapidity")
    if phi:
        ax2.scatter(xs, sydPhi*100,
                    marker=phi_marker, c=colours, label="Phi")
    ax2.legend()
    ax2.set_ylim(0, max(100, np.max(np.concatenate((sydPT, sydRapidity, sydPhi)))))
    if colours is not None:
        norm = matplotlib.colors.Normalize(vmin=0, vmax=max_eig)
        mapable = matplotlib.cm.ScalarMappable(norm=norm, cmap=colour_map)
        mapable.set_array([])
        cbar = fig.colorbar(mapable, ax=ax2, ticks=np.linspace(0, max_eig, len(colour_ticks)))
        cbar.ax.set_yticklabels(colour_ticks)
    plt.show()
    return axes_data


def calculated_grid(records, jet_name=None):
    names = {"FastJet": FormJets.Traditional, "HomeJet": FormJets.Traditional,
             "SpectralJet": FormJets.Spectral, "SpectralMeanJet": FormJets.SpectralMean}
    if jet_name is None:
        jet_name = InputTools.list_complete("Which jet? ", names.keys()).strip()
    default_params = names[jet_name].param_list
    param_list = sorted(default_params.keys())
    # we only want to look at the content that has been scored
    content = records.typed_array()[records.scored]
    catigories = {name: set(content[:, records.indices[name]])
                  for name in param_list}
    print("Parameter: num_catigories")
    for name in catigories:
        print(f"{name}: {len(catigories[name])}")
    horizontal_param = InputTools.list_complete("Horizontal param? ", param_list).strip()
    vertical_param = InputTools.list_complete("Vertical param? ", param_list).strip()
    horizontal_bins = sorted(catigories[horizontal_param])
    if isinstance(horizontal_bins[0], str):
        def get_h_index(value):
            return horizontal_bins.index(value)
    else:
        horizontal_bins_a = np.array(horizontal_bins)
        def get_h_index(value):
            return np.argmin(np.abs(horizontal_bins_a - value))
    vertical_bins = sorted(catigories[vertical_param])
    if isinstance(vertical_bins[0], str):
        def get_v_index(value):
            return vertical_bins.index(value)
    else:
        vertical_bins_a = np.array(vertical_bins)
        def get_v_index(value):
            return np.argmin(np.abs(vertical_bins_a - value))
    grid = [[[] for _ in horizontal_bins] for _ in vertical_bins]
    h_column = records.indices[horizontal_param]
    v_column = records.indices[vertical_param]
    for row in content:
        id_num = row[0]
        v_index = get_v_index(row[h_column])
        h_index = get_h_index(row[v_column])
        grid[v_index][h_index].append(id_num)
    table = [[value] + [len(entry) for entry in row] for value, row in zip(vertical_bins, grid)]
    first_row = [["\\".join([vertical_param, horizontal_param])] + horizontal_bins]
    table = first_row + table
    str_table = tabulate.tabulate(table, headers="firstrow")
    print(str_table)
    if InputTools.yesNo_question("Again? "):
        calculated_grid(records)


class Records:
    delimiter = '\t'
    evaluation_columns = ("score(PT)", "score_uncert(PT)", "symmetric_diff(PT)", "symdiff_std(PT)",
                          "score(Rapidity)", "score_uncert(Rapidity)", "symmetric_diff(Rapidity)", "symdiff_std(Rapidity)",
                          "score(Phi)", "score_uncert(Phi)", "symmetric_diff(Phi)", "symdiff_std(Phi)",
                          "mean_njets", "std_njets")
    def __init__(self, file_path):
        self.file_path = file_path
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r') as existing:
                reader = csv.reader(existing, delimiter=self.delimiter)
                header = next(reader)
                assert header[1] == 'jet_class'
                self.param_names = header[2:]
                self.indices = {name: i+2 for i, name in enumerate(self.param_names)}
                self.content = []
                for line in reader:
                    self.content.append(line)
        else:
            with open(self.file_path, 'w') as new:
                writer = csv.writer(new, delimiter=self.delimiter)
                header = ['id', 'jet_class']
                writer.writerow(header)
            self.content = []
            self.param_name = []
            self.indices = {}
        self.next_uid = int(np.max(self.jet_ids, initial=0)) + 1
        self.uid_length = len(str(self.next_uid))

    def write(self):
        with open(self.file_path, 'w') as overwrite:
            writer = csv.writer(overwrite, delimiter=self.delimiter)
            all_rows = [['', 'jet_class'] + self.param_names] + self.content
            writer.writerows(all_rows)

    def typed_array(self):
        """Convert the contents to an array of apropreate type,
           fill blanks with default"""
        jet_classes = {"HomeJet": FormJets.Traditional,
                       "FastJet": FormJets.Traditional,
                       "SpectralJet": FormJets.Spectral,
                       "SpectralMeanJet": FormJets.SpectralMean}
        typed_content = []
        for row in self.content:
            id_num = int(row[0])
            jet_class = row[1]
            typed_content.append([id_num, jet_class])
            for param_name, entry in zip(self.param_names, row[2:]):
                if entry == '':
                    # set to the default
                    try:
                        typed = jet_classes[jet_class].param_list[param_name]
                    except KeyError:
                        typed = None
                elif entry == 'inf':
                    typed = np.inf
                else:
                    try:
                        typed = ast.literal_eval(entry)
                    except ValueError:
                        # it's probably a string
                        typed = entry
                typed_content[-1].append(typed)
        # got to be an awkward array because numpy hates mixed types
        return awkward.fromiter(typed_content)

    @property
    def jet_ids(self):
        ids = [int(row[0]) for row in self.content]
        return ids

    @property
    def scored(self):
        if 'mean_njets' not in self.param_names:
            return np.full(len(self.content), False)
        scored = [row[self.indices["mean_njets"]] not in ('', None)
                  for row in self.content]
        return np.array(scored)

    def _add_param(self, *new_params):
        new_params = [n for n in new_params if n not in self.param_names]
        self.param_names += new_params
        self.indices = {name: i+2 for i, name in enumerate(self.param_names)}
        new_blanks = ['' for _ in new_params]
        self.content = [row + new_blanks for row in self.content]

    def append(self, jet_class, param_dict, existing_idx=None, write_now=True):
        """ gives the new jet a unique ID and returns that value"""
        if existing_idx is None:
            chosen_id = self.next_uid
        else:
            assert existing_idx not in self.jet_ids
            chosen_id = existing_idx
        new_row = [f"{chosen_id:0{self.uid_length}d}", jet_class]
        new_params = list(set(param_dict.keys()) - set(self.param_names))
        self._add_param(*new_params)
        for name in self.param_names:
            new_row.append(str(param_dict.get(name, '')))
        # write to disk
        if write_now:
            with open(self.file_path, 'a') as existing:
                writer = csv.writer(existing, delimiter=self.delimiter)
                writer.writerow(new_row)
        # update content in memeory
        self.content.append(new_row)
        self.next_uid += 1
        self.uid_length = len(str(self.next_uid))
        return chosen_id

    def scan(self, eventWise):
        eventWise.selected_index = None
        jet_names = {c.split('_', 1)[0] for c in eventWise.columns
                     if (not c.startswith('JetInputs')) and 'Jet' in c}
        existing = {}  # dicts like  "jet_name": int(row_idx)
        added = {}
        starting_ids = self.jet_ids
        num_events = len(eventWise.JetInputs_Energy)
        content = self.typed_array()
        for name in jet_names:
            try:
                num_start = next(i for i, l in enumerate(name) if l.isdigit())
            except StopIteration:
                print(f"{name} does not have an id number, may not be a jet")
                continue
            jet_params = FormJets.get_jet_params(eventWise, name)
            jet_class = name[:num_start]
            id_num = int(name[num_start:])
            if id_num in starting_ids:
                idx = starting_ids.index(id_num)
                row = content[idx]
                # verify the hyperparameters
                match = True  # set match here incase jet has no params
                for p_name in jet_params:
                    if p_name not in self.indices:
                        # this parameter wasn't recorded, add it
                        self._add_param(p_name)
                        # the row length will have changed
                        self.content[idx][self.indices[p_name]] = jet_params[p_name]
                        content = self.typed_array()
                        continue  # no sense in checking it
                    try:
                        match = np.allclose(jet_params[p_name], row[self.indices[p_name]])
                    except TypeError:
                        match = jet_params[p_name] == row[self.indices[p_name]]
                    if not match:
                        break
                if match:
                    # check if it's actually been clustered
                    any_col = next(c for c in eventWise.columns if c.startswith(name))
                    num_found = len(getattr(eventWise, any_col))
                    if num_found == num_events:  # perfect
                        existing[name] = idx
                    elif num_found < num_events:  # incomplete
                        self._add_param("incomplete")
                        self.content[idx][self.indices["incomplete"]] = True
                    elif num_found > num_events:  # wtf
                        raise ValueError(f"Jet {name} has more calculated events than there are jet input events")
                else:
                    self._add_param("match_error")
                    self.content[idx][self.indices["match_error"]] = True
            else:  # this ID not found in jets
                self.append(jet_class, jet_params, existing_idx=id_num, write_now=False)
                added[name] = len(self.content) - 1
        self.write()
        return existing, added

    def score(self, eventWise):
        self._add_param(*self.evaluation_columns)
        print("Scanning eventWise")
        existing, added = self.scan(eventWise)
        all_jets = {**existing, **added}
        num_names = len(all_jets)
        print(f"Found  {num_names}")
        print("Making a continue file, delete it to halt the evauation")
        open("continue", 'w').close()
        scored = self.scored
        for i, name in enumerate(all_jets):
            if i % 2 == 0:
                print(f"{100*i/num_names}%", end='\r', flush=True)
                if not os.path.exists('continue'):
                    break
                if i % 30 == 0:
                    self.write()
                    # delete and reread the eventWise, this should free up some ram
                    # because the eventWise is lazy loading
                    path = os.path.join(eventWise.dir_name, eventWise.save_name)
                    del eventWise
                    eventWise = Components.EventWise.from_file(path)
            row = self.content[all_jets[name]]
            if not scored[i]:
                tag_coords, jet_coords, n_jets_formed = fit_all_to_tags(eventWise, name, silent=True)
                row[self.indices["mean_njets"]] = np.mean(n_jets_formed)
                row[self.indices["std_njets"]] = np.std(n_jets_formed)
                scores, uncerts = score_rank(tag_coords, jet_coords)
                row[self.indices["score(PT)"]] = scores[0]
                row[self.indices["score_uncert(PT)"]] = uncerts[0]
                row[self.indices["score(Rapidity)"]] = scores[1]
                row[self.indices["score_uncert(Rapidity)"]] = uncerts[1]
                row[self.indices["score(Phi)"]] = scores[2]
                row[self.indices["score_uncert(Phi)"]] = uncerts[2]
                syd = 2*np.abs(tag_coords - jet_coords)/(np.abs(tag_coords) + np.abs(jet_coords))
                # but the symetric difernce for phi should be angular
                row[self.indices["symmetric_diff(PT)"]] = np.mean(syd[0])
                row[self.indices["symdiff_std(PT)"]] = np.std(syd[0])
                row[self.indices["symmetric_diff(Rapidity)"]] = np.mean(syd[1])
                row[self.indices["symdiff_std(Rapidity)"]] = np.std(syd[1])
                row[self.indices["symmetric_diff(Phi)"]] = np.mean(syd[2])
                row[self.indices["symdiff_std(Phi)"]] = np.std(syd[2])
        self.write()


if __name__ == '__main__':
    pass

