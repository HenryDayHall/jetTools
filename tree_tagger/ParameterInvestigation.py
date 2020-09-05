""" Module to investigate parameter choices, base on questions in writeup """
import itertools
import warnings
from collections import OrderedDict
from ipdb import set_trace as st
import os
import numpy as np
from tree_tagger import FormJets, FormShower, PlottingTools, Components, InputTools
import scipy.spatial
import scipy.stats
import sklearn.metrics
import time
import awkward
from matplotlib import pyplot as plt
import matplotlib

# a metric function
def min_sep(u, v):
    """
    Difernence in the dimension with minimal seperation.

    Parameters
    ----------
    u : array like of floats
        vector 1
    v : array like of floats
        vector 2

    Returns
    -------
    : float
        the diference

    """
    return np.min(np.abs(u - v))

metrics = OrderedDict(Euclidian = dict(metric='euclidean'),
                      L3 = dict(metric='minkowski', p=3),
                      L4 = dict(metric='minkowski', p=4),
                      taxicab = dict(metric='cityblock'),
                      Braycurtis = dict(metric='braycurtis'),
                      Canberra = dict(metric='canberra'),
                      Min = dict(metric=min_sep),
                      Max = dict(metric='chebyshev'),
                      Correlation = dict(metric='correlation'),
                      Cosine = dict(metric='cosine'))
eig_metric_names = list(metrics.keys()) + [name + " normed" for name in metrics]
phys_metric_names = list(metrics.keys()) + [name + " Luclus" for name in metrics]


def get_seperations(vectors, norm_values=None):
    """
    For each pair of jet inputs in an event get the seperation.
    Each or the metrics in the dict will be tried,
    then retried with a normed space if norming norm_values are given
    A normed space is one where the vectors have been divided
    by the norm_values.

    Parameters
    ----------
    vectors : list of 2d numpy arrays of floats
        the vectors of each event.
    norm_values : list of 1d numpy arrays of floats
        the norm_values of each event.

    Returns
    -------
    seperations : list of 3d numpy arrays of floats
        the distances in each event
        axis 0 is each of the metrics used
        axis 1 and 2 are the particles in each event

    """
    seperations = []
    n_metrics = len(metrics)
    if norm_values:
        n_metrics *= 2
    # suppress warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for event_n, vecs in enumerate(vectors):
            n_points = len(vecs)
            # make a array for the results
            local = np.zeros((n_metrics, n_points, n_points), dtype=float)
            # do without norming
            for metric_n, name in enumerate(metrics):
                distance = scipy.spatial.distance.pdist(vecs, **metrics[name])
                local[metric_n] = scipy.spatial.distance.squareform(distance)
            # now normalise the vectors and go again
            if norm_values is not None:
                vecs /= norm_values[event_n]
                after_norm = metric_n+1
                for metric_n, name in enumerate(metrics):
                    distance = scipy.spatial.distance.pdist(vecs, **metrics[name])
                    local[after_norm + metric_n] = scipy.spatial.distance.squareform(distance)
            seperations.append(local)
    return seperations


def label_parings(eventWise):
    """
    For every pair of jet inputs, label if they are from the same b quark

    Parameters
    ----------
    eventWise : EventWise
        Data set containing particle data.

    Returns
    -------
    labels : list of numpy arrays of bools
        for each 

    """
    labels = []
    eventWise.selected_index = None
    for event_n in range(len(eventWise.X)):
        eventWise.selected_index = event_n
        jet_inputs = eventWise.JetInputs_SourceIdx
        n_inputs = len(jet_inputs)
        local = np.full((n_inputs, n_inputs), False, dtype=bool)
        for b in eventWise.BQuarkIdx:
            decendants = FormShower.descendant_idxs(eventWise, b)
            is_decendent = np.fromiter((p in decendants for p in jet_inputs),
                                       dtype=bool)
            local += np.expand_dims(is_decendent, 0) * np.expand_dims(is_decendent, 1)
        labels.append(local)
    return labels


def label_crossings(labels):
    """ The prinicple is that is a particle is in a b-jet it's disgonal will be true
    If the other particle is not the offdiagonal will be false """
    crossings = []
    for label in labels:
        # where in_jet is false the outcome must be false,
        # but all the off diagonal wil be anyway
        in_jet = np.diag(label)
        local = np.logical_xor(in_jet, in_jet.reshape((-1, 1)))
        # the whole diagonal will be made false by this automaticaly
        crossings.append(local)
    return crossings


def closest_relative(eventWise):
    """
    for evey pair of jet inputs find the minimum number of steps to the
    closest relative

    Parameters
    ----------
    eventWise :
        

    Returns
    -------

    """
    labels = []
    eventWise.selected_index = None
    for event_n in range(len(eventWise.X)):
        eventWise.selected_index = event_n
        jet_inputs = eventWise.JetInputs_SourceIdx.tolist()
        n_inputs = len(jet_inputs)
        parents = eventWise.Parents.tolist()
        n_nodes = len(parents)
        input_mask = [i in jet_inputs for i in range(n_nodes)]
        children = eventWise.Children.tolist()
        linked = [c + p for c, p in zip(parents, children)]
        # each input will propagate the number of steps it must take
        # to reach all inputs up to it
        labels_here = np.zeros((n_inputs, n_inputs), dtype=int)
        # make a matrix that will be reused each time
        steps_to = np.empty(n_nodes, dtype=int)
        for loc_a, glob_a in enumerate(jet_inputs):
            # it cannot take more steps than the nuber of nodes
            steps_to[:] = n_nodes
            # it takes to steps to itself
            steps_to[glob_a] = 0
            stack = [glob_a]
            while stack:
                glob_b = stack.pop()
                next_step = steps_to[glob_b] + 1
                for neigbour in linked[glob_b]:
                    if steps_to[neigbour] > next_step:
                        # we found a shorter route
                        steps_to[neigbour] = next_step
                        # keep moving this way
                        stack.append(neigbour)
                    # else ignore thie neigbour
            # now fill in the local grid
            labels_here[:loc_a, loc_a] = labels_here[loc_a, :loc_a] = steps_to[input_mask][:loc_a]
        labels.append(labels_here)
    return labels


# Plotting tools  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Physical distance ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def physical_distances(phis, rapidities, pts, exp_multipler=0):
    exponent = exp_multipler*2
    empty = np.empty((0, 0))
    vectors = [empty if not len(phi) else np.vstack((phi, rap)).T
               for phi, rap in zip(phis, rapidities)]
    metric_distances = get_seperations(vectors)
    distances = []
    # multiply by 2 to account for Luclus
    num_metrics = len(metrics)*2
    luclus_start = len(metrics)
    for pt, m_dist in zip(pts, metric_distances):
        # the angular section is the sam for normal and luclus
        local = np.concatenate((m_dist, m_dist), axis=0)
        # sort out the pt factor for the normal section
        exp_pt = pt**exponent
        col_pt = pt.reshape((-1, 1))
        col_exp_pt = exp_pt.reshape((-1, 1))
        local[:luclus_start] *= np.minimum(exp_pt, col_exp_pt)
        # sort out the luclus factor
        luclus_factor = (exp_pt * col_exp_pt) * ((pt + col_pt)**-exponent)
        local[luclus_start:] *= luclus_factor
        distances.append(local)
    return distances


def append_phys_metrics(eventWise, jet_names, jet_param_list, duration=np.inf):
    """
    

    Parameters
    ----------
    eventWise :
        
    jet_names :
        
    jet_param_list :
        
    duration :
         (Default value = np.inf)

    Returns
    -------

    """
    if isinstance(eventWise, str):
        eventWise_path = eventWise
        eventWise = Components.EventWise.from_file(eventWise_path)
    else:
        eventWise_path = os.path.join(eventWise.dir_name, eventWise.save_name)
    end_time = time.time() + duration
    print("Making global data")
    if "JetInputs_ShowerDistance" not in eventWise.columns:
        relatives = closest_relative(eventWise)
        eventWise.append(JetInputs_ShowerDistance=awkward.fromiter(relatives))
    else:
        relatives = eventWise.JetInputs_ShowerDistance
        relatives = [np.array(rel, dtype=int) for rel in relatives.tolist()]
    eventWise.selected_index = None
    if "JetInputs_PairLabels" not in eventWise.columns:
        labels = label_parings(eventWise)
        eventWise.append(JetInputs_PairLabels=awkward.fromiter(labels))
    else:
        labels = eventWise.JetInputs_PairLabels
        # these have to be made into lists, else they don't work as masks
        labels = [np.array(lab, dtype=bool) for lab in labels.tolist()]
    eventWise.selected_index = None
    if "JetInputs_PairCrossings" not in eventWise.columns:
        crossings = label_crossings(labels)
        eventWise.append(JetInputs_PairCrossings=awkward.fromiter(crossings))
    else:
        crossings = eventWise.JetInputs_PairCrossings
        # these have to be made into lists, else they don't work as masks
        crossings = [np.array(cross, dtype=bool) for cross in crossings.tolist()]
    # we only care about things taht are either in the b group or crossing over
    either = [np.logical_or(l, c) for l, c in zip(labels, crossings)]
    # the labels and relativesof the subset we care about
    either_labels = [l[e] for l, e in zip(labels, either)]
    either_relatives = [r[e] for r, e in zip(relatives, either)]
    num_configureations = len(jet_names)
    save_interval = 5
    print("Done with global data")
    new_content = {}
    new_hyper = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for i, (name, params) in enumerate(zip(jet_names, jet_param_list)):
            if time.time() > end_time:
                break
            print(f"{i/num_configureations:.1%} {name}" + " "*10, end='\r', flush=True)
            # add the hyper parameters
            for key in params:
                new_hyper[name+'_'+key] = params[key]
            # get the distances
            distance_name = name + "_PhysDistance"
            if distance_name not in eventWise.columns:
                eventWise.selected_index = None
                distances = physical_distances(eventWise.JetInputs_Phi,
                                               eventWise.JetInputs_Rapidity,
                                               eventWise.JetInputs_PT,
                                               params["ExpofPTMultiplier"])
                new_content[distance_name] = awkward.fromiter(distances)
            else:
                eventWise.selected_index = None
                distances = [np.array(dis) for dis in getattr(eventWise, distance_name).tolist()]
            # create the scores
            rank_name = name + "_DistancePhysRank"
            auc_name = name + "_LabelPhysAUC"
            if rank_name not in eventWise.columns or auc_name not in eventWise.columns or True:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    ranks = []; aucs = []
                    for i, dis in enumerate(distances):
                        relevent = dis[:, either[i]]
                        rel = either_relatives[i].flatten()
                        lab = ~either_labels[i].flatten()
                        ranks.append([scipy.stats.spearmanr(metric.flatten(), rel)[0]
                                      for metric in relevent])
                        try:
                            aucs.append([sklearn.metrics.roc_auc_score(lab, metric.flatten())
                                         for metric in relevent])
                        except ValueError:  # means there is only one class going
                            aucs.append([np.nan for metric in relevent])
                new_content[rank_name] = awkward.fromiter(ranks)
                new_content[auc_name] = awkward.fromiter(aucs)
            if (i+5)%save_interval == 0 and new_content:
                eventWise.append_hyperparameters(**new_hyper)
                new_hyper = {}
                eventWise.append(**new_content)
                new_content = {}
                # delete and reload the eventWise
                del eventWise
                eventWise = Components.EventWise.from_file(eventWise_path)
            # to keep memory requirments down, del everything
            del distances
    eventWise.append(**new_content)
    eventWise.append_hyperparameters(**new_hyper)

# doesn't really work great
def plot_phys_event(eventWise, event_num, metric_names=None, jet_names=None):
    """
    

    Parameters
    ----------
    eventWise :
        
    event_num :
        
    *jet_names :
        

    Returns
    -------

    """
    if jet_names is None:
        jet_names = [name.split('_')[0] for name in eventWise.columns
                     if name.endswith("_PhysDistance")]
        jet_names = jet_names[::2]
    if metric_names is None:
        metric_names = []
        name = True
        while name:
            name = InputTools.list_complete("Chose a metric (empty to stop); ", metrics.keys())
            name = name.strip()
            metric_names.append(name)
        del metric_names[-1]
    num_jets = len(jet_names)
    # get global data
    eventWise.selected_index = event_num
    phis = eventWise.JetInputs_Phi
    y_lims = np.min(phis), np.max(phis)
    rapidities = eventWise.JetInputs_Rapidity
    x_lims = np.min(rapidities), np.max(rapidities)
    same_mask = np.array(eventWise.JetInputs_PairLabels.tolist())
    cross_mask = np.array(eventWise.JetInputs_PairCrossings.tolist())
    colours = np.zeros((len(same_mask), len(same_mask[0]), 4), dtype=float)
    colours += 0.3
    colours[same_mask] = [0.1, 1., 0.1, 0.]
    colours[cross_mask] = [1., 0.1, 0., 0.]
    colours[:, :, -1] = same_mask.astype(float)*0.5 + cross_mask.astype(float)*0.5 + 0.2
    # make a grid of axis for each jet name and each metric
    num_metrics = len(metric_names)
    fig, ax_arr = plt.subplots(num_jets, num_metrics, sharex=True, sharey=True)
    ax_arr = ax_arr.reshape((num_jets, num_metrics))
    # now the other axis should contain the plots
    metric_order = list(metrics.keys())
    for jet_n, jet_name in enumerate(jet_names):
        distances = getattr(eventWise, jet_name + "_PhysDistance")
        # normalise the distances
        distances = distances/np.nanmean(distances.tolist(), axis=(1, 2))
        ratios = getattr(eventWise, jet_name + "_DifferencePhysDistance")
        for metric_n, metric in enumerate(metric_names):
            metric_pos = metric_order.index(metric)
            ax = ax_arr[jet_n, metric_n]
            for i1, (p1, r1) in enumerate(zip(phis, rapidities)):
                for i2, (p2, r2) in enumerate(zip(phis, rapidities)):
                    width = distances[metric_pos, i1, i2]
                    line = matplotlib.lines.Line2D([r1, r2], [p1, p2],
                                                   c=colours[i1, i1],
                                                   lw=width)
                    ax.add_line(line)
            if jet_n == num_jets-1:
                ax.set_xlabel(metric)
            if metric_n == 0:
                ax.set_ylabel(jet_name)
            ax.set_xlim(*x_lims)
            ax.set_ylim(*y_lims)
    fig.set_size_inches(num_metrics*3.5, num_jets*1.8)
    #fig.tight_layout()
    fig.subplots_adjust(hspace=0.0, wspace=0., right=1., top=1.)


def plot_phys_overall(eventWise, *jet_names):
    if not jet_names:
        jet_names = [name.split('_')[0] for name in eventWise.columns
                     if name.endswith("_DifferencePhysDistance")]
        # sort the jet names
        prefix = len("ExpofPT")
        jet_nums = [float(name[prefix:].replace('p', '.').replace('m','-')) for name in jet_names]
        jet_names = np.array(jet_names)[np.argsort(jet_nums)]
    num_jets = len(jet_names)
    # get the scores
    num_metrics = len(phys_metric_names)
    # can be resused for each plot
    scores = np.empty((num_jets, num_metrics), dtype=float)
    eventWise.selected_index = None
    score_types = {"ROC-AUC": "_LabelPhysAUC", 
                   "Shower ranks": "_DistancePhysRank",
                   "Normed difference": "_DifferencePhysDistance"}
    num_scores = len(score_types)
    fig, ax_arr = plt.subplots(num_scores, 1)
    for score_name, ax in zip(score_types, ax_arr.flatten()):
        for jet_n, name in enumerate(jet_names):
            score = np.array(getattr(eventWise, name + score_types[score_name]).tolist())
            score[np.isnan(score)] = 0.
            scores[jet_n] = np.mean(score, axis=0)
        ax.set_title(score_name)
        img = ax.imshow(scores)
        plt.colorbar(mappable=img, ax=ax)
        ax.set_xticks([])
        ax.set_yticks(np.arange(num_jets))
        # ... and label them with the respective list entries
        ax.set_yticklabels(jet_names)
        ax.set_ylim(-0.5, num_jets-0.5)
        ax.set_xlim(-0.5, num_metrics-0.5)


        # Loop over data dimensions and create text annotations.
        for metric_n in range(num_metrics):
            for jet_n in range(num_jets):
                text = ax.text(metric_n, jet_n, f"{scores[jet_n, metric_n]:.2g}",
                               ha="center", va="center", color="k", fontsize='smaller')
    # label the last x axis only
    ax.set_xticks(np.arange(num_metrics))
    ax.set_xticklabels(phys_metric_names)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fig.set_size_inches(9, 7*num_scores)


# Eigenspace distance ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def create_eigenvectors(eventWise, jet_params):
    """
    Create and return the initial eigenspace and eigenvectors.
    The jet params relevent to this are;
    PhyDistance, AffinityType, AffinityCutoff, Laplacien, ExpOfPTMultiplier
    SpectralMean will be used.

    Parameters
    ----------
    eventWise : EventWise
        Data set containing particle data.
        
    jet_params : dict
        Input parameter choices

    Returns
    -------
    eigenvalues : list of numpy arrays of floats
        Non trivial eigenvalues for inital eigenspace

    eigenvectors : list of numpy arrays of floats
        Non trivial eigenvectors for inital eigenspace

    """
    eventWise.selected_index = None
    eigenvectors = []; eigenvalues = []
    for event_n in range(len(eventWise.X)):
        eventWise.selected_index = event_n
        jets = FormJets.SpectralMean(eventWise, assign=False, dict_jet_params=jet_params)
        eigenvalues.append(np.array(jets.eigenvalues).flatten())
        eigenvectors.append(np.copy(jets.eigenvectors))
        del jets
    return eigenvalues, eigenvectors


def append_eig_metrics(eventWise, jet_names, jet_param_list, duration=np.inf):
    """
    

    Parameters
    ----------
    eventWise :
        
    jet_names :
        
    jet_param_list :
        
    duration :
         (Default value = np.inf)

    Returns
    -------

    """
    if isinstance(eventWise, str):
        eventWise_path = eventWise
        eventWise = Components.EventWise.from_file(eventWise_path)
    else:
        eventWise_path = os.path.join(eventWise.dir_name, eventWise.save_name)
    end_time = time.time() + duration
    print("Making global data")
    if "JetInputs_PairLabels" not in eventWise.columns:
        labels = label_parings(eventWise)
        relatives = closest_relative(eventWise)
        eventWise.append(JetInputs_PairLabels=awkward.fromiter(labels), 
                         JetInputs_ShowerDistance=awkward.fromiter(relatives))
    else:
        labels = eventWise.JetInputs_PairLabels
        relatives = eventWise.JetInputs_ShowerDistance
    num_configureations = len(jet_names)
    save_interval = 5
    print("Done with global data")
    new_content = {}
    new_hyper = {}
    for i, (name, params) in enumerate(zip(jet_names, jet_param_list)):
        if time.time() > end_time:
            break
        print(f"{i/num_configureations:.1%} {name}" + " "*10, end='\r', flush=True)
        # add the hyper parameters
        for key in params:
            new_hyper[name+'_'+key] = params[key]
        # get the eigenspace
        eigenvalue_name = name + "_InitialEigenvalues"
        eigenvector_name = name + "_InitialEigenvectors"
        if eigenvalue_name not in eventWise.columns:
            values, vectors = create_eigenvectors(eventWise, params)
            new_content[eigenvalue_name] = awkward.fromiter(values)
            new_content[eigenvector_name] = awkward.fromiter(vectors)
        else:
            eventWise.selected_index = None
            values = getattr(eventWise, eigenvalue_name)
            vectors = getattr(eventWise, eigenvector_name)
        # get the seperations
        seperation_name = name + "_EigSeperation"
        if seperation_name not in eventWise.columns:
            if not isinstance(vectors, list):
                # scipy distances cannot deal with jagged arrays
                new_vectors = []
                no_particles = np.empty((0, 0))
                for vs in vectors:
                    if len(vs):
                        new_vectors.append(np.array(vs.tolist()))
                    else:
                        new_vectors.append(no_particles)
                vectors = new_vectors
                del new_vectors
            seperations = get_seperations(vectors, values)
            new_content[seperation_name] = awkward.fromiter(seperations)
        else:
            eventWise.selected_index = None
            seperations = getattr(eventWise, seperation_name)
        # create the scores
        rank_name = name + "_DistanceEigRank"
        auc_name = name + "_LabelEigAUC"
        if rank_name not in eventWise.columns:
            # suppress warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                ranks = []; aucs = []
                for i, sep in enumerate(seperations):
                    rel = relatives[i].flatten()
                    lab = ~labels[i].flatten()
                    ranks.append([scipy.stats.spearmanr(metric.flatten(), rel)[0]
                                  for metric in sep])
                    try:
                        aucs.append([sklearn.metrics.roc_auc_score(lab, metric.flatten())
                                     for metric in sep])
                    except ValueError:  # means there is only one class going
                        aucs.append([np.nan for metric in sep])
            new_content[name + "_DistanceEigRank"] = awkward.fromiter(ranks)
            new_content[name + "_LabelEigAUC"] = awkward.fromiter(aucs)
        if (i+5)%save_interval == 0 and new_content:
            eventWise.append_hyperparameters(**new_hyper)
            new_hyper = {}
            eventWise.append(**new_content)
            new_content = {}
            # delete and reload the eventWise
            del eventWise
            eventWise = Components.EventWise.from_file(eventWise_path)
        # to keep memory requirments down, del everything
        del vectors
        del values
        del seperations
    eventWise.append(**new_content)
    eventWise.append_hyperparameters(**new_hyper)


def plot_eig_event(eventWise, event_num, *jet_names):
    """
    

    Parameters
    ----------
    eventWise :
        
    event_num :
        
    *jet_names :
        

    Returns
    -------

    """
    if not jet_names:
        jet_names = [name.split('_')[0] for name in eventWise.columns
                     if name.endswith("_EigSeperation")]
    num_jets = len(jet_names)
    # give each jet a colour
    jet_map = matplotlib.cm.get_cmap('gist_rainbow')
    jet_colours = [jet_map((i+0.5)/num_jets) for i in range(num_jets)]
    # get global data
    eventWise.selected_index = event_num
    label_mask = eventWise.JetInputs_PairLabels.flatten()
    label_markers = ['o', 'v']
    distances = eventWise.JetInputs_ShowerDistance.flatten()
    # going to plot each seperation on it's own axis against the shower distances
    # represent the jets as colours and the labels as shapes
    # put the rank and the auc in the legend
    # the jets will be tablulated
    num_metrics = len(eig_metric_names)
    n_rows = int(np.ceil(num_metrics/2+1))
    n_cols = 4
    fig, ax_arr = plt.subplots(n_rows, n_cols, sharex='col')
    plot_arr = np.vstack((ax_arr[1:, :2], ax_arr[1:, 2:]))
    # add the table
    table_ax = ax_arr[0, 0]
    PlottingTools.make_inputs_table(eventWise, jet_names, table_ax)
    for blank_ax in ax_arr[0, 1:]:
        PlottingTools.hide_axis(blank_ax)
    # now the other axis should contain the plots
    marker_size = 10
    edge_width = 0.2
    for metric_n, metric in enumerate(eig_metric_names):
        legend_ax, plot_ax = plot_arr[metric_n]
        for name, colour in zip(jet_names, jet_colours):
            seperations = getattr(eventWise, name + "_EigSeperation")[metric_n].flatten()
            rank = getattr(eventWise, name+"_DistanceEigRank")[metric_n]
            auc = getattr(eventWise, name+"_LabelEigAUC")[metric_n]
            plot_ax.scatter(distances[label_mask], seperations[label_mask], s=marker_size,
                    alpha=0.5, 
                    c=[colour], marker=label_markers[1], edgecolors='k', linewidths=edge_width,
                    label=f"{name} same quark.{os.linesep}Rank {rank:.3g}. Auc {auc:.3g}.")
            plot_ax.scatter(distances[~label_mask], seperations[~label_mask], s=marker_size,
                            alpha=0.5,
                            c=[colour], marker=label_markers[0], label=name+" diferent quarks")
        plot_ax.set_ylabel(metric)
        legend_ax.legend(*plot_ax.get_legend_handles_labels(), loc='center')
        PlottingTools.hide_axis(legend_ax)
    plot_ax.set_xlabel("Shower distance")
    fig.set_size_inches(n_cols*3.5, n_rows*1.8)
    #fig.tight_layout()
    fig.subplots_adjust(hspace=0.0, left=0, right=1., top=1., bottom=0.05)


def plot_eig_overall(eventWise, *jet_names):
    if not jet_names:
        jet_names = [name.split('_')[0] for name in eventWise.columns
                     if name.endswith("_EigSeperation")]
    num_jets = len(jet_names)
    # give each jet a colour
    jet_map = matplotlib.cm.get_cmap('gist_rainbow')
    jet_colours = [jet_map((i+0.5)/num_jets) for i in range(num_jets)]
    # set up the axis
    fig, ax_arr = plt.subplots(2, 2)
    PlottingTools.hide_axis(ax_arr[0, 1])
    PlottingTools.make_inputs_table(eventWise, jet_names, ax_arr[0][0])
    # get the scores
    num_metrics = len(eig_metric_names)
    auc_scores = np.empty((num_jets, num_metrics), dtype=float)
    rank_scores = np.empty((num_jets, num_metrics), dtype=float)
    eventWise.selected_index = None
    for jet_n, name in enumerate(jet_names):
        aucs = np.array(getattr(eventWise, name + '_LabelEigAUC').tolist())
        ranks = np.array(getattr(eventWise, name + '_DistanceEigRank').tolist())
        auc_scores[jet_n] = np.nanmean(aucs, axis=0)
        rank_scores[jet_n] = np.nanmean(ranks, axis=0)
    # title the heatmaps
    ax_arr[1, 0].set_title("AUC from labels")
    ax_arr[1, 1].set_title("Rank from shower distances")
    # plot as a heatmap
    for ax, score in zip(ax_arr[1], [auc_scores, rank_scores]):
        img = ax.imshow(score)
        plt.colorbar(mappable=img, ax=ax)
        
        ax.set_xticks(np.arange(num_metrics))
        ax.set_yticks(np.arange(num_jets))
        # ... and label them with the respective list entries
        ax.set_xticklabels(eig_metric_names)
        ax.set_yticklabels(jet_names)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        height = num_jets/2
        for metric_n in range(num_metrics):
            mean = np.nanmean(score[:, metric_n])
            text = ax.text(metric_n, height, f"ave={mean:.3g}",
                           ha="center", va="center", color="k",
                           rotation=90.)

    fig.set_size_inches(10, 8)


def eig_jets():
    jet_names = []
    jet_params = []
    jet_name = 'AngularExponent21'
    jet_ps = dict(PhyDistance='angular',
                      AffinityType='exponent2',
                      AffinityCutoff=None,
                      Laplacien='symmetric',
                      ExpOfPTMultiplier=0)
    jet_names.append(jet_name)
    jet_params.append(jet_ps)
    jet_name = 'AngularExponent1'
    jet_ps = dict(PhyDistance='angular',
                      AffinityType='exponent',
                      AffinityCutoff=None,
                      Laplacien='symmetric',
                      ExpOfPTMultiplier=0)
    jet_names.append(jet_name)
    jet_params.append(jet_ps)
    jet_name = 'LuclusExponent21'
    jet_ps = dict(PhyDistance='Luclus',
                      AffinityType='exponent2',
                      AffinityCutoff=None,
                      Laplacien='symmetric',
                      ExpOfPTMultiplier=0)
    jet_names.append(jet_name)
    jet_params.append(jet_ps)
    jet_name = 'LuclusExponent1'
    jet_ps = dict(PhyDistance='Luclus',
                      AffinityType='exponent',
                      AffinityCutoff=None,
                      Laplacien='symmetric',
                      ExpOfPTMultiplier=0)
    jet_names.append(jet_name)
    jet_params.append(jet_ps)
    jet_name = 'AngularExponent22'
    jet_ps = dict(PhyDistance='angular',
                      AffinityType='exponent2',
                      AffinityCutoff=None,
                      Laplacien='unnormalised',
                      ExpOfPTMultiplier=0)
    jet_names.append(jet_name)
    jet_params.append(jet_ps)
    jet_name = 'AngularExponent2'
    jet_ps = dict(PhyDistance='angular',
                      AffinityType='exponent',
                      AffinityCutoff=None,
                      Laplacien='unnormalised',
                      ExpOfPTMultiplier=0)
    jet_names.append(jet_name)
    jet_params.append(jet_ps)
    jet_name = 'LuclusExponent22'
    jet_ps = dict(PhyDistance='Luclus',
                      AffinityType='exponent2',
                      AffinityCutoff=None,
                      Laplacien='unnormalised',
                      ExpOfPTMultiplier=0)
    jet_names.append(jet_name)
    jet_params.append(jet_ps)
    jet_name = 'LuclusExponent2'
    jet_ps = dict(PhyDistance='Luclus',
                      AffinityType='exponent',
                      AffinityCutoff=None,
                      Laplacien='unnormalised',
                      ExpOfPTMultiplier=0)
    jet_names.append(jet_name)
    jet_params.append(jet_ps)
    jet_name = 'AngularExponent24'
    jet_ps = dict(PhyDistance='angular',
                      AffinityType='exponent2',
                      AffinityCutoff=('distance', 2),
                      Laplacien='symmetric',
                      ExpOfPTMultiplier=0)
    jet_names.append(jet_name)
    jet_params.append(jet_ps)
    jet_name = 'AngularExponent4'
    jet_ps = dict(PhyDistance='angular',
                      AffinityType='exponent',
                      AffinityCutoff=('distance', 2),
                      Laplacien='symmetric',
                      ExpOfPTMultiplier=0)
    jet_names.append(jet_name)
    jet_params.append(jet_ps)
    jet_name = 'LuclusExponent24'
    jet_ps = dict(PhyDistance='Luclus',
                      AffinityType='exponent2',
                      AffinityCutoff=('distance', 2),
                      Laplacien='symmetric',
                      ExpOfPTMultiplier=0)
    jet_names.append(jet_name)
    jet_params.append(jet_ps)
    jet_name = 'LuclusExponent4'
    jet_ps = dict(PhyDistance='Luclus',
                      AffinityType='exponent',
                      AffinityCutoff=('distance', 2),
                      Laplacien='symmetric',
                      ExpOfPTMultiplier=0)
    jet_names.append(jet_name)
    jet_params.append(jet_ps)
    jet_name = 'AngularExponent23'
    jet_ps = dict(PhyDistance='angular',
                      AffinityType='exponent2',
                      AffinityCutoff=('distance', 2),
                      Laplacien='unnormalised',
                      ExpOfPTMultiplier=0)
    jet_names.append(jet_name)
    jet_params.append(jet_ps)
    jet_name = 'AngularExponent3'
    jet_ps = dict(PhyDistance='angular',
                      AffinityType='exponent',
                      AffinityCutoff=('distance', 2),
                      Laplacien='unnormalised',
                      ExpOfPTMultiplier=0)
    jet_names.append(jet_name)
    jet_params.append(jet_ps)
    jet_name = 'LuclusExponent23'
    jet_ps = dict(PhyDistance='Luclus',
                      AffinityType='exponent2',
                      AffinityCutoff=('distance', 2),
                      Laplacien='unnormalised',
                      ExpOfPTMultiplier=0)
    jet_names.append(jet_name)
    jet_params.append(jet_ps)
    jet_name = 'LuclusExponent3'
    jet_ps = dict(PhyDistance='Luclus',
                      AffinityType='exponent',
                      AffinityCutoff=('distance', 2),
                      Laplacien='unnormalised',
                      ExpOfPTMultiplier=0)
    jet_names.append(jet_name)
    jet_params.append(jet_ps)
    return jet_names, jet_params


# affinity cutoff ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def get_isolated(is_linked, labels):
    isolated = []
    percent_isolated = []
    for linked, lab in zip(is_linked, labels):
        # particles in the b-jet have a positive diagonal
        in_bjet = np.diag(lab)
        # we are intrested in how many partices are in the same b-jet and have stayed linked
        # particles not in the b-jet are never considered isolated
        local = np.full_like(in_bjet, False)
        # having only one b jet particle is a special case,
        if np.sum(in_bjet) == 1:
            # if the b jet is linked to itself then it is not isolated
            local[in_bjet] = ~linked[in_bjet, in_bjet]
            # there is nothing else it would eb meaningful for it to be conencted to
        else:
            connected_in_b = np.logical_and(linked[in_bjet], lab[in_bjet])
            # each row of this will have at least one positive value,
            # due to the particles connection to itself
            # if it only has one positive value then it is isolated
            local[in_bjet] += np.sum(connected_in_b, axis=1) < 2
        isolated.append(local)
        percent_isolated.append(np.sum(local)/np.sum(in_bjet))
    return isolated, percent_isolated


def get_linked(eventWise, jet_params):
    n_events = len(eventWise.JetInputs_SourceIdx)
    is_linked = []
    percent_sparcity = []
    AffinityCutoff = jet_params["AffinityCutoff"]
    for event_n in range(n_events):
        eventWise.selected_index = event_n
        jets = FormJets.Traditional(eventWise, jet_params, assign=False)
        distances2 = jets._distances2
        # although distances2 can have a non-zero diagonal this is never meaningful
        np.fill_diagonal(distances2, 0.)
        # distances2 only has the lower triangle filled in
        upper_triangle = np.triu_indices_from(distances2)
        distances2[upper_triangle] = distances2.T[upper_triangle]
        local = np.ones_like(distances2, dtype=bool)
        if AffinityCutoff is None:
            pass # everything is linke
        elif AffinityCutoff[0] == 'knn':
            num_neigbours = AffinityCutoff[1]
            local[~FormJets.knn(distances2, num_neigbours)] = False
        elif AffinityCutoff[0] == 'distance':
            max_distance2 = AffinityCutoff[1]**2
            local[distances2 > max_distance2] = False
        else:
            raise NotImplementedError
        is_linked.append(local)
        percent_sparcity.append(np.sum(~is_linked[-1])/(len(distances2)**2))
    return is_linked, percent_sparcity


def append_cutoff_metrics(eventWise, jet_names, jet_param_list, duration=np.inf):
    """
    

    Parameters
    ----------
    eventWise :
        
    jet_names :
        
    jet_param_list :
        
    duration :
         (Default value = np.inf)

    Returns
    -------

    """
    if isinstance(eventWise, str):
        eventWise_path = eventWise
        eventWise = Components.EventWise.from_file(eventWise_path)
    else:
        eventWise_path = os.path.join(eventWise.dir_name, eventWise.save_name)
    end_time = time.time() + duration
    print("Making global data")
    eventWise.selected_index = None
    if "JetInputs_PairLabels" not in eventWise.columns:
        labels = label_parings(eventWise)
        eventWise.append(JetInputs_PairLabels=awkward.fromiter(labels))
    else:
        labels = eventWise.JetInputs_PairLabels
        # these have to be made into lists, else they don't work as masks
        labels = [np.array(lab, dtype=bool) for lab in labels.tolist()]
    eventWise.selected_index = None
    if "JetInputs_PairCrossings" not in eventWise.columns:
        crossings = label_crossings(labels)
        eventWise.append(JetInputs_PairCrossings=awkward.fromiter(crossings))
    else:
        crossings = eventWise.JetInputs_PairCrossings
        # these have to be made into lists, else they don't work as masks
        crossings = [np.array(cross, dtype=bool) for cross in crossings.tolist()]
    eventWise.selected_index = None
    num_configureations = len(jet_names)
    save_interval = 5
    print("Done with global data")
    new_content = {}
    new_hyper = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for i, (name, params) in enumerate(zip(jet_names, jet_param_list)):
            if time.time() > end_time:
                break
            print(f"{i/num_configureations:.1%} {name}" + " "*10, end='\r', flush=True)
            # add the hyper parameters
            for key in params:
                new_hyper[name+'_'+key] = params[key]
            # get the links
            linked_name = name + "_Linked"
            sparcity_name = name + "_Sparcity"
            if linked_name not in eventWise.columns:
                eventWise.selected_index = None
                is_linked, percent_sparcity = get_linked(eventWise, params)
                new_content[linked_name] = awkward.fromiter(is_linked)
                new_content[sparcity_name] = awkward.fromiter(percent_sparcity)
            else:
                eventWise.selected_index = None
                is_linked = [np.array(linked) for linked in
                             getattr(eventWise, linked_name).tolist()]
                percent_sparcity = getattr(eventWise, sparcity_name).tolist()
            # get the isolated parts
            isolated_name = name + "_Isolated"
            pisolated_name = name + "_PercentIsolated"
            if isolated_name not in eventWise.columns:
                eventWise.selected_index = None
                isolated, percent_isolated = get_isolated(is_linked, labels)
                new_content[isolated_name] = awkward.fromiter(isolated)
                new_content[pisolated_name] = awkward.fromiter(percent_isolated)
            else:
                eventWise.selected_index = None
                isolated = [np.array(isolated) for isolated in
                            getattr(eventWise, isolated_name).tolist()]
                percent_isolated = getattr(eventWise, pisolated_name).tolist()
            if (i+5)%save_interval == 0 and new_content:
                eventWise.append_hyperparameters(**new_hyper)
                new_hyper = {}
                eventWise.append(**new_content)
                new_content = {}
                # delete and reload the eventWise
                del eventWise
                eventWise = Components.EventWise.from_file(eventWise_path)
            # to keep memory requirments down, del everything
            del percent_isolated
            del percent_sparcity
            del isolated
            del is_linked
    eventWise.append(**new_content)
    eventWise.append_hyperparameters(**new_hyper)


def plot_cutoff_event(eventWise, event_num, jet_names=None):
    """
    

    Parameters
    ----------
    eventWise :
        
    event_num :
        
    *jet_names :
        

    Returns
    -------

    """
    if jet_names is None:
        jet_names = [name.split('_')[0] for name in eventWise.columns
                     if name.endswith("_AffinityCutoff")]
        #jet_names = jet_names[::2]
    # get global data
    eventWise.selected_index = event_num
    phis = eventWise.JetInputs_Phi
    y_lims = np.min(phis), np.max(phis)
    rapidities = eventWise.JetInputs_Rapidity
    x_lims = np.min(rapidities), np.max(rapidities)
    same_mask = np.array(eventWise.JetInputs_PairLabels.tolist())
    cross_mask = np.array(eventWise.JetInputs_PairCrossings.tolist())
    colours = np.zeros((len(same_mask), len(same_mask[0]), 4), dtype=float)
    colours += 0.3
    colours[same_mask] = [0.1, 1., 0.1, 0.]
    colours[cross_mask] = [1., 0.1, 0., 0.]
    colours[:, :, -1] = same_mask.astype(float)*0.5 + cross_mask.astype(float)*0.5 + 0.2
    # make a grid of axis for the jets
    num_jets = len(jet_names)
    n_rows = int(np.floor(np.sqrt(num_jets)))
    n_cols = int(np.ceil(num_jets/n_rows))
    fig, ax_arr = plt.subplots(1+n_rows, n_cols, sharex=True, sharey=True)
    # use the first row to discribe the jets
    jet_inputs = ["PhyDistance", "ExpOfPTMultiplier", "AffinityCutoff"]
    PlottingTools.make_inputs_table(eventWise, jet_names, ax_arr[0, 0], jet_inputs)
    for blank_ax in ax_arr[0, 1:]:
        PlottingTools.hide_axis(blank_ax)
    jets_axis = ax_arr[1:].flatten()
    # now the other axis should contain the plots
    for jet_n, jet_name in enumerate(jet_names):
        ax = jets_axis[jet_n]
        is_linked = getattr(eventWise, jet_name+"_Linked")
        # draw the connections that have not been dropped.
        for i1, (p1, r1) in enumerate(zip(phis, rapidities)):
            for i2, (p2, r2) in enumerate(zip(phis, rapidities)):
                if is_linked[i1, i2]:
                    line = matplotlib.lines.Line2D([r1, r2], [p1, p2],
                                                   c=colours[i1, i1])
                    ax.add_line(line)
        # put the isolated points on in red
        isolated = getattr(eventWise, jet_name+"_Isolated")
        ax.scatter(rapidities[isolated], phis[isolated], c='r')
        ax.set_xlim(*x_lims)
        ax.set_ylim(*y_lims)
        ax.text(0, 0, jet_name, fontdict={"family": 'monospace'})
    fig.set_size_inches(n_rows*3.5, n_cols*1.8)
    #fig.tight_layout()
    fig.subplots_adjust(hspace=0.0, wspace=0., right=1., top=1.)


def cutoff_jets():
    jet_names = []
    jet_params = []
    # put the things to be iterated over into a fixed order
    fix_parameters = dict(ExpofPTPosition='input', ExpofPTFormat='Luclus')
    scan_parameters = dict(PhyDistance=['angular', 'taxicab'],
                           ExpOfPTMultiplier=np.linspace(-1, 1, 9),
                           AffinityCutoff=[None] + [('distance', x) for x in np.linspace(0.5, 7.5, 8)]
                                          +[('knn', x) for x in range(1, 6)])
    key_order = list(scan_parameters.keys())
    ordered_values = [scan_parameters[key] for key in key_order]
    num_combinations = np.product([len(vals) for vals in ordered_values])
    for i, combination in enumerate(itertools.product(*ordered_values)):
        #print(f"{i/num_combinations:.1%}", end='\r', flush=True)
        # check if it's been done
        parameters = {**dict(zip(key_order, combination)), **fix_parameters}
        jet_name = "ExpofPT" + \
                   str(parameters["ExpOfPTMultiplier"]).replace('.', 'p').replace('-', 'm') + \
                   parameters["PhyDistance"] + str(i)
        jet_names.append(jet_name)
        jet_params.append(parameters)
    return jet_names, jet_params


if __name__ == '__main__':
    eventWise = Components.EventWise.from_file("megaIgnore/cutoff.awkd")
    jet_names, jet_params = cutoff_jets()
    append_cutoff_metrics(eventWise, jet_names, jet_params)
    #exp_of_pt = np.linspace(-1, 1, 9)
    #jet_names = ["ExpofPT" + str(exp)[:4].replace('.', 'p').replace('-', 'm')
    #             for exp in exp_of_pt]
    #jet_params = [dict(ExpofPTMultiplier=exp) for exp in exp_of_pt]
    #append_phys_metrics(eventWise, jet_names, jet_params)
    print("Done")
    #plot_phys_event(eventWise, 0)
    #jinput()
    #plot_phys_overall(eventWise)
    #plot_eig_event(eventWise, 0, 'AngularExponent2', 'AngularExponent1', 'AngularExponent21')
    #plot_eig_overall(eventWise)
    #input()
    

