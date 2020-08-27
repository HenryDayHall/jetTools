""" Module to investigate parameter choices, base on questions in writeup """
import warnings
from collections import OrderedDict
from ipdb import set_trace as st
import os
import numpy as np
from tree_tagger import FormJets, FormShower, PlottingTools, Components
import scipy.spatial
import scipy.stats
import sklearn.metrics
import time
import awkward
from matplotlib import pyplot as plt
import matplotlib


# Physical distance ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
        eigenvectors.append(np.copy(jets._eigenspace))
        del jets
    return eigenvalues, eigenvectors


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
                      Taxicab = dict(metric='cityblock'),
                      Braycurtis = dict(metric='braycurtis'),
                      Canberra = dict(metric='canberra'),
                      Min = dict(metric=min_sep),
                      Max = dict(metric='chebyshev'),
                      Correlation = dict(metric='correlation'),
                      Cosine = dict(metric='cosine'))
metric_names = list(metrics.keys()) + [name + " normed" for name in metrics]


def get_seperations(eigenvectors, eigenvalues):
    """
    For each pair of jet inputs in an event get the seperation.
    Each or the metrics in the dict will be tried,
    then retried with a normed eigenspace.
    A normed eigenspace is one where the eigenvectors have been divided
    by the eigenvalues.

    Parameters
    ----------
    eigenvectors : list of 2d numpy arrays of floats
        the eigenvectors of each event.
    eigenvectors : list of 1d numpy arrays of floats
        the eigenvalues of each event.

    Returns
    -------
    seperations : list of 3d numpy arrays of floats
        the distances in each event
        axis 0 is each of the metrics used
        axis 1 and 2 are the particles in each event

    """
    seperations = []
    n_metrics = len(metric_names)
    # suppress warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for vectors, values in zip(eigenvectors, eigenvalues):
            n_points = len(vectors)
            # make a array for the results
            local = np.zeros((n_metrics, n_points, n_points), dtype=float)
            # do without norming
            for i, name in enumerate(metrics):
                distance = scipy.spatial.distance.pdist(vectors, **metrics[name])
                local[i] = scipy.spatial.distance.squareform(distance)
            # now normalise the eigenvectors and go again
            vectors /= values
            after_norm = i+1
            for i, name in enumerate(metrics):
                distance = scipy.spatial.distance.pdist(vectors, **metrics[name])
                local[after_norm + i] = scipy.spatial.distance.squareform(distance)
            seperations.append(local)
    return seperations


def append_metrics(eventWise, jet_names, jet_param_list, duration=np.inf):
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
        eventWise.append(JetInputs_PairLabels=labels, JetInputs_ShowerDistance=relatives)
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
        if seperation_name not in eventWise.columns or True:
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
            new_content[seperation_name] = seperations
        else:
            eventWise.selected_index = None
            seperations = getattr(eventWise, seperation_name)
        # create the scores
        rank_name = name + "_DistanceEigRank"
        auc_name = name + "_LabelEigAUC"
        if rank_name not in eventWise.columns or True:
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
            eventWise = Components.EventWise.from_file(eventWise)
        # to keep memory requirments down, del everything
        del vectors
        del values
        del seperations
    eventWise.append(**new_content)
    eventWise.append_hyperparameters(**new_hyper)


def make_inputs_table(eventWise, jet_names, table_ax):
    jet_inputs = ["PhyDistance", "AffinityType", "AffinityCutoff",
                  "Laplacien", "ExpOfPTMultiplier"]
    # construct a text table
    table_content = [[" "] + jet_inputs]
    table_content += [[name] + [getattr(eventWise, name+'_'+inp) for inp in jet_inputs]
                      for name in jet_names]
    table_sep = '|'
    table = []
    cell_fmt = "17.17"
    for row in table_content:
        table.append([])
        for x in row:
            try:
                table[-1].append(f"{x:{cell_fmt}}")
            except ValueError:
                table[-1].append(f"{x:{cell_fmt[0]}}")
            except TypeError:
                table[-1].append(f"{str(x):{cell_fmt}}")
        table[-1] = table_sep.join(table[-1])
    table = os.linesep.join(table)
    table_ax.text(0, 0, table, fontdict={"family": 'monospace'})
    PlottingTools.hide_axis(table_ax)
    return table, jet_inputs


def plot_event(eventWise, event_num, *jet_names):
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
    num_metrics = len(metric_names)
    n_rows = int(np.ceil(num_metrics/2+1))
    n_cols = 4
    fig, ax_arr = plt.subplots(n_rows, n_cols, sharex='col')
    plot_arr = np.vstack((ax_arr[1:, :2], ax_arr[1:, 2:]))
    # add the table
    table_ax = ax_arr[0, 0]
    make_inputs_table(eventWise, jet_names, table_ax)
    for blank_ax in ax_arr[0, 1:]:
        PlottingTools.hide_axis(blank_ax)
    # now the other axis should contain the plots
    marker_size = 10
    edge_width = 0.2
    for metric_n, metric in enumerate(metric_names):
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


def plot_overall(eventWise, *jet_names):
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
    make_inputs_table(eventWise, jet_names, ax_arr[0][0])
    # get the scores
    num_metrics = len(metric_names)
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
        ax.set_xticklabels(metric_names)
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



if __name__ == '__main__':
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
    eventWise = Components.EventWise.from_file("megaIgnore/eigenspace.awkd")
    append_metrics(eventWise, jet_names, jet_params)
    print("Done")
    plot_event(eventWise, 0, 'AngularExponent2', 'AngularExponent1', 'AngularExponent21')
    plot_overall(eventWise)
    input()
    

