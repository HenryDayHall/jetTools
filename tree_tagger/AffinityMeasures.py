""" A module for comparing diferent affinity measures """
from ipdb import set_trace as st
import numpy as np
from matplotlib import pyplot as plt
from tree_tagger import FormShower, PlottingTools, Components, InputTools
import scipy.spatial
import scipy.stats

def component_deltaR(rapidity, phi):
    rapidity_dist2 = scipy.spatial.distance.pdist(rapidity.reshape((-1, 1)), metric='sqeuclidean')
    if rapidity_dist2.shape[0] == 0:
        # no elements
        return rapidity_dist2
    phi_dist = scipy.spatial.distance.pdist(phi.reshape((-1,1)), metric='euclidean')
    phi_dist = Components.raw_to_angular_distance(phi_dist)
    affinities_grid = np.sqrt(phi_dist**2 + rapidity_dist2)
    affinities_grid = scipy.spatial.distance.squareform(affinities_grid)
    return affinities_grid

def deltaR(eventWise):
    assert eventWise.selected_index is not None
    affinities_grid = component_deltaR(eventWise.JetInputs_Rapidity, eventWise.JetInputs_Phi)
    return affinities_grid


def sum_deltaR_tohard(eventWise):
    assert eventWise.selected_index is not None
    hardest_idx = np.argmax(eventWise.JetInputs_PT)
    rapidity_dist2 = np.abs(eventWise.JetInputs_Rapidity - eventWise.JetInputs_Rapidity[hardest_idx])
    phi_dist = np.sqrt(np.abs(eventWise.JetInputs_Phi - eventWise.JetInputs_Phi[hardest_idx]))
    phi_dist = Components.raw_to_angular_distance(phi_dist)
    dist = np.sqrt(phi_dist**2 + rapidity_dist2).reshape((-1, 1))
    affinities_grid = scipy.spatial.distance.pdist(dist, metric=sum)
    affinities_grid = scipy.spatial.distance.squareform(affinities_grid)
    return affinities_grid


def sum_deltaR_tocentre(eventWise):
    assert eventWise.selected_index is not None
    centre = [np.sum(eventWise.JetInputs_Energy),
              np.sum(eventWise.JetInputs_Px),
              np.sum(eventWise.JetInputs_Py),
              np.sum(eventWise.JetInputs_Pz)]
    phi, pt = Components.pxpy_to_phipt(centre[1], centre[2])
    rapidity = Components.ptpze_to_rapidity(pt, centre[3], centre[0])
    rapidity_dist2 = np.abs(eventWise.JetInputs_Rapidity - rapidity)
    phi_dist = np.sqrt(np.abs(eventWise.JetInputs_Phi - phi))
    phi_dist = Components.raw_to_angular_distance(phi_dist)
    dist = np.sqrt(phi_dist**2 + rapidity_dist2).reshape((-1, 1))
    affinities_grid = scipy.spatial.distance.pdist(dist, metric=sum)
    affinities_grid = scipy.spatial.distance.squareform(affinities_grid)
    return affinities_grid


def get_CoM(eventWise):
    assert eventWise.selected_index is not None
    four_vectors = np.vstack((eventWise.JetInputs_Energy,
                              eventWise.JetInputs_Px,
                              eventWise.JetInputs_Py,
                              eventWise.JetInputs_Pz))
    invarient_masses2 = four_vectors[0]**2 - np.sum(four_vectors[1:]**2, axis=0)
    boost = -np.sum(four_vectors[1:], axis=1)
    new_three_vectors = four_vectors[1:] + boost.reshape((-1, 1))
    new_energies2 = invarient_masses2 + np.sum(new_three_vectors**2, axis=0)
    new_four_vectors = np.vstack((np.sqrt(new_energies2), new_three_vectors))
    return new_four_vectors


def CoM_deltaR(eventWise):
    assert eventWise.selected_index is not None
    four_vectors = np.vstack((eventWise.JetInputs_Energy,
                              eventWise.JetInputs_Px,
                              eventWise.JetInputs_Py,
                              eventWise.JetInputs_Pz))
    invarient_masses2 = four_vectors[0]**2 - np.sum(four_vectors[1:]**2, axis=0)
    boost = -np.sum(four_vectors[1:], axis=1)
    new_three_vectors = four_vectors[1:] + boost.reshape((-1, 1))
    new_energies2 = invarient_masses2 + np.sum(new_three_vectors**2, axis=0)
    new_four_vectors = np.vstack((np.sqrt(new_energies2), new_three_vectors))
    phi, pt = Components.pxpy_to_phipt(new_four_vectors[1], new_four_vectors[2])
    rapidity = Components.ptpze_to_rapidity(pt, new_four_vectors[3], new_four_vectors[0])
    rapidity_dist2 = scipy.spatial.distance.pdist(rapidity.reshape((-1, 1)), metric='sqeuclidean')
    phi_dist = scipy.spatial.distance.pdist(phi.reshape((-1, 1)), metric='euclidean')
    phi_dist = Components.raw_to_angular_distance(phi_dist)
    affinities_grid = np.sqrt(phi_dist**2 + rapidity_dist2)
    affinities_grid = scipy.spatial.distance.squareform(affinities_grid)
    return affinities_grid


def num_shared(array1, array2):
    return np.intersect1d(array1, array2).shape[0]


def mutual_neighbours(eventWise, num_neighbours=10):
    """ count of how many points are mututaly most proximate """
    assert eventWise.selected_index is not None
    total_particles = len(eventWise.JetInputs_SourceIdx)
    if total_particles < 2:
        return np.zeros((0, 0))
    proximites_grid = deltaR(eventWise)
    keep = num_neighbours + 1
    neighbours = np.array([np.argsort(row)[1:keep]
                           for row in proximites_grid])
    affinities_grid = scipy.spatial.distance.pdist(neighbours, num_shared)
    affinities_grid = scipy.spatial.distance.squareform(affinities_grid)
    affinities_grid /= total_particles
    return affinities_grid


def rank_shared(array1, array2):
    array1 = [x for x in array1 if x in array2]
    array2 = [x for x in array2 if x in array1]
    if not array1:
        return 0.
    if len(array1) == 1:
        return 1.
    score, _ = scipy.stats.spearmanr(array1, array2)
    return score

    
def ordered_neighbours(eventWise, num_neighbours=20):
    assert eventWise.selected_index is not None
    total_particles = len(eventWise.JetInputs_SourceIdx)
    if total_particles < 2:
        return np.zeros((0, 0))
    proximites_grid = deltaR(eventWise)
    keep = num_neighbours + 1
    neighbours = np.array([np.argsort(row)[1:keep]
                           for row in proximites_grid])
    percent_match = scipy.spatial.distance.pdist(neighbours, num_shared)/num_neighbours
    rank_score = scipy.spatial.distance.pdist(neighbours, rank_shared)
    affinities_grid = scipy.spatial.distance.squareform(percent_match*rank_score)
    return affinities_grid


def PT_shared(array1, array2, pts):
    return np.sum(pts[np.intersect1d(array1, array2)])


def mutual_neighbour_PT(eventWise, num_neighbours=10):
    assert eventWise.selected_index is not None
    total_particles = len(eventWise.JetInputs_SourceIdx)
    if total_particles < 2:
        return np.zeros((0, 0))
    proximites_grid = deltaR(eventWise)
    keep = num_neighbours + 1
    neighbours = np.array([np.argsort(row)[1:keep]
                           for row in proximites_grid])
    pts = eventWise.JetInputs_PT
    affinities_grid = scipy.spatial.distance.pdist(neighbours, PT_shared, pts=pts)
    affinities_grid = scipy.spatial.distance.squareform(affinities_grid)
    return affinities_grid


def closest_third(deltaRs1, deltaRs2):
    third_idxs = np.logical_and(deltaRs1 != 0, deltaRs2 != 0)
    return np.min(deltaRs1[third_idxs] + deltaRs2[third_idxs])


def thirdparty_distance(eventWise):
    assert eventWise.selected_index is not None
    total_particles = len(eventWise.JetInputs_SourceIdx)
    if total_particles < 3:
        return np.zeros((0, 0))
    proximites_grid = deltaR(eventWise)
    affinities_grid = scipy.spatial.distance.pdist(proximites_grid, closest_third)
    affinities_grid = scipy.spatial.distance.squareform(affinities_grid)
    return affinities_grid


def closest_third_diff(deltaRs1, deltaRs2):
    third_idxs = np.logical_and(deltaRs1 != 0, deltaRs2 != 0)
    idx = np.argmin(deltaRs1[third_idxs] + deltaRs2[third_idxs])
    sign = 1 - 2*(np.sum(deltaRs1) > np.sum(deltaRs2))
    return np.abs(deltaRs1[idx] - deltaRs2[idx])*sign


def thirdparty_distancediff(eventWise):
    assert eventWise.selected_index is not None
    total_particles = len(eventWise.JetInputs_SourceIdx)
    if total_particles < 3:
        return np.zeros((0, 0))
    proximites_grid = deltaR(eventWise)
    affinities_grid = scipy.spatial.distance.pdist(proximites_grid, closest_third_diff)
    affinities_grid = scipy.spatial.distance.squareform(affinities_grid)
    return affinities_grid


def area_ptratio(deltaRs1, deltaRs2, pts):
    mask1 = deltaRs1 != 0
    pt1 = np.sum(pts[mask1]/deltaRs1[mask1])
    mask2 = deltaRs2 != 0
    pt2 = np.sum(pts[mask2]/deltaRs2[mask2])
    if np.sum(deltaRs1) > np.sum(deltaRs2):
        return pt1/pt2
    else:
        return pt2/pt1


def localarea_ptratio(eventWise):
    assert eventWise.selected_index is not None
    total_particles = len(eventWise.JetInputs_SourceIdx)
    if total_particles < 3:
        return np.zeros((0, 0))
    proximites_grid = deltaR(eventWise)
    pts = eventWise.JetInputs_PT
    affinities_grid = scipy.spatial.distance.pdist(proximites_grid, area_ptratio, pts=pts)
    affinities_grid = scipy.spatial.distance.squareform(np.log(affinities_grid))
    return affinities_grid


def closest_third_anglediff(deltaRs1, deltaRs2, rapidities, phis, pts):
    idx1 = deltaRs1 == 0
    idx2 = deltaRs2 == 0
    third_idxs = np.logical_and(~idx1, ~idx2)
    closest1 = np.argmin(deltaRs1[third_idxs])
    closest2 = np.argmin(deltaRs2[third_idxs])
    position1 = np.array([rapidities[idx1], phis[idx1]]).flatten()
    position2 = np.array([rapidities[idx2], phis[idx2]]).flatten()
    connection = position1 - position2
    norm_connection = np.linalg.norm(connection)
    c_position1 = np.array([rapidities[third_idxs][closest1], phis[third_idxs][closest1]])
    c_line1 = c_position1 - position1
    cos_angle1 = np.dot(connection, c_line1)/(norm_connection*np.linalg.norm(c_line1))
    angle1 = np.arccos(cos_angle1)
    if angle1 > np.pi:
        angle1 = np.pi - angle1
    c_position2 = np.array([rapidities[third_idxs][closest2], phis[third_idxs][closest2]])
    c_line2 = c_position2 - position2
    cos_angle2 = np.dot(connection, c_line2)/(norm_connection*np.linalg.norm(c_line2))
    angle2 = np.arccos(cos_angle2)
    if angle2 > np.pi:
        angle2 = np.pi - angle2
    if pts[idx1] > pts[idx2]:
        return angle1 - angle2
    else:
        return angle2 - angle1


def thirdparty_anglediff(eventWise):
    assert eventWise.selected_index is not None
    total_particles = len(eventWise.JetInputs_SourceIdx)
    if total_particles < 3:
        return np.zeros((0, 0))
    proximites_grid = deltaR(eventWise)
    pts = eventWise.JetInputs_PT
    phis = eventWise.JetInputs_Phi
    rapidities = eventWise.JetInputs_Rapidity
    affinities_grid = scipy.spatial.distance.pdist(proximites_grid, closest_third_anglediff,
                                                   pts=pts, phis=phis, rapidities=rapidities)
    affinities_grid = scipy.spatial.distance.squareform(affinities_grid)
    return affinities_grid


def third_anglespreaddiff(deltaRs1, deltaRs2, rapidities, phis, pts):
    idx1 = deltaRs1 == 0
    idx2 = deltaRs2 == 0
    third_idxs = np.logical_and(~idx1, ~idx2)
    position1 = np.array([rapidities[idx1], phis[idx1]]).flatten()
    position2 = np.array([rapidities[idx2], phis[idx2]]).flatten()
    connection = position1 - position2
    norm_connection = np.linalg.norm(connection)
    angles1 = []
    angles2 = []
    for third in np.where(third_idxs)[0]:
        c_position1 = np.array([rapidities[third], phis[third]])
        c_line1 = c_position1 - position1
        cos_angle1 = np.dot(connection, c_line1)/(norm_connection*np.linalg.norm(c_line1))
        angles1.append(np.arccos(cos_angle1))
        c_position2 = np.array([rapidities[third], phis[third]])
        c_line2 = c_position2 - position2
        cos_angle2 = np.dot(connection, c_line2)/(norm_connection*np.linalg.norm(c_line2))
        angles2.append(np.arccos(cos_angle2))
    weights1 = deltaRs1[third_idxs]
    varience1 = np.sqrt(np.average((angles1 - np.average(angles1, weights=weights1))**2, weights=weights1))
    weights2 = deltaRs2[third_idxs]
    varience2 = np.sqrt(np.average((angles2 - np.average(angles2, weights=weights2))**2, weights=weights2))
    if pts[idx1] > pts[idx2]:
        return varience1 - varience2
    else:
        return varience2 - varience1


def thirdparty_anglespread(eventWise):
    assert eventWise.selected_index is not None
    total_particles = len(eventWise.JetInputs_SourceIdx)
    if total_particles < 3:
        return np.zeros((0, 0))
    proximites_grid = deltaR(eventWise)
    pts = eventWise.JetInputs_PT
    phis = eventWise.JetInputs_Phi
    rapidities = eventWise.JetInputs_Rapidity
    affinities_grid = scipy.spatial.distance.pdist(proximites_grid, third_anglespreaddiff,
                                                   pts=pts, phis=phis, rapidities=rapidities)
    affinities_grid = scipy.spatial.distance.squareform(affinities_grid)
    return affinities_grid


affinity_choices = {"deltaR": deltaR, "CoM_deltaR": CoM_deltaR,
                    "sum_deltaR_tohard": sum_deltaR_tohard, "sum_deltaR_tocentre": sum_deltaR_tocentre,
                    "mutual_neighbours": mutual_neighbours, "ordered_neighbours": ordered_neighbours, 
                    "mutual_neighbour_PT": mutual_neighbour_PT, "thirdparty_distance": thirdparty_distance,
                    "thirdparty_distancediff": thirdparty_distancediff, "localarea_ptratio": localarea_ptratio,
                    "thirdparty_anglediff": thirdparty_anglediff, "thirdparty_anglespread": thirdparty_anglespread}

# PLotting code

def bins_extent(values, max_bins=20):
    # values must be flat
    values = np.array(sorted(set(values)))
    extent = [np.min(values), np.max(values)]
    width = extent[1] - extent[0]
    bin_length = max(width/max_bins, np.min(values[1:] - values[:-1]))
    n_bins = int(np.ceil(width/bin_length))
    bins = np.linspace(*extent, n_bins+1)
    return bins, extent

def affinities_in_event(eventWise, affinity, results, affinity2=None, results2=None,
                        cluster_class=None, cluster_params=None):
    n_parts = 3
    sourceidx = eventWise.JetInputs_SourceIdx.tolist()
    if len(sourceidx) < 2:  # cannot clasisfy relatiosn without at least 2 particles
        return
    b_decendants = FormShower.decendant_idxs(eventWise, *eventWise.BQuarkIdx)
    b_decendants = [sourceidx.index(d) for d in b_decendants
                    if d in sourceidx]
    #b_decendants = []  # check the labeling
    grid = affinity(eventWise)
    catigory = np.zeros_like(grid, dtype=int)
    catigory[:, b_decendants] += 1
    catigory[b_decendants, :] += 1
    for i in range(n_parts):
        results[i] += grid[catigory == i].tolist()
    if affinity2 is not None:
        grid2 = affinity2(eventWise)
        for i in range(n_parts):
            results2[i] += grid2[catigory == i].tolist()


def plot_affinites(eventWise, affinity, affinity2=None, name=None, name2=None,
                   cluster_class=None, cluster_params=None, silent=False, bins=50):
    twoD = affinity2 is not None
    eventWise.selected_index = None
    n_events = len(eventWise.X)
    start_point = 0
    end_point = n_events
    both_b, both_not, cross = [], [], []
    parts = [both_not, cross, both_b]
    n_parts = len(parts)
    if twoD:
        parts2 = [[] for _ in range(n_parts)]
    else:
        parts2 = None
    if not silent:
        print(f" Will stop at {100*end_point/n_events}%")
    for event_n in range(start_point, end_point):
        if event_n % 10 == 0 and not silent:
            print(f"{100*event_n/n_events}%", end='\r', flush=True)
        eventWise.selected_index = event_n
        affinities_in_event(eventWise, affinity, parts, affinity2, parts2,
                            cluster_class, cluster_params)
    # plot
    labels = ['Both background', 'Crossover', 'Both b decendants']
    if name is not None:
        plt.xlabel(name)
    if twoD:
        bins, extent = bins_extent(np.sum(parts))
        bins2, extent2 = bins_extent(np.sum(parts2))
        extent += extent2
        # bin
        colours = np.zeros((len(bins2)-1, len(bins)-1, 3))
        fig, ax_ar = plt.subplots(1, n_parts+1)
        for i in range(n_parts):
            if len(parts[i]) == 0:
                continue
            density, x_edge, y_edge, binnumber =\
                    scipy.stats.binned_statistic_2d(parts2[i], parts[i], None,
                                                    statistic='count', bins=(bins2, bins))
            # make the density go from 0 to 1
            density /= np.max(density)
            mapable = ax_ar[i+1].imshow(density, origin='lower', extent=extent, aspect='auto')
            ax_ar[i+1].set_xlabel(name + " " + labels[i])
            ax_ar[i+1].set_ylabel(name2)
            colours[:, :, i] += density
        cbar = plt.colorbar(mapable)
        cbar.set_label("Density")
        ax_ar[0].imshow(colours, origin='lower', extent=extent,
                   aspect='auto')
        ax_ar[0].set_xlabel(name)
        ax_ar[0].set_ylabel(name2)
        for i, name in enumerate(labels):
            colour = [0, 0, 0]
            colour[i] = 1
            ax_ar[0].scatter([], [], marker=',', c=[colour], label=name)
        ax_ar[0].legend()
    else:
        # for number of bins
        bins, extent = bins_extent(np.sum(parts))
        plt.hist(parts,
                 bins=bins, histtype='step', density=True,
                 label=labels, range=extent)
        plt.ylabel('counts')
        plt.legend()


def apply_confusion(eventWise, jet_name, function, silent=False):
    eventWise.selected_index = None
    n_events = len(getattr(eventWise, jet_name + "_PT"))
    start_point = 0
    end_point = n_events
    output = []
    if not silent:
        print(f" Will stop at {100*end_point/n_events}%")
    for event_n in range(start_point, end_point):
        if event_n % 10 == 0 and not silent:
            print(f"{100*event_n/n_events}%", end='\r', flush=True)
        eventWise.selected_index = event_n
        jetinputs = eventWise.JetInputs_SourceIdx
        b_decendants = FormShower.decendant_idxs(eventWise, *eventWise.BQuarkIdx)
        bjet_jetidx = [i for i, tags in enumerate(getattr(eventWise, jet_name+"_Tags")) if len(tags)]
        bjet_child1 = getattr(eventWise, jet_name+"_Child1")[bjet_jetidx]
        bjet_inputidxs = getattr(eventWise, jet_name+"_InputIdx")[bjet_jetidx][bjet_child1 == -1]
        bjet = jetinputs[list(set(bjet_inputidxs.flatten()))]
        true_positives = b_decendants.intersection(bjet)
        false_positives = set(bjet) - b_decendants
        true_negatives = set(jetinputs) - set(bjet) - b_decendants
        false_negatives = b_decendants - set(bjet)
        out = function(eventWise, true_positives, false_positives, true_negatives, false_negatives)
        output.append(out)
    return output


def roc(eventWise, jet_name, silent=False):
    def function(ew, tp, fp, tn, fn):
        tp = len(tp)
        fp = len(fp)
        tn = len(tn)
        fn = len(fn)
        total = tp + fp + tn + fn
        positives = tp + fn
        negatives = fp + tn
        if total == 0:
            return [1., 0., 1.]
        percent_pos = (tp + tn)/total
        if positives > 0:
            tpr = tp/positives
        else:
            tpr = 1.
        if negatives > 0:
            fpr = fp/negatives
        else:
            fpr = 0.
        return [tpr, fpr, percent_pos]
    values = apply_confusion(eventWise, jet_name, function, silent)
    values = np.array(values)
    fig, ax = PlottingTools.discribe_jet(eventWise, jet_name)
    scatter = ax.scatter(values[:, 0], values[:, 1], c=values[:, 2], linewidth=0, alpha=0.5, label=jet_name)
    cbar = plt.colorbar(scatter)
    cbar.set_label("Percent b-decendants")
    ax.set_xlabel("True positive rate")
    ax.set_ylabel("False positive rate")
    ax.axis('scaled')


def kinematic_distributions(eventWise, jet_name, silent=False):
    def function(ew, tp, fp, tn, fn):
        tp = list(tp)
        fp = list(fp)
        fn = list(fn)
        tn = list(tn)
        var = ew.PT
        return [var[tp].tolist(), var[fp].tolist(), var[tn].tolist(), var[fn].tolist()]
    values = apply_confusion(eventWise, jet_name, function, silent)
    names = ["true positives", "false positive", "true negatives", "false negatives"]
    collected_values = [[] for _ in names]
    for line in values:
        collected_values = [col + new for col, new in zip(collected_values, line)]
    fig, ax = PlottingTools.discribe_jet(eventWise, jet_name)
    ax.hist(collected_values, label=names, bins=50, histtype='step')
    ax.legend()
    ax.set_xlabel("PT")
    ax.set_ylabel("counts") 
    ax.set_title(jet_name)

if __name__ == '__main__':
    file_name = InputTools.get_file_name("Name the eventWise; ")
    ew = Components.EventWise.from_file(file_name)
    name1 = InputTools.list_complete("Chose affinity 1; ", affinity_choices.keys()).strip()
    affinity1 = affinity_choices[name1]
    if InputTools.yesNo_question("Second affinity? "):
        name2 = InputTools.list_complete("Chose affinity 2; ", affinity_choices.keys()).strip()
        affinity2 = affinity_choices[name2]
    else:
        name2 = affinity2 = None
    plot_affinites(ew, affinity1, affinity2, name=name1, name2=name2)
    plt.show()


