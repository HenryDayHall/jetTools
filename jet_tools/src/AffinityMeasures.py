""" A module for comparing diferent affinity measures """
import os
from ipdb import set_trace as st
import numpy as np
from matplotlib import pyplot as plt
from jet_tools.src import FormShower, PlottingTools, Components, InputTools, TrueTag, SingleFormJets
import scipy.spatial
import scipy.stats
import sklearn.metrics
import attrdict
import ast


# helper functions

def get_deltaR_grid(rapidity, phi):
    """
    

    Parameters
    ----------
    rapidity :
        
    phi :
        

    Returns
    -------

    """
    rapidity_dist2 = scipy.spatial.distance.pdist(rapidity.reshape((-1, 1)), metric='sqeuclidean')
    if rapidity_dist2.shape[0] == 0:
        # no elements
        return rapidity_dist2
    phi_dist = scipy.spatial.distance.pdist(phi.reshape((-1,1)), metric='euclidean')
    phi_dist = Components.raw_to_angular_distance(phi_dist)
    affinities_grid = np.sqrt(phi_dist**2 + rapidity_dist2)
    affinities_grid = scipy.spatial.distance.squareform(affinities_grid)
    return affinities_grid


def get_angle_grid(particle_collection):
    """
    

    Parameters
    ----------
    particle_collection :
        

    Returns
    -------

    """
    phi = particle_collection.JetInputs_Phi
    phi_dist = scipy.spatial.distance.pdist(phi.reshape((-1,1)))
    rapidity = particle_collection.JetInputs_Rapidity
    rapidity_dist = scipy.spatial.distance.pdist(rapidity.reshape((-1,1)))
    angle = np.arctan2(phi_dist, rapidity_dist)
    angle[angle>0.5*np.pi] = np.pi - angle[angle>np.pi]
    angle[angle<-0.5*np.pi] = np.pi + angle[angle>np.pi]
    return angle


def get_CoM_repr(particle_collection):
    """
    

    Parameters
    ----------
    particle_collection :
        

    Returns
    -------

    """
    four_vectors = np.vstack((particle_collection.JetInputs_Energy,
                              particle_collection.JetInputs_Px,
                              particle_collection.JetInputs_Py,
                              particle_collection.JetInputs_Pz))
    invarient_masses2 = four_vectors[0]**2 - np.sum(four_vectors[1:]**2, axis=0)
    boost = -np.sum(four_vectors[1:], axis=1)
    new_three_vectors = four_vectors[1:] + boost.reshape((-1, 1))
    new_energies2 = invarient_masses2 + np.sum(new_three_vectors**2, axis=0)
    new_four_vectors = np.vstack((np.sqrt(new_energies2), new_three_vectors))
    return new_four_vectors


def count_shared(array1, array2):
    """
    

    Parameters
    ----------
    array1 :
        
    array2 :
        

    Returns
    -------

    """
    return np.intersect1d(array1, array2).shape[0]


def score_rank_shared(array1, array2):
    """
    

    Parameters
    ----------
    array1 :
        
    array2 :
        

    Returns
    -------

    """
    array1 = [x for x in array1 if x in array2]
    array2 = [x for x in array2 if x in array1]
    if not array1:
        return 0.
    if len(array1) == 1:
        return 1.
    score, _ = scipy.stats.spearmanr(array1, array2)
    return score


def sum_shared(array1, array2, var):
    """
    

    Parameters
    ----------
    array1 :
        
    array2 :
        
    var :
        

    Returns
    -------

    """
    return np.sum(var[np.intersect1d(array1, array2)])


def find_closest_third(deltaRs1, deltaRs2):
    """
    

    Parameters
    ----------
    deltaRs1 :
        
    deltaRs2 :
        

    Returns
    -------

    """
    # a thing has 0 deltaR to itself
    third_idxs = np.logical_and(deltaRs1 != 0, deltaRs2 != 0)
    return np.min(deltaRs1[third_idxs] + deltaRs2[third_idxs])


def closest_third_diff(deltaRs1, deltaRs2):
    """
    

    Parameters
    ----------
    deltaRs1 :
        
    deltaRs2 :
        

    Returns
    -------

    """
    # a thing has 0 deltaR to itself
    third_idxs = np.logical_and(deltaRs1 != 0, deltaRs2 != 0)
    idx = np.argmin(deltaRs1[third_idxs] + deltaRs2[third_idxs])
    sign = 1 - 2*(np.sum(deltaRs1) > np.sum(deltaRs2))
    return np.abs(deltaRs1[idx] - deltaRs2[idx])*sign


def close_pt_ratio(deltaRs1, deltaRs2, pts):
    """
    

    Parameters
    ----------
    deltaRs1 :
        
    deltaRs2 :
        
    pts :
        

    Returns
    -------

    """
    mask1 = deltaRs1 != 0
    pt1 = np.sum(pts[mask1]/deltaRs1[mask1])
    mask2 = deltaRs2 != 0
    pt2 = np.sum(pts[mask2]/deltaRs2[mask2])
    if np.sum(deltaRs1) > np.sum(deltaRs2):
        return pt1/pt2
    else:
        return pt2/pt1


def closest_third_anglediff(deltaRs1, deltaRs2, rapidities, phis, pts):
    """
    

    Parameters
    ----------
    deltaRs1 :
        
    deltaRs2 :
        
    rapidities :
        
    phis :
        
    pts :
        

    Returns
    -------

    """
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


def closest_third_anglespreaddiff(deltaRs1, deltaRs2, rapidities, phis, pts):
    """
    

    Parameters
    ----------
    deltaRs1 :
        
    deltaRs2 :
        
    rapidities :
        
    phis :
        
    pts :
        

    Returns
    -------

    """
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



# affinity measures

def aff_deltaR(particle_collection):
    """
    

    Parameters
    ----------
    particle_collection :
        

    Returns
    -------

    """
    affinities_grid = get_deltaR_grid(particle_collection.JetInputs_Rapidity, particle_collection.JetInputs_Phi)
    return affinities_grid


def aff_deltaR_tohard(particle_collection):
    """
    

    Parameters
    ----------
    particle_collection :
        

    Returns
    -------

    """
    hardest_idx = np.argmax(particle_collection.JetInputs_PT)
    rapidity_dist2 = np.abs(particle_collection.JetInputs_Rapidity - particle_collection.JetInputs_Rapidity[hardest_idx])
    phi_dist = np.sqrt(np.abs(particle_collection.JetInputs_Phi - particle_collection.JetInputs_Phi[hardest_idx]))
    phi_dist = Components.raw_to_angular_distance(phi_dist)
    dist = np.sqrt(phi_dist**2 + rapidity_dist2).reshape((-1, 1))
    affinities_grid = scipy.spatial.distance.pdist(dist, metric=sum)
    affinities_grid = scipy.spatial.distance.squareform(affinities_grid)
    return affinities_grid


def aff_deltaR_tocentre(particle_collection):
    """
    

    Parameters
    ----------
    particle_collection :
        

    Returns
    -------

    """
    centre = [np.sum(particle_collection.JetInputs_Energy),
              np.sum(particle_collection.JetInputs_Px),
              np.sum(particle_collection.JetInputs_Py),
              np.sum(particle_collection.JetInputs_Pz)]
    phi, pt = Components.pxpy_to_phipt(centre[1], centre[2])
    rapidity = Components.ptpze_to_rapidity(pt, centre[3], centre[0])
    rapidity_dist2 = np.abs(particle_collection.JetInputs_Rapidity - rapidity)
    phi_dist = np.sqrt(np.abs(particle_collection.JetInputs_Phi - phi))
    phi_dist = Components.raw_to_angular_distance(phi_dist)
    dist = np.sqrt(phi_dist**2 + rapidity_dist2).reshape((-1, 1))
    affinities_grid = scipy.spatial.distance.pdist(dist, metric=sum)
    affinities_grid = scipy.spatial.distance.squareform(affinities_grid)
    return affinities_grid


def aff_horrizontal_from_line(particle_collection):
    """
    

    Parameters
    ----------
    particle_collection :
        

    Returns
    -------

    """
    dr = aff_deltaR(particle_collection)
    sdrtc = aff_deltaR_tocentre(particle_collection)
    line = -0.06848*dr*dr + 0.8278*dr + 0.6873
    return np.abs(line - sdrtc)


def aff_deltaR2_minus_displacement(particle_collection):
    """
    

    Parameters
    ----------
    particle_collection :
        

    Returns
    -------

    """
    dr = aff_deltaR(particle_collection)
    sdrtc = aff_deltaR_tocentre(particle_collection)
    line = -0.06848*dr*dr + 0.8278*dr + 0.6873
    return dr*dr - np.abs(line - sdrtc)


def aff_CoM_deltaR(particle_collection):
    """
    

    Parameters
    ----------
    particle_collection :
        

    Returns
    -------

    """
    four_vectors = np.vstack((particle_collection.JetInputs_Energy,
                              particle_collection.JetInputs_Px,
                              particle_collection.JetInputs_Py,
                              particle_collection.JetInputs_Pz))
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


def aff_mutual_ns(particle_collection, num_neighbours=10):
    """
    count of how many points are mututaly most proximate

    Parameters
    ----------
    particle_collection :
        
    num_neighbours :
         (Default value = 10)

    Returns
    -------

    """
    total_particles = len(particle_collection.JetInputs_Energy)
    if total_particles < 2:
        return np.zeros((0, 0))
    proximites_grid = aff_deltaR(particle_collection)
    keep = num_neighbours + 1
    neighbours = np.array([np.argsort(row)[1:keep]
                           for row in proximites_grid])
    affinities_grid = scipy.spatial.distance.pdist(neighbours, count_shared)
    affinities_grid = scipy.spatial.distance.squareform(affinities_grid)
    affinities_grid /= total_particles
    return affinities_grid

    
def aff_score_rank_ns(particle_collection, num_neighbours=20):
    """
    

    Parameters
    ----------
    particle_collection :
        
    num_neighbours :
         (Default value = 20)

    Returns
    -------

    """
    total_particles = len(particle_collection.JetInputs_Energy)
    if total_particles < 2:
        return np.zeros((0, 0))
    proximites_grid = aff_deltaR(particle_collection)
    keep = num_neighbours + 1
    neighbours = np.array([np.argsort(row)[1:keep]
                           for row in proximites_grid])
    percent_match = scipy.spatial.distance.pdist(neighbours, count_shared)/num_neighbours
    rank_score = scipy.spatial.distance.pdist(neighbours, score_rank_shared)
    affinities_grid = scipy.spatial.distance.squareform(percent_match*rank_score)
    return affinities_grid


def aff_mutual_ns_PT(particle_collection, num_neighbours=10):
    """
    

    Parameters
    ----------
    particle_collection :
        
    num_neighbours :
         (Default value = 10)

    Returns
    -------

    """
    total_particles = len(particle_collection.JetInputs_Energy)
    if total_particles < 2:
        return np.zeros((0, 0))
    proximites_grid = aff_deltaR(particle_collection)
    keep = num_neighbours + 1
    neighbours = np.array([np.argsort(row)[1:keep]
                           for row in proximites_grid])
    pts = particle_collection.JetInputs_PT
    affinities_grid = scipy.spatial.distance.pdist(neighbours, sum_shared, var=pts)
    affinities_grid = scipy.spatial.distance.squareform(affinities_grid)
    return affinities_grid


def aff_thirdp_distance(particle_collection):
    """
    

    Parameters
    ----------
    particle_collection :
        

    Returns
    -------

    """
    total_particles = len(particle_collection.JetInputs_Energy)
    if total_particles < 3:
        return np.zeros((0, 0))
    proximites_grid = aff_deltaR(particle_collection)
    affinities_grid = scipy.spatial.distance.pdist(proximites_grid, find_closest_third)
    affinities_grid = scipy.spatial.distance.squareform(affinities_grid)
    return affinities_grid


def aff_thirdp_distancediff(particle_collection):
    """
    

    Parameters
    ----------
    particle_collection :
        

    Returns
    -------

    """
    total_particles = len(particle_collection.JetInputs_Energy)
    if total_particles < 3:
        return np.zeros((0, 0))
    proximites_grid = aff_deltaR(particle_collection)
    affinities_grid = scipy.spatial.distance.pdist(proximites_grid, closest_third_diff)
    affinities_grid = scipy.spatial.distance.squareform(affinities_grid)
    return affinities_grid


def aff_close_pt_ratio(particle_collection):
    """
    

    Parameters
    ----------
    particle_collection :
        

    Returns
    -------

    """
    total_particles = len(particle_collection.JetInputs_Energy)
    if total_particles < 3:
        return np.zeros((0, 0))
    proximites_grid = aff_deltaR(particle_collection)
    pts = particle_collection.JetInputs_PT
    affinities_grid = scipy.spatial.distance.pdist(proximites_grid, close_pt_ratio, pts=pts)
    affinities_grid = scipy.spatial.distance.squareform(np.log(affinities_grid))
    return affinities_grid


def aff_thirdp_anglediff(particle_collection):
    """
    

    Parameters
    ----------
    particle_collection :
        

    Returns
    -------

    """
    total_particles = len(particle_collection.JetInputs_Energy)
    if total_particles < 3:
        return np.zeros((0, 0))
    proximites_grid = aff_deltaR(particle_collection)
    pts = particle_collection.JetInputs_PT
    phis = particle_collection.JetInputs_Phi
    rapidities = particle_collection.JetInputs_Rapidity
    affinities_grid = scipy.spatial.distance.pdist(proximites_grid, closest_third_anglediff,
                                                   pts=pts, phis=phis, rapidities=rapidities)
    affinities_grid = scipy.spatial.distance.squareform(affinities_grid)
    return affinities_grid


def aff_thirdp_anglespread(particle_collection):
    """
    

    Parameters
    ----------
    particle_collection :
        

    Returns
    -------

    """
    total_particles = len(particle_collection.JetInputs_Energy)
    if total_particles < 3:
        return np.zeros((0, 0))
    proximites_grid = aff_deltaR(particle_collection)
    pts = particle_collection.JetInputs_PT
    phis = particle_collection.JetInputs_Phi
    rapidities = particle_collection.JetInputs_Rapidity
    affinities_grid = scipy.spatial.distance.pdist(proximites_grid, closest_third_anglespreaddiff,
                                                   pts=pts, phis=phis, rapidities=rapidities)
    affinities_grid = scipy.spatial.distance.squareform(affinities_grid)
    return affinities_grid


affinity_choices = {"deltaR": aff_deltaR,
                    "deltaR_tohard": aff_deltaR_tohard,
                    "deltaR_tocentre": aff_deltaR_tocentre,
                    "horrizontal_from_line": aff_horrizontal_from_line,
                    "deltaR2_minus_displacement": aff_deltaR2_minus_displacement,
                    "CoM_deltaR": aff_CoM_deltaR,
                    "mutual_ns": aff_mutual_ns,
                    "score_rank_ns": aff_score_rank_ns,
                    "mutual_ns_PT": aff_mutual_ns_PT,
                    "thirdp_distance": aff_thirdp_distance,
                    "thirdp_distancediff": aff_thirdp_distancediff,
                    "close_pt_ratio": aff_close_pt_ratio,
                    "thirdp_anglediff": aff_thirdp_anglediff,
                    "thirdp_anglespread": aff_thirdp_anglespread}

# from paper

def aff_original_JADE(particle_collection):
    """
    

    Parameters
    ----------
    particle_collection :
        

    Returns
    -------

    """
    four_vectors = np.vstack((particle_collection.JetInputs_Energy,
                              particle_collection.JetInputs_Px,
                              particle_collection.JetInputs_Py,
                              particle_collection.JetInputs_Pz)).T
    combined = scipy.spatial.distance.pdist(four_vectors, sum)
    invar_s2 = combined[:, :, 0]**2 - np.sum(combined[:, :, 1:]**2, axis=2)
    CoM = np.sum(four_vectors, axis=0)
    CoM_s2 = CoM[0]**2 - np.sum(CoM[1:]**2)
    affinities_grid = invar_s2/CoM_s2
    return affinities_grid
    


# Plotting code

def bins_extent(values, max_bins=20):
    """
    

    Parameters
    ----------
    values :
        
    max_bins :
         (Default value = 20)

    Returns
    -------

    """
    # values must be flat
    values = np.array(sorted(set(values)))
    extent = [np.min(values), np.max(values)]
    width = extent[1] - extent[0]
    bin_length = max(width/max_bins, np.min(values[1:] - values[:-1]))
    n_bins = int(np.ceil(width/bin_length))
    bins = np.linspace(*extent, n_bins+1)
    return bins, extent


def affinities_in_event(eventWise, affinities, results,
                        cluster_class=None, cluster_params=None):
    """
    

    Parameters
    ----------
    eventWise :
        
    affinities :
        
    results :
        
    cluster_class :
         (Default value = None)
    cluster_params :
         (Default value = None)

    Returns
    -------

    """
    n_parts = 3
    sourceidx = eventWise.JetInputs_SourceIdx.tolist()
    if len(sourceidx) < 2:  # cannot clasisfy relatiosn without at least 2 particles
        return
    b_idxs = FormShower.descendant_idxs(eventWise, *eventWise.BQuarkIdx)
    b_decendants = [sourceidx.index(d) for d in b_idxs
                    if d in sourceidx]
    #b_decendants = []  # check the labeling
    grids = [affinity(eventWise) for affinity in affinities]
    catigory = np.zeros_like(grids[0], dtype=int)
    catigory[:, b_decendants] += 1
    catigory[b_decendants, :] += 1
    grid = np.empty_like(catigory, dtype=float)
    for a, affinity in enumerate(affinities):
        try:
            grid[:] = affinity(eventWise)
        except ValueError:
            continue
        for i in range(n_parts):
            results[a][i] += grid[catigory == i].tolist()
    if cluster_class is not None:
        jets = cluster_class(eventWise, dict_jet_params=cluster_params, assign=False)
        attrs = ["Energy", "Px", "Py", "Pz", "PT", "Rapidity", "Phi"]
        attrs = {f'JetInputs_{name}': getattr(jets, f'_{name}_col') for name in attrs}
        while jets.currently_avalible > 0:
            # cluster and get the new affinity grid
            jets._step_assign_parents()
            float_array = np.array(jets._floats)
            parent_idxs = jets.Parent
            jet_idxs = jets.InputIdx
            # this recalcualtes from scratch each time
            # if this takes too long consider butchering this fiunction into the while loop
            tagging_criteria = TrueTag.percent_pos(jet_idxs, parent_idxs, b_decendants,
                                                   float_array[:, jets._Energy_col])
            top_level = parent_idxs == -1
            if np.sum(top_level) < 2:
                return
            particle_collection = attrdict.AttrDict({name: float_array[top_level, col]
                                                     for name, col in attrs.items()})
            b_decendants = np.where(tagging_criteria[top_level] > 0.5)[0]
            catigory = np.zeros_like(grid, dtype=int)
            catigory[:, b_decendants] += 1
            catigory[b_decendants, :] += 1
            grid = np.empty_like(catigory, dtype=float)
            for a, affinity in enumerate(affinities):
                try:
                    grid[:] = affinity(eventWise)
                except ValueError:  # bcuase somtimes the affinity will return an empty list
                    continue
                for i in range(n_parts):
                    results[a][i] += grid[catigory == i].tolist()


def all_affinities(eventWise, affinities,
                   cluster_class, cluster_params, silent=False):
    """
    

    Parameters
    ----------
    eventWise :
        
    affinities :
        
    cluster_class :
        
    cluster_params :
        
    silent :
         (Default value = False)

    Returns
    -------

    """
    n_parts = 3
    results = np.empty((len(affinities), n_parts, 0)).tolist()
    n_events = len(eventWise.X)
    start_point = 0
    end_point = n_events
    if not silent:
        print(f" Will stop at {100*end_point/n_events}%")
    for event_n in range(start_point, end_point):
        if event_n % 10 == 0 and not silent:
            print(f"{100*event_n/n_events}%", end='\r', flush=True)
        eventWise.selected_index = event_n
        affinities_in_event(eventWise, affinities, results,
                            cluster_class, cluster_params)
    return results


def plot_affinities(eventWise, affinity, affinity2=None, name=None, name2=None,
                   cluster_class=None, cluster_params=None, silent=False, bins=50):
    """
    

    Parameters
    ----------
    eventWise :
        
    affinity :
        
    affinity2 :
         (Default value = None)
    name :
         (Default value = None)
    name2 :
         (Default value = None)
    cluster_class :
         (Default value = None)
    cluster_params :
         (Default value = None)
    silent :
         (Default value = False)
    bins :
         (Default value = 50)

    Returns
    -------

    """
    twoD = affinity2 is not None
    labels = ['Both background', 'Crossover', 'Both b decendants']
    if name is not None:
        plt.xlabel(name)
    if twoD:
        parts, parts2 = all_affinities(eventWise, [affinity, affinity2], cluster_class, cluster_params)
        n_parts = len(parts)
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
            #density /= np.max(density)
            mapable = ax_ar[i+1].imshow(density, origin='lower', extent=extent, aspect='auto')
            ax_ar[i+1].set_xlabel(name + " " + labels[i])
            ax_ar[i+1].set_ylabel(name2)
            colours[:, :, i] += density
        cbar = plt.colorbar(mapable)
        cbar.set_label("Density")
        positive = colours[:, :, 1] - colours[:, :, 0] - colours[:, :, 2]
        ax_ar[0].imshow(positive, origin='lower', extent=extent,
                   aspect='auto')
        #ax_ar[0].imshow(colours, origin='lower', extent=extent,
        #           aspect='auto')
        ax_ar[0].set_xlabel(name)
        ax_ar[0].set_ylabel(name2)
        for i, name in enumerate(labels):
            colour = [0, 0, 0]
            colour[i] = 1
            ax_ar[0].scatter([], [], marker=',', c=[colour], label=name)
        #ax_ar[0].legend()
        ys = np.linspace(0, 3, 100)
        line = -0.06848*ys*ys + 0.8278*ys + 0.6873
        ax_ar[0].plot(line, ys)
    else:
        # for number of bins
        parts, = all_affinities(eventWise, [affinity], cluster_class, cluster_params)
        bins, extent = bins_extent(np.sum(parts))
        plt.hist(parts,
                 bins=bins, histtype='step', density=True,
                 label=labels, range=extent)
        plt.ylabel('counts')
        plt.legend()


class AffinitySet:
    """ """
    def __init__(self, dir_name, eventWise=None):
        self.dir_name = dir_name
        if eventWise is None:
            try:
                with open(os.path.join(dir_name, "eventWise.txt"), 'r') as ew_loc:
                    eventWise = ew_loc.read().strip()
            except FileNotFoundError:
                eventWise = InputTools.get_file_name("Name the eventWise file to use; ", '.awkd')
        if isinstance(eventWise, str):
            eventWise = Components.EventWise.from_file(eventWise)
        self.eventWise = eventWise

    def calculate(self, affinity_names=None, cluster_class=None, cluster_params=None):
        """
        

        Parameters
        ----------
        affinity_names :
             (Default value = None)
        cluster_class :
             (Default value = None)
        cluster_params :
             (Default value = None)

        Returns
        -------

        """
        if affinity_names is None:
            affinity_names = [InputTools.list_complete("Name affinity to recalculate; ",
                                                       affinity_choices.keys()).strip()]
            if len(affinity_names[0]) == 0:
                return
        affinities = [affinity_choices[name] for name in affinity_names]
        results = all_affinities(self.eventWise, affinities, cluster_class, cluster_params)
        self.save(affinity_names, results)
        return results

    def save(self, names, results):
        """
        

        Parameters
        ----------
        names :
            
        results :
            

        Returns
        -------

        """
        with open(os.path.join(self.dir_name, "eventWise.txt"), 'w') as ew_loc:
            ew_loc.write(os.path.join(self.eventWise.dir_name, self.eventWise.save_name))
        for name, result in zip(names, results):
            save_name = os.path.join(self.dir_name, f'{name}_affinities.csv')
            with open(save_name, 'w') as save:
                save.write(f"{name}, {self.eventWise.save_name} affinities\n")
                for l in range(len(result)):
                    save.write(', '.join([str(x) for x in result[l]]) + '\n')

    def load(self, affinity_name):
        """
        

        Parameters
        ----------
        affinity_name :
            

        Returns
        -------

        """
       save_path = os.path.join(self.dir_name, f"{affinity_name}_affinities.csv")
       return self._load_path(save_path)

    def load_all(self):
        """ """
        names = os.listdir(self.dir_name)
        names.remove('eventWise.txt')
        paths = [os.path.join(self.dir_name, name) for name in names]
        affinity_names = [name.split('_affinities', 1)[0] for name in names]
        results = [self._load_path(path) for path in paths]
        return affinity_names, results

    def _load_path(self, save_path):
        """
        

        Parameters
        ----------
        save_path :
            

        Returns
        -------

        """
        with open(save_path, 'r') as save:
            header = save.readline()
            data = save.readlines()
        results = [[float(x) for x in line[:-1].split(', ')] for line in data]
        return results

    def print_aucs(self):
        """ """
        names, results = self.load_all()
        for name, result in zip(names, results):
            classA = result[0] + result[2]
            classB = result[1]
            score = np.array(classA + classB)
            # normalise this
            score /= np.sum(score) 
            label = np.concatenate((np.ones(len(classA)), np.zeros(len(classB))))
            auc_here = sklearn.metrics.roc_auc_score(label, score)
            if auc_here < 0.5:
                # invert the positive catigory
                auc_here = 1-auc_here
            print(f'{name}; {auc_here}')

    def plot_rocs(self):
        """ """
        names, results = self.load_all()
        for name, result in zip(names, results):
            classA = result[0] + result[2]
            classB = result[1]
            score = np.array(classA + classB)
            # normalise this
            score /= np.sum(score) 
            label = np.concatenate((np.ones(len(classA)), np.zeros(len(classB))))
            fpr, tpr, _ = sklearn.metrics.roc_curve(label, score)
            auc_here = sklearn.metrics.auc(fpr, tpr)
            if auc_here < 0.5:
                # invert the positive catigory
                tpr, fpr = fpr, tpr
                auc_here = 1-auc_here
            if auc_here > 0.6:
                label = f'{name} {auc_here:.2f}'
                plt.plot(fpr, tpr, alpha=0.5, label=label)
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.legend(loc='lower right')
        plt.savefig("auc_cuves.png")
        plt.show(block=True)


def apply_confusion(eventWise, jet_name, function, silent=False):
    """
    

    Parameters
    ----------
    eventWise :
        
    jet_name :
        
    function :
        
    silent :
         (Default value = False)

    Returns
    -------

    """
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
        b_decendants = FormShower.descendant_idxs(eventWise, *eventWise.BQuarkIdx)
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
    """
    

    Parameters
    ----------
    eventWise :
        
    jet_name :
        
    silent :
         (Default value = False)

    Returns
    -------

    """
    def function(ew, tp, fp, tn, fn):
        """
        

        Parameters
        ----------
        ew :
            
        tp :
            
        fp :
            
        tn :
            
        fn :
            

        Returns
        -------

        """
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
    """
    

    Parameters
    ----------
    eventWise :
        
    jet_name :
        
    silent :
         (Default value = False)

    Returns
    -------

    """
    def function(ew, tp, fp, tn, fn):
        """
        

        Parameters
        ----------
        ew :
            
        tp :
            
        fp :
            
        tn :
            
        fn :
            

        Returns
        -------

        """
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


def visulise_affinities():
    """ """
    file_name = InputTools.get_file_name("Name the eventWise; ")
    ew = Components.EventWise.from_file(file_name)
    name1 = InputTools.list_complete("Chose affinity 1; ", affinity_choices.keys()).strip()
    affinity1 = affinity_choices[name1]
    if InputTools.yesNo_question("Second affinity? "):
        name2 = InputTools.list_complete("Chose affinity 2; ", affinity_choices.keys()).strip()
        affinity2 = affinity_choices[name2]
    else:
        name2 = affinity2 = None
    if InputTools.yesNo_question("Apply clustering? "):
        _, cluster_class, chosen_parameters = SingleFormJets.pick_class_params()
    else:
        cluster_class = chosen_parameters = None
    InputTools.last_selections.write("affinity_selections.dat")
    plot_affinities(ew, affinity1, affinity2, name=name1, name2=name2,
                   cluster_class=cluster_class, cluster_params=chosen_parameters)
    plt.show(block=True)

if __name__ == '__main__':
    visulise_affinities()
    #affs = AffinitySet("./500_affinities")
    #affs.calculate()
    #affs.print_aucs()
    #affs.plot_rocs()

