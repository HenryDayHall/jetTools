""" module to contain the framework for linking towers and tracks """
import itertools
import operator
import numpy as np

def proximity_lists(etas_1, phis_1, etas_2, phis_2, eta_seperation, phi_seperation):
    """ Function to create a list of proximate particles in the other list
        for both lists. L2 norm. """

    list1 = [[] for _ in etas_1]
    list2 = [[] for _ in etas_2]

    # start by binning the data on the scale of the seperation
    eps = 0.001
    eta_min = min(np.min(etas_1), np.min(etas_2)) - eps
    eta_max = max(np.max(etas_1), np.max(etas_2)) + eps
    phi_min = min(np.min(phis_1), np.min(phis_2)) - eps
    phi_max = max(np.max(phis_1), np.max(phis_2)) + eps
    n_bins_eta = int(np.ceil((eta_max-eta_min)/eta_seperation))
    eta_bins = np.linspace(eta_min, eta_max, n_bins_eta + 1)
    n_bins_phi = int(np.ceil(2*np.pi/phi_seperation))
    phi_bins = np.linspace(phi_min, phi_max, n_bins_phi + 1)

    # prepare the bin lists, with padding at every edge
    bins_1 = [[[] for _ in range(n_bins_eta + 2)] for _ in range(n_bins_phi + 2)]
    bins_2 = [[[] for _ in range(n_bins_eta + 2)] for _ in range(n_bins_phi + 2)]
    # the digitalise function considered bin 0 to be below the lower boundary
    # so this will automatically index with the padding
    # first binning in eta
    eta_bin_1 = np.digitize(etas_1, eta_bins)
    eta_bin_2 = np.digitize(etas_2, eta_bins)
    # then phi
    phi_bin_1 = np.digitize(phis_1, phi_bins)
    phi_bin_2 = np.digitize(phis_2, phi_bins)

    # rescale the eta and phi by the seperation, 
    # this will make calculating distances later easier
    etas_1_rescale = etas_1/eta_seperation
    etas_2_rescale = etas_2/eta_seperation
    phis_1_rescale = phis_1/phi_seperation
    phis_2_rescale = phis_2/phi_seperation

    # place in bins
    for index, (eta, phi) in enumerate(zip(etas_1_rescale, phis_1_rescale)):
        obj = (index, eta, phi)
        bins_1[phi_bin_1[index]][eta_bin_1[index]].append(obj)
    for index, (eta, phi) in enumerate(zip(etas_2_rescale, phis_2_rescale)):
        obj = (index, eta, phi)
        bins_2[phi_bin_2[index]][eta_bin_2[index]].append(obj)

    # in the upper and lower phi direction padding is changed to account for cyclic nature
    bins_1[0] = bins_1[-2]
    bins_1[-1] = bins_1[1]
    bins_2[0] = bins_2[-2]
    bins_2[-1] = bins_2[1]

    # everything in now binned, with padding
    # make list to store the objects proximate to each object
    proximates_1 = [] # will store objects from bins_2
    # will store objects from bins 1, will be added to in a non linear way
    proximates_2 = [[] for _ in etas_2]

    for index, (eta, phi) in enumerate(zip(etas_1_rescale, phis_1_rescale)):
        eta_bin = eta_bin_1[index]
        phi_bin = phi_bin_1[index]
        # add everything in the same bin
        proximates = [obj[0] for obj in bins_2[phi_bin][eta_bin]]
        surrounding = []
        for idx in itertools.product((eta_bin-1, eta_bin, eta_bin+1),
                                     (phi_bin-1, phi_bin, phi_bin+1)):
            if idx != (eta_bin, phi_bin):
                surrounding += bins_2[idx[1]][idx[2]]
        # go through the surroundings deciding which to add
        for obj in surrounding:
            # calculate the L2 norm with the rescaled coordinates
            # these are in range if their squared L2 norm is less than 1
            normed_dist = (eta - obj[1])**2 + (phi - obj[2])**2
            if normed_dist < 1:
                proximates.append(obj[0])
                proximates_1.append(proximates)
        # put this index in the appropreate parts of the other list
        for index_2 in proximates:
            proximates_2[index_2].append(index)
    return proximates_1, proximates_2



def MC_truth_links(tower_list, track_list):
    """ the monte carlo truth links between tracks and towers """
    links = {}
    for j, track in enumerate(track_list):
        gid = track.global_id
        linked = next([i for i, tower in enumerate(tower_list)
                       if gid in tower.global_ids],
                      None)
        links[j] = linked
    return links






