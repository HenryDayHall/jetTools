import numpy as np
import awkward
import itertools
from matplotlib import pyplot as plt
import matplotlib
import scipy.spatial
from ipdb import set_trace as st
from tree_tagger import Constants, Components, FormShower, PlottingTools, TrueTag, FormJets


def filter(eventWise, jet_name, jet_idxs, track_cut=None, min_jet_PT=None):
    """
    Return a subset of a list of jet indices, in a particular event,
    that match the required criteria

    Parameters
    ----------
    eventWise : EventWise
        dataset containing the jets
    jet_name : str
        prefix of the jet's variables in the eventWise
    jet_idxs : array like of int
        the indices of the jets in this event to be considered
    track_cut : int
        required minimum number of tracks for the jet to be selected
        if None the value s taken from Constants.py
         (Default value = None)
    min_jet_PT : float
        required minimum jet PT for the jet to be selected
        if None the value s taken from Constants.py
         (Default value = None)

    Returns
    -------
    valid : array like of ints
        the subset of jet_idxs that pass criteria

    """
    if track_cut is None:
        track_cut = Constants.min_ntracks
    if min_jet_PT is None:
        min_jet_PT = Constants.min_jetpt
    assert eventWise.selected_index is not None
    input_name = jet_name + '_InputIdx'
    root_name = jet_name + '_RootInputIdx'
    children = getattr(eventWise, jet_name+'_Child1')[jet_idxs]
    valid = np.where([np.sum(c == -1) >= track_cut for c in children])[0]
    pt = eventWise.match_indices(jet_name+'_PT', root_name, input_name)[jet_idxs[valid]]
    valid = valid[pt.flatten() > min_jet_PT]
    return jet_idxs[valid]


def order_tagged_jets(eventWise, jet_name, ranking_variable="PT", jet_pt_cut=None, track_cut=None):
    """
    Find the indices required to put only the valid jets into
    accending order as determined by some parameter.

    Parameters
    ----------
    eventWise : EventWise
        dataset containing the jets
    jet_name : str
        prefix of the jet's variables in the eventWise
    ranking_variable : str
        The name of the variable by which or order
        Should exist per consitiuent in the jet
         (Default value = "PT")
    track_cut : int
        required minimum number of tracks for the jet to be used
        if None the value s taken from Constants.py
         (Default value = None)
    jet_pt_cut : float
        required minimum jet PT for the jet to be used
        if None the value s taken from Constants.py
         (Default value = None)

    Returns
    -------
    jet_idxs : array like of int
        the ordered indices of the jets in this event to be considered

    """
    assert eventWise.selected_index is not None
    tagged_idxs = np.where([len(t) > 0 for t in getattr(eventWise, jet_name + '_MTags')])[0]
    tagged_idxs = filter(eventWise, jet_name, tagged_idxs, track_cut, jet_pt_cut)
    input_name = jet_name + '_InputIdx'
    root_name = jet_name + '_RootInputIdx'
    ranking_values = eventWise.match_indices(jet_name+'_'+ranking_variable, root_name, input_name).flatten()
    order = np.argsort(ranking_values[tagged_idxs])
    return tagged_idxs[order]


def combined_jet_mass(eventWise, jet_name, jet_idxs):
    """
    calcualte the invarient mass of a combination of jets

    Parameters
    ----------
    eventWise : EventWise
        dataset containing the jets
    jet_name : str
        prefix of the jet's variables in the eventWise
    jet_idxs : array like of int
        the indices of the jets in this event to be considered

    Returns
    -------
    mass : float
        mass of the specified jets

    """
    assert eventWise.selected_index is not None
    input_name = jet_name + '_InputIdx'
    root_name = jet_name + '_RootInputIdx'
    px = eventWise.match_indices(jet_name+'_Px', root_name, input_name)[jet_idxs].flatten()
    py = eventWise.match_indices(jet_name+'_Py', root_name, input_name)[jet_idxs].flatten()
    pz = eventWise.match_indices(jet_name+'_Pz', root_name, input_name)[jet_idxs].flatten()
    e = eventWise.match_indices(jet_name+'_Energy', root_name, input_name)[jet_idxs].flatten()
    mass = np.sqrt(np.sum(e)**2 - np.sum(px)**2 - np.sum(py)**2 - np.sum(pz)**2)
    return mass


def cluster_mass(eventWise, particle_idxs):
    """
    calculate the invarint mass of a group of particles

    Parameters
    ----------
    eventWise :
        
    particle_idxs :
        

    Returns
    -------

    """
    assert eventWise.selected_index is not None
    particle_idxs = list(particle_idxs)
    px = np.sum(eventWise.Px[particle_idxs])
    py = np.sum(eventWise.Py[particle_idxs])
    pz = np.sum(eventWise.Pz[particle_idxs])
    e = np.sum(eventWise.Energy[particle_idxs])
    return np.sqrt(e**2 - px**2 - py**2 - pz**2)


def inverse_condensed_indices(idx, n):
    """ Get the indices of the input points from a scipy distance vector

    Parameters
    ----------
    idx : int
        poition in the flat distance vector
    n : int
        number of points that made the distance vector

    Returns
    -------
    : int
     the first index this point originated from
    : int
     the second index this point originated from
    """
    k = 0
    for i in range(n):
        for j in range(i+1, n):
            if k == idx:
                return (i, j)
            k +=1
    else:
        return None


def smallest_angle_parings(eventWise, jet_name, jet_pt_cut):
    """
    In a single event greedly pair the jets that are tagged and
    have the smallest angular distances.
    

    Parameters
    ----------
    eventWise : EventWise
        dataset containing the jets
    jet_name : str
        prefix of the jet's variables in the eventWise
    jet_pt_cut : float
        required minimum jet PT for the jet to be selected
        if None the value s taken from Constants.py

    Returns
    -------
    pairs : list of tuples of ints
        each item in the list is a tuple containing two ints
        each int refering to a jet index in the eventWise

    """
    assert eventWise.selected_index is not None
    tagged_idxs = np.where([len(t) > 0 for t in getattr(eventWise, jet_name + '_MTags')])[0]
    tagged_idxs = filter(eventWise, jet_name, tagged_idxs, min_jet_PT=jet_pt_cut)
    num_tags = len(tagged_idxs)
    if num_tags < 2:
        return []
    elif num_tags == 2:
        return [tagged_idxs]
    else:
        input_name = jet_name + '_InputIdx'
        root_name = jet_name + '_RootInputIdx'
        phi = eventWise.match_indices(jet_name+'_Phi', root_name, input_name)[tagged_idxs]
        rapidity = eventWise.match_indices(jet_name+'_Rapidity', root_name, input_name)[tagged_idxs]
        points = np.vstack((phi, rapidity)).T
        distances = scipy.spatial.distance.pdist(points)
        closest = np.argmin(distances)
        pairs = [inverse_condensed_indices(closest, num_tags)]
        if num_tags == 4:
            pairs.append(tuple(set(range(num_tags)) - set(pairs[0])))
        pairs = [[tagged_idxs[p[0]], tagged_idxs[p[1]]] for p in pairs]
        return pairs


def all_smallest_angles(eventWise, jet_name, jet_pt_cut):
    """
    In all events greedly pair the jets that are tagged and
    have the smallest angular distances.
    Find the masses of the combination and make a list.
    

    Parameters
    ----------
    eventWise : EventWise
        dataset containing the jets
    jet_name : str
        prefix of the jet's variables in the eventWise
    jet_pt_cut : float
        required minimum jet PT for the jet to be selected
        if None the value s taken from Constants.py

    Returns
    -------
    pair_masses : list of floats
        masses of the pairs

    """
    eventWise.selected_index = None
    n_events = len(getattr(eventWise, jet_name+'_InputIdx'))
    pair_masses = []
    for event_n in range(n_events):
        if event_n % 10 == 0:
            print(f"{event_n/n_events:.1%}", end='\r')
        eventWise.selected_index = event_n
        pairs = smallest_angle_parings(eventWise, jet_name, jet_pt_cut)
        if len(pairs) == 0:
            continue
        for pair in pairs:
            pair_masses.append(combined_jet_mass(eventWise, jet_name, pair))
    return pair_masses


def all_jet_masses(eventWise, jet_name, jet_pt_cut=None):
    """
    Calculate the mass of the combination of all tagged jets in each event.
    
    Parameters
    ----------
    eventWise : EventWise
        dataset containing the jets
    jet_name : str
        prefix of the jet's variables in the eventWise
    jet_pt_cut : float
        required minimum jet PT for the jet to be selected
        if None the value s taken from Constants.py
        (Default = None)

    Returns
    -------
    all_masses : list of floats
        masses of the all the jets in each event

    """
    eventWise.selected_index = None
    n_events = len(getattr(eventWise, jet_name+'_InputIdx'))
    all_masses = []
    for event_n in range(n_events):
        if event_n % 100 == 0:
            print(f"{event_n/n_events:.1%}", end='\r')
        eventWise.selected_index = event_n
        tagged_jets = np.where([len(t) for t in getattr(eventWise, jet_name + "_MTags")])[0]
        tagged_jets = filter(eventWise, jet_name, tagged_jets, track_cut=2, min_jet_PT=jet_pt_cut)
        all_masses.append(combined_jet_mass(eventWise, jet_name, tagged_jets))
    return all_masses


def plot_smallest_angles(eventWise, jet_name, jet_pt_cut, show=True):
    """
    Make a plot of masses of all the tagged jets grouped by smallest angle.

    Parameters
    ----------
    eventWise : EventWise
        dataset containing the jets
    jet_name : str
        prefix of the jet's variables in the eventWise
    jet_pt_cut : float
        required minimum jet PT for the jet to be selected
        if None the value s taken from Constants.py
        (Default = None)
    show : bool
        Should it call plt.show()?
         (Default value = True)

    """
    pair_masses = all_smallest_angles(eventWise, jet_name, jet_pt_cut)
    n_events = len(pair_masses)
    plt.hist(pair_masses, bins=50, label="Masses of single jets at smallest angles")
    plt.title("Masses of pair of single jets at smallest angles")
    plt.xlabel("Mass (GeV)")
    plt.ylabel(f"Counts in {n_events} events")
    if show:
        plt.show()


def all_PT_pairs(eventWise, jet_name, jet_pt_cut=None, max_tag_angle=0.8, track_cut=None, seperate_multi_tagged=False):
    """
    Gather all possible pairings of jets by PT order.
    Highest PT first.

    Parameters
    ----------
    eventWise : EventWise
        dataset containing the jets
    jet_name : str
        prefix of the jet's variables in the eventWise
    jet_pt_cut : float
        required minimum jet PT for the jet to be selected
        if None the value s taken from Constants.py
        (Default = None)
    track_cut : int
        required minimum number of tracks for the jet to be used
        if None the value s taken from Constants.py
         (Default value = None)
    max_tag_angle : float
        The maximum deltaR betweeen a tag and its jet
         (Default value = 0.8)
    seperate_multi_tagged: bool
        should multi tagged be put in a seperate list?

    Returns
    -------
    all_masses : list of floats
        the masses of all the tagged jets in each event.
    pairs : list of list of ints
        the indices refering to the combinations chosen
    pair_masses : list of list of floats
        the masses of the combinations chosen.
        Inner list may not have the same length as the number of events
        beuase for some event not all pairings are possble.


    """
    eventWise.selected_index = None
    if jet_name + "_MTags" not in eventWise.columns:
        # jet_pt_cut = None becuase we don't want to cut before tagging
        TrueTag.add_tags(eventWise, jet_name, max_tag_angle, np.inf, jet_pt_cut=None)
        TrueTag.add_mass_share(eventWise, jet_name, np.inf)
    n_events = len(getattr(eventWise, jet_name+'_InputIdx'))
    # becuase we will use these to take indices from a numpy array they need to be lists 
    # not tuples
    pairs = [list(pair) for pair in itertools.combinations(range(4), 2)]
    pair_masses = [[] for _ in pairs]
    all_masses = []
    multi_tagged_masses = []
    for event_n in range(n_events):
        if event_n % 100 == 0:
            print(f"{event_n/n_events:.1%}", end='\r')
        eventWise.selected_index = event_n
        sorted_idx = order_tagged_jets(eventWise, jet_name, "PT", jet_pt_cut, track_cut)
        # this is accending order, we need decending
        sorted_idx = sorted_idx[::-1]
        n_jets = len(sorted_idx)
        if n_jets == 0:
            continue
        all_masses.append(combined_jet_mass(eventWise, jet_name, sorted_idx))
        for i, pair in enumerate(pairs):
            if max(pair) < n_jets:
                pair_masses[i].append(combined_jet_mass(eventWise, jet_name, sorted_idx[pair]))
    return all_masses, pairs, pair_masses


def all_h_combinations(eventWise, jet_name, jet_pt_cut=None, track_cut=None, tag_type=None):
    """
    Gather all jets tagged by b-quarks from the same light higgs.


    Parameters
    ----------
    eventWise : EventWise
        dataset containing the jets
    jet_name : str
        prefix of the jet's variables in the eventWise
    jet_pt_cut : float
        required minimum jet PT for the jet to be selected
        if None the value s taken from Constants.py
        (Default = None)
    track_cut : int
        required minimum number of tracks for the jet to be used
        if None the value s taken from Constants.py
         (Default value = None)

    Returns
    -------
    four_tags : list of floats
        the combined masses of the jets that captured all 4 tags
    pairs : list of floats
        the combined masses of the jets that captured two tags from the same light higgs

    """
    eventWise.selected_index = None
    if tag_type is None:
        tag_type = "_MTags"
    tag_name = jet_name + tag_type
    n_events = len(getattr(eventWise, jet_name+'_InputIdx'))
    # becuase we will use these to take indices from a numpy array they need to be lists 
    # not tuples
    light_higgs_pid = 25
    pair1 = []
    pair2 = []
    four_tags = []
    for event_n in range(n_events):
        if event_n % 100 == 0:
            print(f"{event_n/n_events:.1%}", end='\r')
        eventWise.selected_index = event_n
        tags = getattr(eventWise, tag_name)
        flat_tags = tags.flatten()
        # check if all 4 tags are present
        if len(flat_tags) == 4:
            tagged_jets = [i for i, t in enumerate(tags) if len(t)]
            four_tags.append(combined_jet_mass(eventWise, jet_name, tagged_jets))
        else:
            four_tags.append(0.)
        # now see if there are any same h pairs
        if len(flat_tags) > 0:
            parents = []
            # look for the closest h to each tag
            for tag in flat_tags:
                stack = eventWise.Parents[tag].tolist()
                while stack:
                    check = stack.pop()
                    if eventWise.MCPID[check] == light_higgs_pid:
                        parents.append(check)
                        break
                    stack += eventWise.Parents[check].tolist()
                else:  # if we his this we didn't hit the break clause
                    parents.append(-1)
            # check through the parents
            masses = []
            for i, p in enumerate(parents):
                tag = flat_tags[i]
                # if there is another of these in the list then we have both decendants
                if p in parents[i+1:]:
                    jet_idxs = [i for i, t in enumerate(tags) if tag in t]
                    masses.append(combined_jet_mass(eventWise, jet_name, jet_idxs))
            mass1 = mass2 = 0
            if len(masses) == 2:
                mass2, mass1 = sorted(masses)
            elif len(masses) == 1:
                mass1 = masses[0]
            pair1.append(mass1)
            pair2.append(mass2)
    return four_tags, pair1, pair2


def plot_correct_pairs(eventWise, jet_names, show=True):
    """
    Plot all possible pairings of jets by PT order.

    Parameters
    ----------
    eventWise : EventWise
        dataset containing the jets
    jet_name : str
        prefix of the jet's variables in the eventWise
    
    """
    fig, ax_array = plt.subplots(1, 3)
    # get lobal inputs
    eventWise.selected_index = None
    heavy, light1, light2 = descendants_masses(eventWise)
    n_events = len(getattr(eventWise, jet_names[0]+'_InputIdx'))
    # prepare the hist parameters
    cmap = matplotlib.cm.get_cmap('tab10')
    colours = [cmap(x) for x in np.linspace(0, 1, len(jet_names))]
    hist_params = dict(bins=40, density=False, histtype='step')
    other_params = dict(bins=40, density=False, histtype='stepfilled', color='gray', alpha=0.8)
    # get the jet data
    four_tag_masses = []
    pair1_masses = []
    pair2_masses = []
    for name in jet_names:
        four_masses, pair1, pair2 = all_h_combinations(eventWise, name)
        four_tag_masses.append(four_masses)
        pair1_masses.append(pair1)
        pair2_masses.append(pair2)

    hist_params['range'] = other_params['range'] = np.nanmin(heavy), np.nanmax(heavy) + 20
    ax_array[0].set_title(f"Heavy higgs")
    ax_array[0].hist(heavy, label="heavy decendants", **other_params)
    for i, name in enumerate(jet_names):
        linewidth = 1+i/2
        alpha = 1-i/(len(jet_names)+1)
        ax_array[0].hist(four_tag_masses[i], label=name, linewidth=linewidth, alpha=alpha,
                         color=colours[i], **hist_params)
    ax_array[0].set_xlabel("Mass (GeV)")
    ax_array[0].set_ylabel(f"Counts in {n_events}")
    ax_array[0].legend()
    ax_array[0].set_xlim(hist_params['range'])

    ax_array[1].set_title(f"Better observed light higgs")
    hist_params['range'] = other_params['range'] = np.nanmin(light1), np.nanmax(light1) + 10
    ax_array[1].hist(light1, label="light decendants", **other_params)
    for i, name in enumerate(jet_names):
        linewidth = 1+i/2
        alpha = 1-i/(len(jet_names)+1)
        ax_array[1].hist(pair1_masses[i], label=name, linewidth=linewidth, alpha=alpha,
                         color=colours[i], **hist_params)
    ax_array[1].set_xlabel("Mass (GeV)")
    ax_array[1].set_ylabel(f"Counts in {n_events}")
    ax_array[1].legend()
    ax_array[1].set_xlim(hist_params['range'])

    ax_array[2].set_title(f"Less observed light higgs")
    hist_params['range'] = other_params['range'] = np.nanmin(light2), np.nanmax(light2) + 10
    ax_array[2].hist(light2, label="light decendants", **other_params)
    for i, name in enumerate(jet_names):
        linewidth = 1+i/2
        alpha = 1-i/(len(jet_names)+1)
        ax_array[2].hist(pair2_masses[i], label=name, linewidth=linewidth, alpha=alpha,
                         color=colours[i], **hist_params)
    ax_array[2].set_xlabel("Mass (GeV)")
    ax_array[2].set_ylabel(f"Counts in {n_events}")
    ax_array[2].legend()
    ax_array[2].set_xlim(hist_params['range'])
    if show:
        plt.show()
    return four_tag_masses, pair1_masses, pair2_masses


def plot_scatter_correct_pairs(eventWise, jet_names, show=True):
    """
    Plot all possible pairings of jets by PT order.

    Parameters
    ----------
    eventWise : EventWise
        dataset containing the jets
    jet_name : str
        prefix of the jet's variables in the eventWise
    
    """
    fig, ax_array = plt.subplots(1, 3)
    # get lobal inputs
    eventWise.selected_index = None
    heavy, light1, light2 = descendants_masses(eventWise)
    n_events = len(getattr(eventWise, jet_names[0]+'_InputIdx'))
    # prepare the hist parameters
    cmap = matplotlib.cm.get_cmap('tab10')
    colours = [cmap(x) for x in np.linspace(0, 1, len(jet_names))]
    hist_params = dict(bins=40, density=False, histtype='step')
    other_params = dict(bins=40, density=False, histtype='stepfilled', color='gray', alpha=0.8)
    # get the jet data
    four_tag_masses = []
    pair1_masses = []
    pair2_masses = []
    for name in jet_names:
        four_masses, pair1, pair2 = all_h_combinations(eventWise, name)
        four_tag_masses.append(four_masses)
        pair1_masses.append(pair1)
        pair2_masses.append(pair2)

    hist_params['range'] = other_params['range'] = np.nanmin(heavy), np.nanmax(heavy) + 20
    ax_array[0].set_title(f"Heavy higgs")
    ax_array[0].hist(heavy, label="heavy decendants", **other_params)
    for i, name in enumerate(jet_names):
        linewidth = 1+i/2
        alpha = 1-i/(len(jet_names)+1)
        ax_array[0].hist(four_tag_masses[i], label=name, linewidth=linewidth, alpha=alpha,
                         color=colours[i], **hist_params)
    ax_array[0].set_xlabel("Mass (GeV)")
    ax_array[0].set_ylabel(f"Counts in {n_events}")
    ax_array[0].legend()
    ax_array[0].set_xlim(hist_params['range'])

    ax_array[1].set_title(f"Better observed light higgs")
    hist_params['range'] = other_params['range'] = np.nanmin(light1), np.nanmax(light1) + 10
    ax_array[1].hist(light1, label="light decendants", **other_params)
    for i, name in enumerate(jet_names):
        linewidth = 1+i/2
        alpha = 1-i/(len(jet_names)+1)
        ax_array[1].hist(pair1_masses[i], label=name, linewidth=linewidth, alpha=alpha,
                         color=colours[i], **hist_params)
    ax_array[1].set_xlabel("Mass (GeV)")
    ax_array[1].set_ylabel(f"Counts in {n_events}")
    ax_array[1].legend()
    ax_array[1].set_xlim(hist_params['range'])

    ax_array[2].set_title(f"Less observed light higgs")
    hist_params['range'] = other_params['range'] = np.nanmin(light2), np.nanmax(light2) + 10
    ax_array[2].hist(light2, label="light decendants", **other_params)
    for i, name in enumerate(jet_names):
        linewidth = 1+i/2
        alpha = 1-i/(len(jet_names)+1)
        ax_array[2].hist(pair2_masses[i], label=name, linewidth=linewidth, alpha=alpha,
                         color=colours[i], **hist_params)
    ax_array[2].set_xlabel("Mass (GeV)")
    ax_array[2].set_ylabel(f"Counts in {n_events}")
    ax_array[2].legend()
    ax_array[2].set_xlim(hist_params['range'])
    if show:
        plt.show()
    return four_tag_masses, pair1_masses, pair2_masses


def plot_all_jets(eventWise, jet_names, show=True):
    """
    Plot all possible pairings of jets by PT order.

    Parameters
    ----------
    eventWise : EventWise
        dataset containing the jets
    jet_name : str
        prefix of the jet's variables in the eventWise
    
    """
    fig, ax = plt.subplots(1, 1)
    # get lobal inputs
    eventWise.selected_index = None
    n_events = len(getattr(eventWise, jet_names[0]+'_InputIdx'))
    # prepare the hist parameters
    cmap = matplotlib.cm.get_cmap('tab10')
    colours = [cmap(x) for x in np.linspace(0, 1, len(jet_names))]
    hist_params = dict(bins=120, density=False, histtype='step')
    # get the jet data
    all_masses = []
    for name in jet_names:
        all_masses.append(all_jet_masses(eventWise, name))
    hist_params['range'] = np.nanmin(all_masses[0]), np.nanmax(all_masses[0])
    ax.set_title(f"Total b-jet mass")
    for i, name in enumerate(jet_names):
        linewidth = 1+i/2
        alpha = 1-i/(len(jet_names)+1)
        ax.hist(all_masses[i], label=name, linewidth=linewidth, alpha=alpha,
                         color=colours[i], **hist_params)
    ax.set_xlabel("Mass (GeV)")
    ax.set_ylabel(f"Counts in {n_events}")
    ax.legend()
    height = 800
    ax.vlines(40, 0, height, ls='--', colors='r')
    ax.text(40+2., 0.5*height, "Light higgs",
                  rotation='vertical', c='r')
    ax.vlines(125, 0, height, ls='--', colors='r')
    ax.text(125.+2, 0.5*height, "Heavy higgs",
                  rotation='vertical', c='r')
    if show:
        plt.show()


def plot_PT_pairs(eventWise, jet_names, jet_pt_cut=None, show=True, max_tag_angle=0.8):
    """
    Plot all possible pairings of jets by PT order.

    Parameters
    ----------
    eventWise : EventWise
        dataset containing the jets
    jet_name : str
        prefix of the jet's variables in the eventWise
    jet_pt_cut : float
        required minimum jet PT for the jet to be selected
        if None the value s taken from Constants.py
        (Default = None)
    max_tag_angle : float
        The maximum deltaR betweeen a tag and its jet
         (Default value = 0.8)
    
    """
    fig, ax_array = plt.subplots(3, 3)
    input_names = ["NumEigenvectors", "AffinityType", "ExpofPTMultiplier"]
    # get lobal inputs
    eventWise.selected_index = None
    heavy, light1, light2 = descendants_masses(eventWise)
    #PlottingTools.make_inputs_table(eventWise, jet_names, ax_array[2, 1], input_names)
    n_events = len(getattr(eventWise, jet_names[0]+'_InputIdx'))
    labels = jet_names + ["heavy descendants", "light descendants 1", "light descendants 2"]
    cmap = matplotlib.cm.get_cmap('gist_rainbow')
    colours = [cmap(x) for x in np.linspace(0, 1, len(labels))]
    all_masses = []
    all_pair_masses = [[], [], [], [], [], [], []]
    for name in jet_names:
        masses, pairs, pair_masses = all_PT_pairs(eventWise, name, jet_pt_cut, max_tag_angle=max_tag_angle)
        all_masses.append(masses)
        for i in range(6):
            all_pair_masses[i].append(pair_masses[i])
    all_masses += [heavy, light1, light2]
    for i in range(6):
        all_pair_masses[i] += [heavy, light1, light2]
    ax_array[0, 0].hist(all_masses, bins=100, density=True, histtype='step', label=labels, color=colours)
    ax_array[0, 0].set_xlabel("Mass (GeV)")
    ax_array[0, 0].set_ylabel(f"Density")

    ax_array[0, 1].set_title(f"Highest and second highest $p_T$ jets")
    ax_array[0, 2].set_title(f"Highest and third highest $p_T$ jets")
    ax_array[1, 0].set_title(f"Highest and lowest $p_T$ jets")
    ax_array[1, 1].set_title(f"Second and third highest $p_T$ jets")
    ax_array[1, 2].set_title(f"Second highest and lowest $p_T$ jets")
    ax_array[2, 0].set_title(f"Third highest and lowest $p_T$ jets")
    for ax, pair in zip(ax_array.flatten()[1:-2], all_pair_masses):
        ax.set_xlabel("Mass (GeV)")
        ax.set_ylabel(f"Density")
        ax.hist(pair, bins=100, histtype='step', label=labels, density=True, color=colours)
    PlottingTools.hide_axis(ax_array[2,1])
    PlottingTools.hide_axis(ax_array[2,2])
    ax_array[2, 2].hist([[]]*len(labels), color=colours, label=labels)
    ax_array[2, 2].legend()
    if show:
        plt.show()
    return all_masses, pair_masses


def all_doubleTagged_jets(eventWise, jet_name, jet_pt_cut=None):
    """
    Calculate the masses of all jets that have been allocated exactly 2 tags.

    Parameters
    ----------
    eventWise : EventWise
        dataset containing the jets
    jet_name : str
        prefix of the jet's variables in the eventWise
    jet_pt_cut : float
        required minimum jet PT for the jet to be selected
        if None the value s taken from Constants.py
        (Default = None)
        

    Returns
    -------
    masses : list of floats
        the masses of all jets with 2 tags

    """
    eventWise.selected_index = None
    n_events = len(getattr(eventWise, jet_name+'_InputIdx'))
    masses = []
    for event_n in range(n_events):
        if event_n % 10 == 0:
            print(f"{event_n/n_events:.1%}", end='\r')
        eventWise.selected_index = event_n
        tagged_idxs = np.where([len(t) == 2 for t in getattr(eventWise, jet_name + '_MTags')])[0]
        tagged_idxs = filter(eventWise, jet_name, tagged_idxs, min_jet_PT=jet_pt_cut)
        if len(tagged_idxs) == 0:
            continue
        for idx in tagged_idxs:
            masses.append(combined_jet_mass(eventWise, jet_name, idx))
    return masses


def plot_doubleTagged_jets(eventWise, jet_name, show=True):
    """
    Plot the masses of all jets that have been allocated exactly 2 tags.

    Parameters
    ----------
    eventWise : EventWise
        dataset containing the jets
    jet_name : str
        prefix of the jet's variables in the eventWise
    show : bool
        should this call plt.show()
         (Default value = True)

    """
    eventWise.selected_index = None
    n_events = len(getattr(eventWise, jet_name+'_InputIdx'))
    masses = all_doubleTagged_jets(eventWise, jet_name)
    plt.hist(masses, bins=50, label="Masses of fat jets")
    plt.title("Masses of fat jets")
    plt.xlabel("Mass (GeV)")
    plt.ylabel(f"Counts in {n_events} events")
    if show:
        plt.show()


def descendants_masses(eventWise, use_jetInputs=True):
    """
    From the JetInputs, calculate the masses of all particles
    that originate from a light higgs,
    and all particles that originate from the heavy higgs.
    Effectivly shows the mass that would eb obtained by perfect clustering.

    Parameters
    ----------
    eventWise : EventWise
        dataset containing the jets
    use_jetInputs : bool
        should only the particle in JetInputs be considered
         (Default value = True)

    Returns
    -------
    heavy_descendants_mass : list of float
        list of masses decendent from the heavy higgs
    light_descendants1_mass : list of float
        list of larger masses decendent from the light higgs
    light_descendants2_mass : list of float
        list of smaller masses decendent from the light higgs

    """
    eventWise.selected_index = None
    if "HeavyDecendentsMass" in eventWise.columns:
        heavy_descendants_mass = eventWise.HeavyDecendentsMass
        light_descendants1_mass = eventWise.Light1DecendentsMass
        light_descendants2_mass = eventWise.Light2DecendentsMass
        return heavy_descendants_mass, light_descendants1_mass, light_descendants2_mass
    heavy_higgs_pid = 35
    heavy_descendants_mass = []
    light_higgs_pid = 25
    light_descendants1_mass = []
    light_descendants2_mass = []
    n_events = len(eventWise.MCPID)
    for event_n in range(n_events):
        if event_n % 100 == 0:
            print(f"{event_n/n_events:.1%}", end='\r', flush=True)
        eventWise.selected_index = event_n
        # we only expect one heavy higgs, excluding radation dups
        heavy_idx = np.where(eventWise.MCPID == heavy_higgs_pid)[0]
        heavy_idx = {Components.last_instance(eventWise, hidx) for hidx in heavy_idx}
        assert len(heavy_idx) == 1, f"Problem, expected 1 higgs pid {heavy_higgs_pid}, found {len(heavy_idx)}"
        heavy_idx = list(heavy_idx)[0]
        heavy_descendants = FormShower.descendant_idxs(eventWise, heavy_idx)
        if use_jetInputs:
            # select only particles that have already been used in sourceidx
            heavy_descendants = heavy_descendants.intersection(eventWise.JetInputs_SourceIdx)
        if heavy_descendants:
            heavy_descendants_mass.append(cluster_mass(eventWise, heavy_descendants))
        else:
            heavy_descendants_mass.append(0.)
        # we expect two light higgs
        light_idxs = np.where(eventWise.MCPID == light_higgs_pid)[0]
        light_idxs = {Components.last_instance(eventWise, lidx) for lidx in light_idxs}
        assert len(light_idxs) == 2, f"Problem, expected 2 higgs pid {light_higgs_pid}, found {len(light_idxs)}"
        light_descendants = [FormShower.descendant_idxs(eventWise, lidx) for lidx in light_idxs]
        if use_jetInputs:
            # select only particles that have already been used in sourceidx
            light_descendants = [ldec.intersection(eventWise.JetInputs_SourceIdx) for ldec in light_descendants]
        # we also expect that the combination of the light descendants should be the heavy descendants
        light_mass = sorted([cluster_mass(eventWise, ldec) if ldec else 0.
                             for ldec in light_descendants])
        light_descendants1_mass.append(light_mass[1])
        light_descendants2_mass.append(light_mass[0])
    eventWise.append(HeavyDecendentsMass= awkward.fromiter(heavy_descendants_mass),
                     Light1DecendentsMass= awkward.fromiter(light_descendants1_mass),
                     Light2DecendentsMass= awkward.fromiter(light_descendants2_mass))
    return heavy_descendants_mass, light_descendants1_mass, light_descendants2_mass


if __name__ == '__main__':
    from tree_tagger import Components
    ew = Components.EventWise.from_file("best.awkd")
    jet_names = FormJets.get_jet_names(ew)
    
    #plot_PT_pairs(ew, jet_names, True)
    plot_correct_pairs(ew, jet_names, True)
    input()


