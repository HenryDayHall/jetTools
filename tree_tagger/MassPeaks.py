import numpy as np
import itertools
from matplotlib import pyplot as plt
import scipy.spatial
from ipdb import set_trace as st
from tree_tagger import Constants, Components, FormShower, PlottingTools, TrueTag


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
    tagged_idxs = np.where([len(t) > 0 for t in getattr(eventWise, jet_name + '_Tags')])[0]
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
    tagged_idxs = np.where([len(t) > 0 for t in getattr(eventWise, jet_name + '_Tags')])[0]
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
        tagged_jets = np.where([len(t) for t in getattr(eventWise, jet_name + "_Tags")])[0]
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


def all_PT_pairs(eventWise, jet_name, jet_pt_cut=None, max_tag_angle=0.8, track_cut=None):
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
    if jet_name + "_Tags" not in eventWise.columns:
        # jet_pt_cut = None becuase we don't want to cut before tagging
        TrueTag.add_tags(eventWise, jet_name, max_tag_angle, np.inf, jet_pt_cut=None)
    n_events = len(getattr(eventWise, jet_name+'_InputIdx'))
    # becuase we will use these to take indices from a numpy array they need to be lists 
    # not tuples
    pairs = [list(pair) for pair in itertools.combinations(range(4), 2)]
    pair_masses = [[] for _ in pairs]
    all_masses = []
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


def plot_PT_pairs(eventWise, jet_name, jet_pt_cut=None, show=True, max_tag_angle=0.8):
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
    all_masses, pairs, pair_masses = all_PT_pairs(eventWise, jet_name, jet_pt_cut, max_tag_angle=max_tag_angle)
    eventWise.selected_index = None
    n_events = len(getattr(eventWise, jet_name+'_InputIdx'))
    fig, ax_array = plt.subplots(3, 3)
    #fig, ax_array = plt.subplots(1, 3)
    ax_array = ax_array.flatten()
    PlottingTools.discribe_jet(eventWise, jet_name, ax_array[-1])
    heavy, light = descendants_masses(eventWise)
    label = [jet_name, "heavy descendants", "light descendants"]
    #for ax, pair, masses in zip(ax_array, pairs, pair_masses):
    #    data = [masses, heavy, light]
    #    title = f"Jets {pair[0]} and {pair[1]} (pT ordered), counts={len(masses)}"
    #    ax.set_title(title)
    #    #ax.hist(data, bins=500, histtype='step', label=label)
    #    ax.hist(masses, bins=50, histtype='step', label=jet_name)
    ax_array[0].set_title(f"All b-jets, counts={len(all_masses)}")
    ax_array[0].hist(all_masses, bins=50, histtype='step', label=jet_name)
    ax_array[1].set_title(f"Highest and second highest $p_T$ jets, counts={len(pair_masses[0])}")
    ax_array[1].hist(pair_masses[0], bins=50, histtype='step', label=jet_name)
    ax_array[2].set_title(f"Highest and third highest $p_T$ jets, counts={len(pair_masses[1])}")
    ax_array[2].hist(pair_masses[1], bins=50, histtype='step', label=jet_name)
    #ax.legend()
    ax_array[0].set_xlabel("Mass (GeV)")
    ax_array[1].set_xlabel("Mass (GeV)")
    ax_array[2].set_xlabel("Mass (GeV)")
    ax_array[0].set_ylabel(f"Counts in {n_events} events")
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
        tagged_idxs = np.where([len(t) == 2 for t in getattr(eventWise, jet_name + '_Tags')])[0]
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
    light_descendants_mass : list of float
        list of masses decendent from the light higgs

    """
    eventWise.selected_index = None
    heavy_higgs_pid = 35
    heavy_descendants_mass = []
    light_higgs_pid = 25
    light_descendants_mass = []
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
        # we expect two light higgs
        light_idxs = np.where(eventWise.MCPID == light_higgs_pid)[0]
        light_idxs = {Components.last_instance(eventWise, lidx) for lidx in light_idxs}
        assert len(light_idxs) == 2, f"Problem, expected 2 higgs pid {light_higgs_pid}, found {len(light_idxs)}"
        light_descendants = [FormShower.descendant_idxs(eventWise, lidx) for lidx in light_idxs]
        if use_jetInputs:
            # select only particles that have already been used in sourceidx
            light_descendants = [ldec.intersection(eventWise.JetInputs_SourceIdx) for ldec in light_descendants]
        # we also expect that the combination of the light descendants should be the heavy descendants
        all_light_descendants = light_descendants[0].union(light_descendants[1])
        assert all_light_descendants == heavy_descendants
        light_descendants_mass += [cluster_mass(eventWise, ldec) for ldec in light_descendants if ldec]
    return heavy_descendants_mass, light_descendants_mass


if __name__ == '__main__' and False:
    from tree_tagger import Components
    #ew = Components.EventWise.from_file("megaIgnore/scans/scan0_pt2.awkd")
    #jet_name = "SpectralFullJet696"
    ##jet_name = "SpectralFullJet684"
    #plot_largest_PT_pair(ew, jet_name, False)
    #ew = Components.EventWise.from_file("megaIgnore/scans/scan2.awkd")
    #jet_name = "SpectralFullJet1685"
    #plot_largest_PT_pair(ew, jet_name, False)
    #plot_smallest_angles(ew, jet_name, False)
    #plot_fat_jets(ew, jet_name,False)
    ew = Components.EventWise.from_file("megaIgnore/scans/scan0_pt1.awkd")
    jet_name = "HomeJet95"
    #plot_smallest_angles(ew, jet_name,False)
    #plot_fat_jets(ew, jet_name, False)
    #plot_largest_PT_pair(ew, jet_name, False)
    jet_name = "HomeJet363"
    plot_largest_PT_pair(ew, jet_name, False)


