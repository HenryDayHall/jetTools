import numpy as np
import itertools
from matplotlib import pyplot as plt
import scipy.spatial
#import debug
from ipdb import set_trace as st
from tree_tagger import Constants, Components, FormShower


def filter(eventWise, jet_name, jet_idxs, track_cut=None, min_jet_PT=None):
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
    

def order_tagged_jets(eventWise, jet_name, ranking_variable="PT"):
    assert eventWise.selected_index is not None
    tagged_idxs = np.where([len(t) > 0 for t in getattr(eventWise, jet_name + '_Tags')])[0]
    tagged_idxs = filter(eventWise, jet_name, tagged_idxs)
    input_name = jet_name + '_InputIdx'
    root_name = jet_name + '_RootInputIdx'
    ranking_values = eventWise.match_indices(jet_name+'_'+ranking_variable, root_name, input_name).flatten()
    order = np.argsort(ranking_values[tagged_idxs])
    return tagged_idxs[order]


def jet_mass(eventWise, jet_name, jet_idxs):
    """ calcualte the invarient masses of jets """
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
    """ calculate the invarint mass of a group of particles """
    assert eventWise.selected_index is not None
    particle_idxs = list(particle_idxs)
    px = np.sum(eventWise.Px[particle_idxs])
    py = np.sum(eventWise.Py[particle_idxs])
    pz = np.sum(eventWise.Pz[particle_idxs])
    e = np.sum(eventWise.Energy[particle_idxs])
    return np.sqrt(e**2 - px**2 - py**2 - pz**2)


def smallest_angle_parings(eventWise, jet_name, jet_pt_cut):
    assert eventWise.selected_index is not None
    tagged_idxs = np.where([len(t) > 0 for t in getattr(eventWise, jet_name + '_Tags')])[0]
    tagged_idxs = filter(eventWise, jet_name, tagged_idxs, min_jet_PT=jet_pt_cut)
    num_tags = len(tagged_idxs)
    if num_tags < 2:
        return []
    elif num_tags == 2:
        return [tuple(tagged_idxs)]
    else:
        input_name = jet_name + '_InputIdx'
        root_name = jet_name + '_RootInputIdx'
        phi = eventWise.match_indices(jet_name+'_Phi', root_name, input_name)[tagged_idxs]
        rapidity = eventWise.match_indices(jet_name+'_Rapidity', root_name, input_name)[tagged_idxs]
        points = np.vstack((phi, rapidity)).T
        distances = scipy.spatial.distance.pdist(points)
        closest = np.argmin(distances)
        pairs = [(closest//num_tags, closest%num_tags)]
        if num_tags == 4:
            pairs.append(tuple(set(range(num_tags)) - set(pairs[0])))
        pairs = [(tagged_idxs[p[0]], tagged_idxs[p[1]]) for p in pairs]
        return pairs


def plot_smallest_angles(eventWise, jet_name, jet_pt_cut, show=True):
    eventWise.selected_index = None
    n_events = len(getattr(eventWise, jet_name+'_InputIdx'))
    pair_masses = []
    for event_n in range(n_events):
        if event_n % 10 == 0:
            print(f"{100*event_n/n_events}%", end='\r')
        eventWise.selected_index = event_n
        pairs = smallest_angle_parings(eventWise, jet_name, jet_pt_cut)
        pairs = np.array(pairs).flatten()
        if len(pairs) == 0:
            continue
        masses = jet_mass(eventWise, jet_name, pairs)
        pair_masses += (masses[::2] + masses[1::2]).tolist()
    plt.hist(pair_masses, bins=50, label="Masses of single jets at smallest angles")
    plt.title("Masses of pair of single jets at smallest angles")
    plt.xlabel("Mass (GeV)")
    plt.ylabel(f"Counts in {n_events} events")
    if show:
        plt.show()
            

def plot_PT_pairs(eventWise, jet_name, jet_pt_cut=None, show=True):
    eventWise.selected_index = None
    n_events = len(getattr(eventWise, jet_name+'_InputIdx'))
    fig, ax_array = plt.subplots(3, 2)
    ax_array = ax_array.flatten()
    # becuase we will use these to take indices from a numpy array they need to be lists 
    # not tuples
    pairs = [list(pair) for pair in itertools.combinations(range(4), 2)]
    pair_masses = [[] for _ in pairs]
    for event_n in range(n_events):
        if event_n % 100 == 0:
            print(f"{100*event_n/n_events}%", end='\r')
        eventWise.selected_index = event_n
        sorted_idx = order_tagged_jets(eventWise, jet_name)
        sorted_idx = filter(eventWise, jet_name, sorted_idx, track_cut=2, min_jet_PT=jet_pt_cut)
        n_jets = len(sorted_idx)
        if n_jets == 0:
            continue
        for i, pair in enumerate(pairs):
            if max(pair) < n_jets:
                pair_masses[i].append(jet_mass(eventWise, jet_name, sorted_idx[pair]))
    heavy, light = decendants_masses(eventWise)
    label = [jet_name, "heavy decendants", "light decendants"]
    for ax, pair, masses in zip(ax_array, pairs, pair_masses):
        data = [masses, heavy, light]
        title = f"Jets {pair[0]} and {pair[1]} (pT ordered), counts={len(masses)}"
        ax.set_title(title)
        ax.hist(data, density=True, bins=500, histtype='step', label=label)
    ax.legend()
    plt.xlabel("Mass (GeV)")
    plt.ylabel(f"Counts in {n_events} events")
    if show:
        plt.show()


def plot_doubleTagged_jets(eventWise, jet_name, show=True):
    eventWise.selected_index = None
    n_events = len(getattr(eventWise, jet_name+'_InputIdx'))
    masses = []
    for event_n in range(n_events):
        if event_n % 10 == 0:
            print(f"{100*event_n/n_events}%", end='\r')
        eventWise.selected_index = event_n
        tagged_idxs = np.where([len(t) == 2 for t in getattr(eventWise, jet_name + '_Tags')])[0]
        tagged_idxs = filter(eventWise, jet_name, tagged_idxs)
        if len(tagged_idxs) == 0:
            continue
        masses += jet_mass(eventWise, jet_name, tagged_idxs).tolist()
    plt.hist(masses, bins=50, label="Masses of fat jets")
    plt.title("Masses of fat jets")
    plt.xlabel("Mass (GeV)")
    plt.ylabel(f"Counts in {n_events} events")
    if show:
        plt.show()


def decendants_masses(eventWise, use_jetInputs=True):
    """ from the JetInputs, plot all tracks that originate from a light higgs,
    and all tracks that originate from the heavy higgs """
    heavy_higgs_pid = 35
    heavy_decendants_mass = []
    light_higgs_pid = 25
    light_decendants_mass = []
    n_events = len(eventWise.MCPID)
    for event_n in range(n_events):
        if event_n % 100 == 0:
            print(f"{100*event_n/n_events}%", end='\r', flush=True)
        eventWise.selected_index = event_n
        # we only expect one heavy higgs, excluding radation dups
        heavy_idx = np.where(eventWise.MCPID == heavy_higgs_pid)[0]
        heavy_idx = {Components.last_instance(eventWise, hidx) for hidx in heavy_idx}
        assert len(heavy_idx) == 1, f"Problem, expected 1 higgs pid {heavy_higgs_pid}, found {len(heavy_idx)}"
        heavy_idx = list(heavy_idx)[0]
        heavy_decendants = FormShower.decendant_idxs(eventWise, heavy_idx)
        if use_jetInputs:
            # select only particles that have already been used in sourceidx
            heavy_decendants = heavy_decendants.intersection(eventWise.JetInputs_SourceIdx)
        if heavy_decendants:
            heavy_decendants_mass.append(cluster_mass(eventWise, heavy_decendants))
        # we expect two light higgs
        light_idxs = np.where(eventWise.MCPID == light_higgs_pid)[0]
        light_idxs = {Components.last_instance(eventWise, lidx) for lidx in light_idxs}
        assert len(light_idxs) == 2, f"Problem, expected 2 higgs pid {light_higgs_pid}, found {len(light_idxs)}"
        light_decendants = [FormShower.decendant_idxs(eventWise, lidx) for lidx in light_idxs]
        if use_jetInputs:
            # select only particles that have already been used in sourceidx
            light_decendants = [ldec.intersection(eventWise.JetInputs_SourceIdx) for ldec in light_decendants]
        # we also expect that the combination of the light decendants should be the heavy decendants
        all_light_decendants = light_decendants[0].union(light_decendants[1])
        assert all_light_decendants == heavy_decendants
        light_decendants_mass += [cluster_mass(eventWise, ldec) for ldec in light_decendants if ldec]
    return heavy_decendants_mass, light_decendants_mass


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


