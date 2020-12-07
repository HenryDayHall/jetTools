from jet_tools.tree_tagger import Components
from ipdb import set_trace as st
from jet_tools.tree_tagger import FormShower, InputTools, FormJets
import numpy as np
import awkward

def get_visible_children(eventWise):
    visibles = set(eventWise.JetInputs_SourceIdx)
    showers = FormShower.get_showers(eventWise)
    results = [sorted(visibles.intersection(s.particle_idxs)) for s in showers]
    roots = [s.root_idxs[0] for s in showers]
    results = awkward.fromiter(results)
    assert len(visibles) == len(set(results.flatten())), f"Event {eventWise.selected_index}"
    return results, roots

              
def sort_overlap(visibles, roots, eventWise):
    visibles = visibles.astype(int)
    momentums = np.vstack((eventWise.Px[roots], eventWise.Py[roots], eventWise.Rapidity[roots])).T
    all_visibles = sorted(set(visibles.flatten()))
    new_visibles = [[] for _ in roots]
    for i, vis in enumerate(all_visibles):
        in_showers = [n_s for n_s, shower in enumerate(visibles) if vis in shower]
        if len(in_showers) == 1:
            new_visibles[in_showers[0]].append(vis)
            continue
        p = np.array([eventWise.Px[vis], eventWise.Py[vis], eventWise.Rapidity[vis]])
        abs_p = np.sqrt(np.sum(p**2))
        abs_m = np.sqrt(np.sum(momentums[in_showers]**2, axis=1))
        cos_angle = np.sum(p*momentums[in_showers], axis=1)/(abs_p*abs_m)
        new_visibles[in_showers[np.argmax(cos_angle)]].append(vis)
    new_visibles = awkward.fromiter(new_visibles)
    assert len(set(new_visibles.flatten())) == len(new_visibles.flatten())
    assert len(all_visibles) == len(new_visibles.flatten())
    return new_visibles
    

              
def make_weights(visibles, eventWise):
    idx_order = eventWise.JetInputs_SourceIdx.tolist()
    weights = np.full_like(idx_order, np.nan, dtype=float)
    all_indices = visibles.flatten().tolist()
    for shower in visibles:
        if len(shower) == 0:
            continue
        total_mass = np.sqrt(max(0,
                             np.sum(eventWise.Energy[shower])**2 -
                             np.sum(eventWise.Px[shower])**2 -
                             np.sum(eventWise.Py[shower])**2 -
                             np.sum(eventWise.Pz[shower])**2))
        shower_idxs = [idx_order.index(i) for i in shower]
        if total_mass == 0:
            weights[shower_idxs] = 1/len(shower_idxs)
            continue
        shower = list(shower)
        #print(f"{weight}; {shower}")
        for i, idx in zip(shower, shower_idxs):
            other_indices = [s for s in shower if s != i]
            # prevent nan masses
            mass = np.sqrt(max(0,
                           np.sum(eventWise.Energy[other_indices])**2 -
                           np.sum(eventWise.Px[other_indices])**2 -
                           np.sum(eventWise.Py[other_indices])**2 -
                           np.sum(eventWise.Pz[other_indices])**2))
            weights[idx] = (total_mass - mass)/total_mass
        # normalise the set
        weights[shower_idxs] = weights[shower_idxs]/np.sum(weights[shower_idxs])
    assert not np.any(np.isnan(weights))
    return weights


def append_solution_and_weights(eventWise):
    n_events = len(eventWise.JetInputs_SourceIdx)

    solutions = []
    perfect_denomonator = []
            
    for event_n in range(n_events):
        print(f"{event_n/n_events:.1%}", end='\r', flush=True)
        eventWise.selected_index = event_n
        visibles, roots = get_visible_children(eventWise)
        visibles = sort_overlap(visibles, roots, eventWise)
        solutions.append(visibles)
        weights = make_weights(visibles, eventWise)
        perfect_denomonator.append(weights)
    eventWise.append(JetInputs_PerfectDenominator=awkward.fromiter(perfect_denomonator),
                     Solution=awkward.fromiter(solutions))


def solution_to_jet(eventWise, idealised_jet_clusters=None, jet_name=None):
    eventWise.selected_index = None
    if idealised_jet_clusters is None:
        idealised_jet_clusters = eventWise.Solution
    if jet_name is None:
        jet_name = "SolJet"
    # updated_dict will be replaced in the first batch
    updated_dict = None
    n_events = len(idealised_jet_clusters)
    for event_n, jets_gids in enumerate(idealised_jet_clusters):
        if event_n % 100 == 0:
            print(f"{event_n/n_events:.1%}", end='\r', flush=True)
        eventWise.selected_index = event_n
        jet = FormJets.Traditional(eventWise)
        source_idx = eventWise.JetInputs_SourceIdx.tolist()
        for gids in jets_gids:
            if len(gids):
                idxs = [source_idx.index(i) for i in gids]
                jet._merge_complete_jet(idxs)
        jet = jet.split()
        updated_dict = FormJets.Traditional.create_updated_dict(jet, jet_name, event_n, eventWise, updated_dict)
    updated_dict = {name: awkward.fromiter(updated_dict[name]) for name in updated_dict}
    eventWise.append(**updated_dict)




if __name__ == '__main__':
    path = InputTools.get_file_name("Name eventWise file: ", '.awkd').strip()
    if path:
        eventWise = Components.EventWise.from_file(path)
        append_solution_and_weights(eventWise)
        solution_to_jet(eventWise)

