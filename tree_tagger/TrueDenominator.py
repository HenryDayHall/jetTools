from tree_tagger import Components
from ipdb import set_trace as st
from tree_tagger import FormShower, InputTools, FormJets
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
    all_angles = np.full((len(roots), len(all_visibles)), np.inf)
    assigned = np.full(len(all_visibles), False)
    for i, vis in enumerate(all_visibles):
        in_showers = [n_s for n_s, shower in enumerate(visibles) if vis in shower]
        if len(in_showers) == 1:
            new_visibles[in_showers[0]].append(vis)
            assigned[i] = True
            continue
        p = np.array([eventWise.Px[vis], eventWise.Py[vis], eventWise.Rapidity[vis]])
        abs_p = np.sqrt(np.sum(p**2))
        abs_m = np.sqrt(np.sum(momentums[in_showers]**2, axis=1))
        cos_angle = np.sum(p*momentums[in_showers], axis=1)/(abs_p*abs_m)
        all_angles[in_showers, i] = cos_angle
    while not np.all(assigned):
        order = np.argsort([len(n) for n in new_visibles])
        for shower_n in order:
            angles = all_angles[shower_n]
            if np.any(np.isfinite(angles)):
                add_i = np.argmin(angles)
                assigned[add_i] = True
                all_angles[:, add_i] = np.inf
                new_visibles[shower_n].append(all_visibles[add_i])
                break
    new_visibles = awkward.fromiter(new_visibles)
    assert len(set(new_visibles.flatten())) == len(new_visibles.flatten())
    assert len(all_visibles) == len(new_visibles.flatten())
    return new_visibles
    
              
def make_weights(visibles, eventWise):
    idx_order = eventWise.JetInputs_SourceIdx.tolist()
    weights = np.full_like(idx_order, np.nan, dtype=float)
    for shower in visibles:
        if len(shower) == 0:
            continue
        weight = 1/len(shower)
        #print(f"{weight}; {shower}")
        for i in shower:
            weights[idx_order.index(i)] = weight
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
    path = InputTools.get_dir_name("Name eventWise file: ").strip()
    if path:
        eventWise = Components.EventWise.from_file(path)
        append_solution_and_weights(eventWise)
        solution_to_jet(eventWise)

