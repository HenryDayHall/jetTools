""" Module for optimising the clustering process without gradient decent """
import nevergrad as ng
from ipdb import set_trace as st
from jet_tools.src import InputTools, CompareClusters, Components, FormJets, TrueTag, Constants, FormShower
import numpy as np


if __name__ == '__main__':
    eventWise_name = InputTools.get_dir_name("Name the eventWise: ").strip()
    eventWise = Components.EventWise.from_file(eventWise_name)
    n_events = len(eventWise.Px)
    # check we have object needed for tagging
    if "DetectableTag_Roots" not in eventWise.columns:
        TrueTag.add_detectable_fourvector(eventWise, silent=False)
    min_tracks = Constants.min_ntracks
    min_jetpt = Constants.min_pt
    max_angle = Constants.max_tagangle
    max_angle2 = max_angle**2
    # get the initial clustering parameters
    spectral_jet_params = dict(ExpofPTMultiplier=0,
                               ExpofPTPosition='input',
                               ExpofPTFormat='Luclus',
                               NumEigenvectors=np.inf,
                               StoppingCondition='meandistance',
                               EigNormFactor=1.5,
                               Laplacien='symmetric',
                               DeltaR=3.2,
                               AffinityType='exponent',
                               AffinityExp=1.,
                               Sigma=.6,
                               CombineSize='sum',
                               EigDistance='abscos',
                               CutoffKNN=None,
                               CutoffDistance=None,
                               PhyDistance='angular')
    jet_class = FormJets.SpectralMean
    loss = np.nan
    for event_n in range(n_events):
        print(f"{event_n/n_events:.2%}, {loss}", flush=True, end='\r')
        eventWise.selected_index = event_n
        jets = jet_class(eventWise, dict_jet_params=spectral_jet_params,
                         assign=True)
        # get the tags
        tags = eventWise.BQuarkIdx.tolist()
        tag_phi = eventWise.Phi[tags]
        tag_rapidity = eventWise.Rapidity[tags]
        # get the valiables to cut on
        jet_roots = [i for i, ii in enumerate(jets.InputIdx) if ii in jets.root_jetInputIdxs]
        jet_pt = jets.PT[jet_roots]
        if len(jet_pt) == 0:
            continue
        jet_rapidity = jets.Rapidity[jet_roots]
        jet_phi = jets.Phi[jet_roots]
        jets = jets.split()
        source_idx = eventWise.JetInputs_SourceIdx.tolist()
        len_source = len(source_idx)
        input_idxs = [jet.InputIdx for jet in jets]
        jet_idxs = [source_idx[idx[idx < len_source]] for idx in input_idxs]
        num_tracks = [len(jet) for jet in jet_idxs]
        valid_jets = np.where(num_tracks > min_tracks-0.1)[0]
        jets_tags = [[] for _ in jet_pt]
        if len(tags) == 0 or len(valid_jets) == 0:
            # there may not be any of the particles we wish to tag in the event
            # or there may not be any jets
            continue
        closest_matches = TrueTag.allocate(jet_phi, jet_rapidity, tag_phi, tag_rapidity,
                                           max_angle2, valid_jets)
        # keep only the indices for space reasons
        for match, particle in zip(closest_matches, tags):
            if match != -1:
                jets_tags[match].append(particle)
        tag_groups = eventWise.DetectableTag_Roots
        # need to calculate the mass of each tag group in each jet
        jet_masses2 = np.zeros((len(jet_idxs), len(tags)))
        energy = eventWise.Energy
        px = eventWise.Px
        py = eventWise.Py
        pz = eventWise.Pz
        for tag_n, b_idx in enumerate(tags):
            b_decendants = {source_idx.index(d) for d in
                            FormShower.descendant_idxs(eventWise, b_idx)
                            if d in source_idx}
            for jet_n, jet_idx in enumerate(jet_idxs):
                b_in_jet = list(b_decendants.intersection(jet_idx))
                mass2 = np.sum(energy[b_in_jet])**2 - np.sum(px[b_in_jet])**2 - \
                        np.sum(py[b_in_jet])**2 - np.sum(pz[b_in_jet])**2
                jet_masses2[jet_n, tag_n] = mass2
        jet_masses = np.sqrt(jet_masses)
        # add the pt cut onto the valid jets
        valid_jets = valid_jets[jet_pt[valid_jets] > min_jetpt]
        seperate_jets, matched_jets = CompareClusters.match_jets(tags, jets_tags, tag_groups,
                                                                 jet_masses, valid_jets)
        mass = np.zeros((len(matched_jets), 2))
        for tag_n, tag_leaves in enumerate(eventWise.DetectableTag_Leaves):
            tag_mass2, bg_mass2, _, _, _, _, _, _, _ = \
                   CompareClusters.event_detectables(tag_leaves, matched_jets[tag_n],
                                                     input_idxs, source_idx, energy,
                                                     px, py, pz, mass_only=True)
            mass[tag_n, 0] += bg_mass2
            mass[tag_n, 1] += tag_mass2
        mass = np.sqrt(mass)
        mass[:, 1] = eventWise.DetectableTag_Mass - mass[:, 1]
        assert np.all(mass > 0)
        loss = np.sum(mass)
        
                                                   
                                             


def onemax(*x):
    return len(x) - x.count(1)

# Discrete, ordered
variables = list(ng.p.TransitionChoice(list(range(7))) for _ in range(10))
instrum = ng.p.Instrumentation(*variables)
optimizer = ng.optimizers.DiscreteOnePlusOne(parametrization=instrum, budget=100, num_workers=1)

recommendation = optimizer.provide_recommendation()
for _ in range(optimizer.budget):
    x = optimizer.ask()
    loss = onemax(*x.args, **x.kwargs)
    optimizer.tell(x, loss)

recommendation = optimizer.provide_recommendation()
print(recommendation.value)
# >>> ((1, 1, 0, 1, 1, 4, 1, 1, 1, 1), {})
print(recommendation.args)
# >>> (1, 1, 0, 1, 1, 4, 1, 1, 1, 1)
