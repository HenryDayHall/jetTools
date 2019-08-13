""" a collection of scripts to assocate each jet to it's MC truth """
from ipdb import set_trace as st
from tree_tagger import FormJets, Components
import numpy as np

def allocate(tag_particles, jets):
    """
    each tag will be assigned to a jet
    both tag particles and jets must offer rap and phi methods
    """
    phi_distance = np.array([[tag.phi() - jet.phi for jet in jets]
                             for tag in tag_particles])
    rap_distance = np.array([[tag.rapidity() - jet.rap for jet in jets]
                             for tag in tag_particles])
    dist2 = np.square(phi_distance) + np.square(rap_distance)
    closest = np.argmin(dist2, axis=1)
    return closest


def from_hard_interaction(event, jets, hard_interaction_pids=[25, 35], tag_pids=None, include_antiparticles=True):
    """ tag jets based on particles emmited by the hard scattering 
        follows taggable partilces to last tagable decendant"""
    # if no tag pids given, anything that contains a b
    if tag_pids is None:
        tag_pids = np.genfromtxt('tree_tagger/contains_b_quark.csv', dtype=int)
    if include_antiparticles:
        tag_pids = np.concatenate((tag_pids, -tag_pids))
    hard_emmision = []
    # two possibilities, "hard particles" may be found in the event
    # or they are exculded from the particle list, and first gen is parentless
    hard_global_id = [p.global_id for p in event.particle_list
                      if p.pid in hard_interaction_pids]
    hard_emmision += [particle for particle in event.particle_list
                      if len(particle.mother_ids) == 0
                      or set(particle.mother_ids).intersection(hard_global_id)]
    tag_particles = [particle for particle in hard_emmision if particle.pid in tag_pids]
    remove_indices = []
    # now if there have decendants in the tag list favour the decendant
    for i in range(len(tag_particles)):
        particle = tag_particles[i]
        children = [c for c in event.particle_list if c.global_id in particle.daughter_ids]
        for child in children:
            if child.pid in tag_pids:
                remove_indices.append(i)
                tag_particles.append(child)
    for i in sorted(remove_indices, reverse=True):
        del tag_particles[i]
    jets_tags = [[] for _ in jets]
    if tag_particles: # there may not be any of the particles we wish to tag in the event
        closest_matches = allocate(tag_particles, jets)
        for match, particle in zip(closest_matches, tag_particles):
            jets_tags[match].append(particle.pid)
    return jets_tags


def add_tag(multievent_filename, multijet_filename):
    # this is a memory intense operation, so must be done in batches
    batch_size = 15
    batch_start = 0
    n_jets = 0
    jets_tags = []
    while True:
        events = Components.MultiParticleCollections.from_file(multievent_filename, batch_start, batch_start+batch_size)
        if events is None:
            break  # we reached the end
        print(batch_start, end=' ', flush=True)
        jets_by_event = FormJets.PsudoJets.multi_from_file(multijet_filename, batch_start, batch_start+batch_size)
        for event_n, (event, event_jets) in enumerate(zip(events, jets_by_event)):
            event_jets = event_jets.split()
            jets_tags += from_hard_interaction(event, event_jets)
            n_jets += len(event_jets)
        batch_start += batch_size
    try:
        assert n_jets == len(jets_tags)
    except Exception as e:
        print(e)
        st()
    tag_lengths = np.array([len(t) for t in jets_tags])
    mask = tag_lengths[:, None] > np.arange(tag_lengths.max())
    padded_tags = np.zeros_like(mask, dtype=int)
    padded_tags[mask] = np.concatenate(jets_tags)
    data = np.load(multijet_filename)
    np.savez(multijet_filename, **data, truth_tags=padded_tags)






