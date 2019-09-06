""" a collection of scripts to assocate each jet to it's MC truth """
from ipdb import set_trace as st
from tree_tagger import FormJets, Components, DrawBarrel, InputTools
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


def from_hard_interaction(event, jets, hard_interaction_pids=[25, 35], tag_pids=None, include_antiparticles=True, full_return=False):
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
    possible_tag = [particle for particle in hard_emmision if particle.pid in tag_pids]
    tag_particles = []
    # now if there have decendants in the tag list favour the decendant
    convergent_roots = 0
    while len(possible_tag) > 0:
        possible = possible_tag.pop()
        eligable_children = [child for child in event.particle_list
                             if child.global_id in possible.daughter_ids
                             and child.pid in tag_pids]
        if eligable_children:
            possible_tag += eligable_children
        elif possible not in tag_particles:
            tag_particles.append(possible)
        else:
            convergent_roots += 1
    if convergent_roots:
        print(f"{convergent_roots} b hadrons merge, may not form expected number of jets")
    jets_tags = [[] for _ in jets]
    if tag_particles: # there may not be any of the particles we wish to tag in the event
        closest_matches = allocate(tag_particles, jets)
    else:
        closest_matches = []
    if full_return:  # just return the particles and jets, no clever index stuff
        tag_jets = [jets[match] for match in closest_matches]
        return tag_particles, tag_jets
    # keep only the indices for space reasons
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
    tag_lengths = np.array([len(t) for t in jets_tags])
    mask = tag_lengths[:, None] > np.arange(tag_lengths.max())
    padded_tags = np.zeros_like(mask, dtype=int)
    padded_tags[mask] = np.concatenate(jets_tags)
    data = np.load(multijet_filename)
    np.savez(multijet_filename, **data, truth_tags=padded_tags)


def main():
    from tree_tagger import ReadSQL, FormJets, DrawBarrel
    num_tags = []
    num_convergent_roots = []
    hard_interaction_pids=[25, 35]
    tag_pids = np.genfromtxt('tree_tagger/contains_b_quark.csv', dtype=int)
    for event_num in range(999):
        if event_num %10 == 0:
            print('.', end='', flush=True)
            if event_num %100 ==0:
                print(f"set num_tags = {set(num_tags)}")
        try:
            event, track_list, tower_list, observations = ReadSQL.main(event_num)
        except AssertionError:
            continue
        except Exception:
            break
        hard_emmision = []
        # two possibilities, "hard particles" may be found in the event
        # or they are exculded from the particle list, and first gen is parentless
        hard_global_id = [p.global_id for p in event.particle_list
                          if p.pid in hard_interaction_pids]
        hard_emmision += [particle for particle in event.particle_list
                          if len(particle.mother_ids) == 0
                          or set(particle.mother_ids).intersection(hard_global_id)]
        possible_tag = [particle for particle in hard_emmision if particle.pid in tag_pids]
        tag_particles = []
        # now if there have decendants in the tag list favour the decendant
        convergent_roots = 0
        while len(possible_tag) > 0:
            possible = possible_tag.pop()
            eligable_children = [child for child in event.particle_list
                                 if child.global_id in possible.daughter_ids
                                 and child.pid in tag_pids]
            if eligable_children:
                possible_tag += eligable_children
            elif possible not in tag_particles:
                tag_particles.append(possible)
            else:
                convergent_roots += 1
        if len(tag_particles) != 4:
            st()
        num_tags.append(len(tag_particles))
        num_convergent_roots.append(num_convergent_roots)
    st()
    np.savetxt("woop_num_tags.csv", num_tags)
    print(np.mean(num_tags))

def alt_main():
    from tree_tagger import ReadSQL, FormJets, DrawBarrel
    repeat = True
    while repeat:
        event_num = int(input("Event num: "))
        event, track_list, tower_list, observations = ReadSQL.main(event_num)
        psudojet = FormJets.PsudoJets(observations, deltaR=0.4)
        psudojet.assign_mothers()
        pjets = psudojet.split()
        outer_pos, tower_pos = DrawBarrel.plot_tracks_towers(track_list, tower_list)
        tag_particles, tag_jets = from_hard_interaction(event, pjets, full_return=True)
        # start by putting the tag particles on the image
        tag_distance = np.max(np.abs(tower_pos)) * 1.2
        tag_colours = DrawBarrel.colour_set(len(tag_particles))
        # set fat jets to share a colour
        for i, jet in enumerate(tag_jets):
            locations = [j for j, other_jet in enumerate(tag_jets)
                         if other_jet == jet]
            for l in locations:
                tag_colours[l] = tag_colours[i]
        for colour, particle in zip(tag_colours, tag_particles):
            pos = np.array([particle.x, particle.y, particle.z,
                            particle.px, particle.py, particle.pz])
            pos *= tag_distance/np.linalg.norm(pos)
            DrawBarrel.add_single(pos, colour, name=f'tag({particle.pid})', scale=300)
        # highlight the towers and tracks assocated with each tag
        for colour, jet in zip(tag_colours, tag_jets):
            jet_oids = jet.global_obs_ids[jet.global_obs_ids!=-1] 
            tower_indices = [i for i, t in enumerate(tower_list) if t.global_obs_id in jet_oids]
            track_indices = [i for i, t in enumerate(track_list) if t.global_obs_id in jet_oids]
            DrawBarrel.highlight_indices(tower_pos, tower_indices, colours=colour, colourmap=False)
            DrawBarrel.highlight_indices(outer_pos, track_indices, colours=colour, colourmap=False)

        repeat = InputTools.yesNo_question("Again? ")


if __name__ == '__main__':
    main()




