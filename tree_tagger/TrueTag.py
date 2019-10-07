""" a collection of scripts to assocate each jet to it's MC truth """
display = False
from ipdb import set_trace as st
if display:
    from tree_tagger import FormJets, Components, DrawBarrel, InputTools
else:
    from tree_tagger import Components, InputTools
import numpy as np
import awkward
import os

def allocate(eventWise, jet_name, tag_idx, max_angle2):
    """
    each tag will be assigned to a jet
    both tag particles and jets must offer rap and phi methods
    """
    phi_distance = np.vstack([[eventWise.Phi[tag_i] - np.mean(jet_phi) for jet_phi
                              in getattr(eventWise, jet_name+"_Phi")]
                             for tag_i in tag_idx])
    phi_distance[phi_distance > np.pi] = 2*np.pi - phi_distance[phi_distance > np.pi]
    rap_distance = np.vstack([[eventWise.Rapidity[tag_i] - np.mean(jet_rap) for jet_rap
                              in getattr(eventWise, jet_name+"_Rapidity")]
                             for tag_i in tag_idx])
    dist2 = np.square(phi_distance) + np.square(rap_distance)
    closest = np.argmin(dist2, axis=1)
    dist2_closest = np.min(dist2, axis=1)
    closest[dist2_closest > max_angle2] = -1
    return closest


def from_hard_interaction(eventWise, jet_name, hard_interaction_pids=[25, 35], tag_pids=None, include_antiparticles=True, max_angle2=np.inf):
    """ tag jets based on particles emmited by the hard scattering 
        follows taggable partilces to last tagable decendant"""
    # check if the selected event containes jets
    jet_energies = getattr(eventWise, jet_name+"_Energy")
    if len(jet_energies) == 0:
        return []
    # if no tag pids given, anything that contains a b
    if tag_pids is None:
        tag_pids = np.genfromtxt('tree_tagger/contains_b_quark.csv', dtype=int)
    if include_antiparticles:
        tag_pids = np.concatenate((tag_pids, -tag_pids))
    hard_emmision_idx = []
    # two possibilities, "hard particles" may be found in the event
    # or they are exculded from the particle list, and first gen is parentless
    hard_idx = [i for i, pid in enumerate(eventWise.MCPID)
                if pid in hard_interaction_pids]
    hard_emmision_idx += [i for i, parents in enumerate(eventWise.Parents)
                          if len(parents) == 0
                          or set(parents).intersection(hard_idx)]
    possible_tag = [i for i in hard_emmision_idx if eventWise.MCPID[i] in tag_pids]
    tag_idx = []
    # now if there have decendants in the tag list favour the decendant
    convergent_roots = 0
    while len(possible_tag) > 0:
        possible = possible_tag.pop()
        eligable_children = [child for child in eventWise.Children[possible]
                             if eventWise.MCPID[child] in tag_pids]
        if eligable_children:
            possible_tag += eligable_children
        elif possible not in tag_idx:
            tag_idx.append(possible)
        else:
            convergent_roots += 1
    if convergent_roots:
        pass
        #print(f"{convergent_roots} b hadrons merge, may not form expected number of jets")
    jets_tags = [[] for _ in jet_energies]
    if tag_idx: # there may not be any of the particles we wish to tag in the event
        closest_matches = allocate(eventWise, jet_name, tag_idx, max_angle2)
    else:
        closest_matches = []
    # keep only the indices for space reasons
    for match, particle in zip(closest_matches, tag_idx):
        if match != -1:
            jets_tags[match].append(particle)
    return jets_tags


def add_tags(eventWise, jet_name, max_angle, batch_length=100):
    eventWise.selected_index = None
    name = jet_name+"_Tags"
    namePID = jet_name+"_TagPIDs"
    n_events = len(getattr(eventWise, jet_name+"_Energy", []))
    jet_tags = list(getattr(eventWise, name, []))
    jet_tagpids = list(getattr(eventWise, namePID, []))
    start_point = len(jet_tags)
    if start_point >= n_events:
        print("Finished")
        return True
    end_point = min(n_events, start_point+batch_length)
    print(f" Will stop at {100*end_point/n_events}%")
    # this is a memory intense operation, so must be done in batches
    eventWise.selected_index = None
    tag_pids = np.genfromtxt('tree_tagger/contains_b_quark.csv', dtype=int)
    max_angle2 = max_angle**2
    for event_n in range(start_point, end_point):
        if event_n % 10 == 0:
            print(f"{100*event_n/n_events}%", end='\r', flush=True)
        if os.path.exists("stop"):
            print(f"Completed event {event_n-1}")
            break
        eventWise.selected_index = event_n
        tags = from_hard_interaction(eventWise, jet_name, tag_pids=tag_pids, max_angle2=max_angle2)
        jet_tags.append(awkward.fromiter(tags))
        tagpids = [eventWise.MCPID[jet] for jet in tags]
        jet_tagpids.append(awkward.fromiter(tagpids))
    content = {}
    content[name] = awkward.fromiter(jet_tags)
    content[namePID] = awkward.fromiter(jet_tagpids)
    try:
        eventWise.append(content)
    except Exception:
        print("Problem")
        return content

display=True
if display:  # have to comment out to run without display

    def main():
        from tree_tagger import Components, DrawBarrel
        repeat = True
        eventWise = Components.EventWise.from_file("/home/henry/lazy/dataset2/h1bBatch2_particles.awkd")
        jet_name = "HomeJet"
        while repeat:
            eventWise.selected_index = int(input("Event num: "))
            outer_pos, tower_pos = DrawBarrel.plot_tracks_towers(eventWise)
            tags_by_jet = from_hard_interaction(eventWise, jet_name)
            tag_particle_idxs = []
            tag_jet_idx = []
            for j, tags in enumerate(tags_by_jet):
                tag_particle_idxs += tags
                tag_jet_idx += [j for _ in tags]
            # start by putting the tag particles on the image
            tag_distance = np.max(np.abs(tower_pos)) * 1.2
            tag_colours = DrawBarrel.colour_set(len(tag_particle_idxs))
            # set fat jets to share a colour
            for i, tag in enumerate(tag_jet_idx):
                locations = [j for j, other_idx in enumerate(tag_jet_idx)
                             if other_idx == tag]
                for l in locations:
                    tag_colours[l] = tag_colours[i]
            for colour, idx in zip(tag_colours, tag_particle_idxs):
                pos = np.array([eventWise.X[idx], eventWise.Y[idx], eventWise.Z[idx],
                                eventWise.Px[idx], eventWise.Py[idx], eventWise.Pz[idx]])
                # check the position is off the origin
                if np.sum(np.abs(pos[:3])) <= 0.:
                    pos[:3] = pos[3:]  #make the momentum the position
                pos *= tag_distance/np.linalg.norm(pos)
                DrawBarrel.add_single(pos, colour, name=f'tag({eventWise.MCPID[idx]})', scale=300)
            # highlight the towers and tracks assocated with each tag
            for colour, jet_idx in zip(tag_colours, tag_jet_idx):
                external_jetidx = getattr(eventWise, jet_name + "_Child1")[jet_idx] < 0
                input_jetidx = getattr(eventWise, jet_name + "_InputIdx")[jet_idx][external_jetidx]
                particle_idx = eventWise.JetInputs_SourceIdx[input_jetidx]
                tower_idx = eventWise.Particle_Tower[particle_idx]
                tower_idx = tower_idx[tower_idx>0]
                track_idx = eventWise.Particle_Track[particle_idx]
                track_idx = track_idx[track_idx>0]
                DrawBarrel.highlight_indices(tower_pos, tower_idx, colours=colour, colourmap=False)
                DrawBarrel.highlight_indices(outer_pos, track_idx, colours=colour, colourmap=False)

            repeat = InputTools.yesNo_question("Again? ")


    if __name__ == '__main__':
        main()




