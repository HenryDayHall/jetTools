""" a collection of scripts to assocate each jet to it's MC truth """
display = False
from ipdb import set_trace as st
if display:
    from tree_tagger import FormJets, DrawBarrel
from tree_tagger import Components, InputTools, Constants
import numpy as np
import awkward
import os
import scipy.stats


def allocate(eventWise, jet_name, tag_idx, max_angle2, valid_jets=None):
    """
    each tag will be assigned to a jet

    Parameters
    ----------
    eventWise :
        
    jet_name :
        
    tag_idx :
        
    max_angle2 :
        

    Returns
    -------

    """
    root_name = jet_name + "_RootInputIdx"
    inputidx_name = jet_name + "_InputIdx"
    attr_name = jet_name + "_Phi"
    jet_phis = eventWise.match_indices(attr_name, root_name, inputidx_name).flatten()
    attr_name = jet_name + "_Rapidity"
    jet_raps = eventWise.match_indices(attr_name, root_name, inputidx_name).flatten()
    if valid_jets is not None:
        # check that a list of indices not a bool mask has been passed
        if len(valid_jets) == len(jet_raps):
            # should they be the same length every index must appear
            # just check for the last one
            assert len(jet_raps) - 1 in valid_jets
        # select only valid jets for comparison
        jet_phis = jet_phis[valid_jets]
        jet_raps = jet_raps[valid_jets]
    phi_distance = np.array([[eventWise.Phi[tag_i] - jet_phi for jet_phi
                               in jet_phis]
                              for tag_i in tag_idx])
    phi_distance[phi_distance > np.pi] = 2*np.pi - phi_distance[phi_distance > np.pi]
    rap_distance = np.array([[eventWise.Rapidity[tag_i] - jet_rap for jet_rap
                               in jet_raps]
                              for tag_i in tag_idx])
    dist2 = np.square(phi_distance) + np.square(rap_distance)
    closest = np.argmin(dist2, axis=1)
    if valid_jets is not None:
        # revert to true indices
        closest = valid_jets[closest]
    # remove anything with too large an angle
    dist2_closest = np.min(dist2, axis=1)
    closest[dist2_closest > max_angle2] = -1
    return closest


def tag_particle_indices(eventWise, hard_interaction_pids=[25, 35], tag_pids=None, include_antiparticles=True):
    """
    tag jets based on particles emmited by the hard scattering
        follows taggable partilces to last tagable decendant

    Parameters
    ----------
    eventWise :
        
    hard_interaction_pids :
         (Default value = [25)
    35] :
        
    tag_pids :
         (Default value = None)
    include_antiparticles :
         (Default value = True)

    Returns
    -------

    """
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
    i = 0
    while len(possible_tag) > 0:
        i+=1
        if i > 1000:
            st()
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
    return tag_idx


def add_tag_particles(eventWise, silent=False):
    eventWise.selected_index = None
    name = "TagIndex"
    n_events = len(eventWise.X)
    tags = list(getattr(eventWise, name, []))
    start_point = len(tags)
    if start_point >= n_events:
        print("Finished")
        return True
    end_point = n_events
    if not silent:
        print(f" Will stop at {100*end_point/n_events}%")
    eventWise.selected_index = None
    tag_pids = np.genfromtxt('tree_tagger/contains_b_quark.csv', dtype=int)
    for event_n in range(start_point, end_point):
        if event_n % 10 == 0 and not silent:
            print(f"{100*event_n/n_events}%", end='\r', flush=True)
        if os.path.exists("stop"):
            print(f"Completed event {event_n-1}")
            break
        eventWise.selected_index = event_n
        tags.append(tag_particle_indices(eventWise, tag_pids=tag_pids))
    content = {name: tags}
    # only one of these needed per file, so always append
    eventWise.append(**content)


def add_tags(eventWise, jet_name, max_angle, batch_length=100, jet_pt_cut=None, min_tracks=None, silent=False, append=True):
    """
    

    Parameters
    ----------
    eventWise :
        
    jet_name :
        
    max_angle :
        
    batch_length :
         (Default value = 100)

    Returns
    -------

    """
    if jet_pt_cut is None:
        jet_pt_cut = Constants.min_jetpt
    if min_tracks is None:
        min_tracks = Constants.min_ntracks
    if max_angle is None:
        max_angle = Constants.max_tagangle
    eventWise.selected_index = None
    name = jet_name+"_Tags"
    namePID = jet_name+"_TagPIDs"
    n_events = len(getattr(eventWise, jet_name+"_Energy", []))
    if "TagIndex" not in eventWise.columns:
        add_tag_particles(eventWise, silent=silent)
    jet_tags = list(getattr(eventWise, name, []))
    jet_tagpids = list(getattr(eventWise, namePID, []))
    start_point = len(jet_tags)
    hyperparameter_content = {jet_name + "_TagAngle": max_angle}
    if start_point >= n_events:
        print("Finished")
        content = {}
        content[name] = awkward.fromiter(jet_tags)
        content[namePID] = awkward.fromiter(jet_tagpids)
        if append:
            return
        else:
            return hyperparameter_content, content
    end_point = min(n_events, start_point+batch_length)
    if not silent:
        print(f" Will stop at {100*end_point/n_events}%")
    # name the vaiables to be cut on
    inputidx_name = jet_name + "_InputIdx"
    rootinputidx_name = jet_name+"_RootInputIdx"
    # will actually compare the square of the angle for speed
    max_angle2 = max_angle**2
    for event_n in range(start_point, end_point):
        if event_n % 10 == 0 and not silent:
            print(f"{100*event_n/n_events}%", end='\r', flush=True)
        if os.path.exists("stop"):
            print(f"Completed event {event_n-1}")
            break
        eventWise.selected_index = event_n
        # get the tags
        tags = eventWise.TagIndex
        # get the valiables to cut on
        jet_pt = eventWise.match_indices(jet_name+"_PT", inputidx_name, rootinputidx_name).flatten()
        if len(jet_pt) == 0:
            num_tracks = []
            valid_jets = []
        else:
            # note this actually counts num pesudojets, but for more than 2 that is sufficient
            num_tracks = Components.apply_array_func(len, getattr(eventWise, jet_name+"_PT")).flatten()
            valid_jets = np.where(np.logical_and(jet_pt > jet_pt_cut, num_tracks > min_tracks-0.1))[0]
        jets_tags = [[] for _ in jet_pt]
        if tags and len(valid_jets) > 0:
            # there may not be any of the particles we wish to tag in the event
            # or there may not be any jets
            closest_matches = allocate(eventWise, jet_name, tags, max_angle2, valid_jets)
        else:
            closest_matches = []
        # keep only the indices for space reasons
        for match, particle in zip(closest_matches, tags):
            if match != -1:
                jets_tags[match].append(particle)
        jet_tags.append(awkward.fromiter(jets_tags))
        tagpids = [eventWise.MCPID[jet] for jet in tags]
        jet_tagpids.append(awkward.fromiter(tagpids))
    content = {}
    content[name] = awkward.fromiter(jet_tags)
    content[namePID] = awkward.fromiter(jet_tagpids)
    hyperparameter_content = {jet_name + "_TagAngle": max_angle}
    if append:
        eventWise.append(**content)
        eventWise.append_hyperparameters(**hyperparameter_content)
    else:
        return hyperparameter_content, content


display=False  # note needs full simulation
if display:  # have to comment out to run without display
    def main():
        """ """
        from tree_tagger import Components, DrawBarrel
        repeat = True
        eventWise = Components.EventWise.from_file("megaIgnore/deltaRp4_akt.awkd")
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


