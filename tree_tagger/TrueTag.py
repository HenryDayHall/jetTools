""" a collection of scripts to assocate each jet to it's MC truth """
display = False
from ipdb import set_trace as st
if display:
    from tree_tagger import FormJets, DrawBarrel
from tree_tagger import Components, InputTools, Constants, FormShower
import numpy as np
import awkward
import os
import scipy.stats


def allocate(eventWise, jet_name, tag_idx, max_angle2, valid_jets=None):
    """
    In a given event each tag is allocated to a jet. 
    Each tag may only be allocated up to once, jets may recive multiple tags.
    If no valid jet is found inside the max_angle the tag will not be allocated.
    Unallocated tags are returned as -1.

    Parameters
    ----------
    eventWise : EventWise
        dataset containing locations of particles and jets
    tag_idx : iterable of ints
        the idx of the tag particles as found in the EventWise
    valid_jets : array like of ints
        The idx of the jets that can be tagged, as found in the eventWise
        If None then all jets are valid
        (Default value = None)
    jet_name : str
        The prefix of the jet vairables in the eventWise
    max_angle2 : float
        the maximium angle that a tag may be alloated to squared
        (a deltaR mesurment)
        

    Returns
    -------
    closest : numpy array of ints
        The indices of the closest jet to each tag,
        -1 if no sutable jet found
    
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
    phi_distance = Components.raw_to_angular_distance(phi_distance)
    rap_distance = np.array([[eventWise.Rapidity[tag_i] - jet_rap for jet_rap
                               in jet_raps]
                              for tag_i in tag_idx])
    dist2 = np.square(phi_distance) + np.square(rap_distance)
    try:
        closest = np.argmin(dist2, axis=1)
    except np.AxisError:
        assert len(dist2) == 0
        return dist2.reshape((0, 0))
    if valid_jets is not None:
        # revert to true indices
        closest = valid_jets[closest]
    # remove anything with too large an angle
    dist2_closest = np.min(dist2, axis=1)
    closest[dist2_closest > max_angle2] = -1
    return closest


def tag_particle_indices(eventWise, hard_interaction_pids=None, tag_pids=None, include_antiparticles=True):
    """
    Identify tag particles emmited by the hard scattering
    follows taggable partilces to last tagable decendant

    Parameters
    ----------
    eventWise : EventWise
        dataset containing locations of particles and jets
    hard_interaction_pids : list of ints
        These, along with the roots of the shower, are
        starting poitns for looking for tag particle chains
        (Default value = [25, 35] if found in event, else
         the contents of the proton is used [1, 2, 3, 4, -1, -2, -3, -4, 21])
    tag_pids : array like of ints, or the string 'hadrons'
        All the mcpids that should be considered to be part of a tag particle chain.
        If None, then just b-quarks are considered (5).
        If "hadrons" then all b-hadrons are considered.
        (Default value = None)
    include_antiparticles : bool
        If true, include the negative of any tag_pid,
        that is, also considere antiparticles to be tags
        (Default value = True)

    Returns
    -------
    tag_idx : iterable of ints
        the idx of the tag particles as found in the EventWise
    
    """
    assert eventWise.selected_index is not None
    # if no tag pids given, anything that contains a b
    if hard_interaction_pids is None:
        if 25 in eventWise.MCPID and 35 in eventWise.MCPID:
            hard_interaction_pids = [25, 35]
        else:
            hard_interaction_pids = [1, 2, 3, 4, -1, -2, -3, -4, 21]
    if tag_pids is None:
        tag_pids = np.array([-5])
    elif isinstance(tag_pids, str) and tag_pids == 'hadrons':
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
    return tag_idx


def add_tag_particles(eventWise, silent=False):
    """
    Appends the indices of the tag particles, found using tag_particle_indices
    to the eventWise.
    Operates inplace.

    Parameters
    ----------
    eventWise : EventWise
        dataset containing locations of particles and jets
    silent : bool
        Should the progress be printed?
        (Default value = False)
    
    """
    eventWise.selected_index = None
    name = "TagIndex"
    n_events = len(eventWise.MCPID)
    tags = list(getattr(eventWise, name, []))
    start_point = len(tags)
    if start_point >= n_events:
        print("Finished")
        return True
    end_point = n_events
    if not silent:
        print(f" Will stop at {end_point/n_events:.1%}")
    #tag_pids = np.genfromtxt('tree_tagger/contains_b_quark.csv', dtype=int)
    tag_pids = np.array([5])
    for event_n in range(start_point, end_point):
        if event_n % 10 == 0 and not silent:
            print(f"{event_n/n_events:.1%}", end='\r', flush=True)
        if os.path.exists("stop"):
            print(f"Completed event {event_n-1}")
            break
        eventWise.selected_index = event_n
        tags.append(tag_particle_indices(eventWise, tag_pids=tag_pids))
    content = {name: tags}
    # only one of these needed per file, so always append
    eventWise.append(**content)


def add_tags(eventWise, jet_name, max_angle, batch_length=100, jet_pt_cut=None, min_tracks=None, silent=False, append=True, overwrite=False):
    """
    Calculate and allocate the tags in the traditional way, using add_tag_particles. 

    Parameters
    ----------
    eventWise : EventWise
        dataset containing locations of particles and jets
    jet_name : str
        The prefix of the jet vairables in the eventWise
    max_angle : float
        the maximium angle that a tag may be alloated to a jet
        (a deltaR mesurment)
        If None then max_angle is drawn from Constants.py
        (Default value = None)
    batch_length: int
        max number of events to process
        (Default value = 100)
    jet_pt_cut : float
        Minimum pt value for a jet to be considered for tagging.
        If None, then no mimumum is applied.
        If not none then value is included in parameter name in eventWise.
        (Default value = None)
    min_tracks : int
        the minimum number of track for a jet to eb considered for tagging.
        If None then max_angle is drawn from Constants.py
        (Default value = None)
    silent : bool
        Should the progress be printed?
        (Default value = False)
    append : bool
        Should the results be appended to the eventWise?
        (Default value = True)
    overwrite : bool
        Should existing results be abandoned?
        (Default value = True)
        

    Returns
    -------
    (if not appending)
    hyperparameter_content : dict
        hyperparameter values for eventWise
    content: dict of awkward arrays
        content for eventWise
    
    """
    if min_tracks is None:
        min_tracks = Constants.min_ntracks
    if max_angle is None:
        max_angle = Constants.max_tagangle
    eventWise.selected_index = None
    if jet_pt_cut is not None:
        name = jet_name+f"_{int(jet_pt_cut)}Tags"
        namePID = jet_name+f"_{int(jet_pt_cut)}TagPIDs"
    else:
        name = jet_name + "_Tags"
        namePID = jet_name+f"_TagPIDs"
    n_events = len(getattr(eventWise, jet_name+"_Energy", []))
    if "TagIndex" not in eventWise.columns:
        add_tag_particles(eventWise, silent=silent)
    if overwrite:
        jet_tags = []
        jet_tagpids = []
    else:
        jet_tags = list(getattr(eventWise, name, []))
        jet_tagpids = list(getattr(eventWise, namePID, []))
    start_point = len(jet_tags)
    name_tagangle = jet_name + f"_TagAngle"
    hyperparameter_content = {name_tagangle: max_angle}
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
        print(f" Will stop at {end_point/n_events:.1%}")
    # name the vaiables to be cut on
    inputidx_name = jet_name + "_InputIdx"
    rootinputidx_name = jet_name+"_RootInputIdx"
    # will actually compare the square of the angle for speed
    max_angle2 = max_angle**2
    for event_n in range(start_point, end_point):
        if event_n % 10 == 0 and not silent:
            print(f"{event_n/n_events:.1%}", end='\r', flush=True)
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
            if jet_pt_cut is None:
                valid_jets = np.where(num_tracks > min_tracks-0.1)[0]
            else:
                valid_jets = np.where(np.logical_and(jet_pt > jet_pt_cut, num_tracks > min_tracks-0.1))[0]
        jets_tags = [[] for _ in jet_pt]
        if len(tags) > 0 and len(valid_jets) > 0:
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
    hyperparameter_content = {name_tagangle: max_angle}
    if append:
        eventWise.append(**content)
        eventWise.append_hyperparameters(**hyperparameter_content)
    else:
        return hyperparameter_content, content


def percent_pos(jet_idxs, parent_idxs, pos_idxs, weights=None):
    """
    The jet is regarded as a tree, the leaves of which are the input particles.
    Input particles may either be from a b-decendant (positive) or not (negative)
    All particle in the tree may be assigned a weight (else all weights are considred 1)
    Percentage positivity is propagated down the nodes of the tree to the root
    with the incoming values being moderated by the wieghts of their nodes.

    Parameters
    ----------
    jet_idxs : array like of ints
        integers that identify the jet constituents
    parent_idxs : array like of ints
        integers that identify the parent of each jet constituent
    pos_idxs : array like of ints
        integers that identify the leaf components considred positive
    weights : array like of floats
        weights for each jet constituent
         (Default value = None)

    Returns
    -------
    percents : array like of floats
        the percent positivity of each consituent
        in the same order as jet_idxs was given

    """
    if weights is None:
        weights = np.ones_like(jet_idxs)
    percents = np.zeros_like(jet_idxs, dtype=float)
    # first gen contain the pos idxs
    percents += [i in pos_idxs for i in jet_idxs]
    this_gen = np.fromiter((i not in parent_idxs for i in jet_idxs), bool)
    next_gen = np.empty_like(jet_idxs, dtype=bool)
    # make jets a list becuase we will index it
    jet_idxs = list(jet_idxs)
    while np.any(this_gen):
        next_gen[:] = False
        in_next_gen = set(parent_idxs[this_gen])
        in_next_gen.discard(-1)
        for nidx in in_next_gen:
            local_idx = jet_idxs.index(nidx)
            next_gen[local_idx] = True
            children = parent_idxs == nidx
            percents[local_idx] = np.sum(percents[children]*weights[children])/np.sum(weights[children])
        this_gen[:] = next_gen
    return percents


def get_root_rest_energies(root_idxs, energies, pxs, pys, pzs):
    """
    Find the energies (of anything really, but presumably jets) in the rest frame
    of particles identified a root particles
    
    Parameters
    ----------
    root_idxs : array like of ints
        indices identifying the root particles
    energies : array like of floats
        the energies of the particles
    pxs : array like of floats
        the momentum in the x direction of the particles
    pys : array like of floats
        the momentum in the y direction of the particles
    pzs : array like of floats
        the momentum in the z direction of the particles

    Returns
    -------
    energies : array like of floats
        the energies of the particles in the rest frame of the root
    """
    # don't give this empty events, it screws with indexing
    assert len(root_idxs.flatten()) > 0
    # if we are to use the roots as indices they must have this form
    assert isinstance(root_idxs, awkward.array.jagged.JaggedArray)
    momentum = np.vstack((pxs, pys, pzs)).T
    masses2 = energies**2 - pxs**2 - pys**2 - pzs**2
    pxs = pxs - pxs[root_idxs].flatten()
    pys = pys - pys[root_idxs].flatten()
    pzs = pzs - pzs[root_idxs].flatten()
    energies = np.sqrt(masses2 + pxs**2 + pys**2 + pzs**2)
    return energies


def add_inheritance(eventWise, jet_name, batch_length=100, silent=False, append=True):
    """
    Add the inheritance from each to the tagging particles
    Represents the portion of the energy that has been derived from the true particles
    in the rest frame of the root particle.
    The highest percentage inheritance*jet energy gets the itag.

    Parameters
    ----------
    eventWise : EventWise
        dataset containing locations of particles and jets
    jet_name : str
        The prefix of the jet vairables in the eventWise
    batch_length: int
        max number of events to process
        (Default value = 100)
    silent : bool
        Should the progress be printed?
        (Default value = False)
    append : bool
        Should the results be appended to the eventWise?
        (Default value = True)

    Returns
    -------
    (if append is false)
    content: dict of awkward arrays
        content for eventWise

    
    """
    eventWise.selected_index = None
    name = jet_name + "_Inheritance"
    tag_name = jet_name + "_ITags"
    n_events = len(getattr(eventWise, jet_name+"_Energy", []))
    jet_inhs = list(getattr(eventWise, name, []))
    jet_tags = list(getattr(eventWise, tag_name, []))
    start_point = len(jet_inhs)
    if start_point >= n_events:
        print("Finished")
        if append:
            return
        else:
            content = {}
            content[name] = awkward.fromiter(jet_inhs)
            return content
    end_point = min(n_events, start_point+batch_length)
    if not silent:
        print(f" Will stop at {end_point/n_events:.1%}")
    # will actually compare the square of the angle for speed
    for event_n in range(start_point, end_point):
        if event_n % 10 == 0 and not silent:
            print(f"{event_n/n_events:.1%}", end='\r', flush=True)
        if os.path.exists("stop"):
            print(f"Completed event {event_n-1}")
            break
        eventWise.selected_index = event_n
        jets_idxs = getattr(eventWise, jet_name + "_InputIdx")
        inhs_here = []
        tags_here = [[] for _ in jets_idxs]
        if len(tags_here) > 0:
            parents_idxs = getattr(eventWise, jet_name + "_Parent")
            roots_inputidxs = getattr(eventWise, jet_name + "_RootInputIdx")
            roots = awkward.fromiter([np.where(jet == root[0])[0] for jet, root
                                      in zip(jets_idxs, roots_inputidxs)])
            energies = getattr(eventWise, jet_name + "_Energy")
            pxs = getattr(eventWise, jet_name + "_Px")
            pys = getattr(eventWise, jet_name + "_Py")
            pzs = getattr(eventWise, jet_name + "_Pz")
            rf_energies = get_root_rest_energies(roots, energies, pxs, pys, pzs)
            root_energies = energies[roots]
            sourceidx = eventWise.JetInputs_SourceIdx.tolist()
            for b_idx in eventWise.BQuarkIdx:
                inhs_here.append([])
                b_decendants = [sourceidx.index(d) for d in
                                FormShower.descendant_idxs(eventWise, b_idx)
                                if d in sourceidx]
                for jet_idx, parent_idx, energy in zip(jets_idxs, parents_idxs, rf_energies):
                    ratings = percent_pos(jet_idx, parent_idx, b_decendants, energy)
                    inhs_here[-1].append(ratings)
                inhs_here[-1] = awkward.fromiter(inhs_here[-1])
                if (inhs_here[-1] > 0).any().any(): # if all the inheritances are 0, then no tags
                    # decide who gets the tag
                    root_scores = root_energies*inhs_here[-1][roots]
                    tags_here[np.argmax(root_scores)].append(b_idx)
        jet_inhs.append(awkward.fromiter(inhs_here))
        jet_tags.append(awkward.fromiter(tags_here))
    content = {}
    content[name] = awkward.fromiter(jet_inhs)
    content[tag_name] = awkward.fromiter(jet_tags)
    if append:
        eventWise.append(**content)
    else:
        return content


def add_mass_share(eventWise, jet_name, batch_length=100, silent=False, append=True):
    """
    Tagging procedure based on which jet has the largest portion of the tag's mass.

    Parameters
    ----------
    eventWise : EventWise
        dataset containing locations of particles and jets
    jet_name : str
        The prefix of the jet vairables in the eventWise
    batch_length: int
        max number of events to process
        (Default value = 100)
    silent : bool
        Should the progress be printed?
        (Default value = False)
    append : bool
        Should the results be appended to the eventWise?
        (Default value = True)

    Returns
    -------
    (if append is false)
    content: dict of awkward arrays
        content for eventWise

    
    """
    eventWise.selected_index = None
    name = jet_name + "_TagMass"
    tag_name = jet_name + "_MTags"
    n_events = len(getattr(eventWise, jet_name+"_InputIdx", []))
    jet_tagmass2 = list(getattr(eventWise, name, np.array([]))**2)
    jet_tags = list(getattr(eventWise, tag_name, []))
    start_point = len(jet_tagmass2)
    if start_point >= n_events:
        print("Finished")
        if append:
            return
        else:
            content = {}
            content[name] = awkward.fromiter(jet_tagmass2)**0.5
            content[tag_name] = awkward.fromiter(jet_tags)
            return content
    end_point = min(n_events, start_point+batch_length)
    if not silent:
        print(f" Will stop at {end_point/n_events:.1%}")
    for event_n in range(start_point, end_point):
        if event_n % 10 == 0 and not silent:
            print(f"{event_n/n_events:.1%}", end='\r', flush=True)
        if os.path.exists("stop"):
            print(f"Completed event {event_n-1}")
            break
        eventWise.selected_index = event_n
        jets_idxs = getattr(eventWise, jet_name + "_InputIdx")
        tags_here = [[] for _ in jets_idxs]
        mass2_here = [[] for _ in jets_idxs]
        this_tag = np.zeros(len(jets_idxs))
        if len(tags_here) > 0:
            this_tag[:] = 0.
            energies = eventWise.Energy
            pxs = eventWise.Px
            pys = eventWise.Py
            pzs = eventWise.Pz
            sourceidx = eventWise.JetInputs_SourceIdx.tolist()
            for b_idx in eventWise.BQuarkIdx:
                b_decendants = {sourceidx.index(d) for d in
                                FormShower.descendant_idxs(eventWise, b_idx)
                                if d in sourceidx}

                for jet_n, jet_idx in enumerate(jets_idxs):
                    b_in_jet = list(b_decendants.intersection(jet_idx))
                    mass2 = np.sum(energies[b_in_jet])**2 - np.sum(pxs[b_in_jet])**2 - \
                            np.sum(pys[b_in_jet])**2 - np.sum(pzs[b_in_jet])**2
                    mass2_here[jet_n].append(mass2)
                    this_tag[jet_n] = mass2  # IndexError
                if (this_tag > 0).any(): # if all the inheritances are 0, then no tags
                    # decide who gets the tag
                    tags_here[np.argmax(this_tag)].append(b_idx)
        jet_tagmass2.append(awkward.fromiter(mass2_here))
        jet_tags.append(awkward.fromiter(tags_here))
    content = {}
    content[name] = awkward.fromiter(jet_tagmass2)**0.5
    content[tag_name] = awkward.fromiter(jet_tags)
    if append:
        eventWise.append(**content)
    else:
        return content


def add_detectable_fourvector(eventWise, tag_name="BQuarkIdx"):
    """
    Add a list of detectable four vectors for the tags, 
    as present in the JetInputs.
    also add the indices themselves.
    
    Parameters
    ----------
    eventWise : EventWise
        dataset containing locations of particles and jets
    tag_name : str
        name of the column in the eventWise that countains the
        indices of the tags that we wish to use
        (Default="BQuarkIdx")
    """
    eventWise.selected_index = None
    name = "DetectableTag"
    tag_particles = getattr(eventWise, tag_name)
    # the leaves are the bits that are detected, the roots are the tag particles
    # group roots with common leaves
    leaves = []; roots = []
    px = []; py = []; pz = []; energy = []
    for i, tag_idxs in enumerate(tag_particles):
        eventWise.selected_index = i
        shower_inputs = set(eventWise.JetInputs_SourceIdx)
        all_energy = eventWise.Energy
        all_px = eventWise.Px
        all_py = eventWise.Py
        all_pz = eventWise.Pz
        per_tag = []
        for tag in tag_idxs:
            tag_decendants = FormShower.descendant_idxs(eventWise, tag)
            detectables = shower_inputs.intersection(tag_decendants)
            per_tag.append(detectables)
        # now work out what overlaps
        leaves.append([])
        roots.append([])
        energy.append([])
        px.append([])
        py.append([])
        pz.append([])
        unallocated = np.ones_like(tag_idxs, dtype=bool)
        while np.any(unallocated):
            position = next(i for i, free in enumerate(unallocated) if free)
            unallocated[position] = False
            # start from the first free tag
            seed = per_tag[position]
            # make a mask of what will be grouped with
            if not seed:  # this tag is undetectable
                continue
            group_with = [g for g, other in enumerate(per_tag)
                          if not seed.isdisjoint(other)]
            unallocated[group_with] = False
            roots[-1].append(tag_idxs[group_with].tolist())
            detectables = sorted(set().union(*(per_tag[g] for g in group_with)))
            leaves[-1].append(detectables)
            # now find the kinematics
            energy[-1].append(np.sum(all_energy[detectables]))
            px[-1].append(np.sum(all_px[detectables]))
            py[-1].append(np.sum(all_py[detectables]))
            pz[-1].append(np.sum(all_pz[detectables]))
    params = {name+"_Leaves": awkward.fromiter(leaves),
              name+"_Roots": awkward.fromiter(roots),
              name+"_Energy": awkward.fromiter(energy),
              name+"_Px": awkward.fromiter(px),
              name+"_Py": awkward.fromiter(py),
              name+"_Pz": awkward.fromiter(pz)}
    eventWise.append(**params)


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


def tags_to_quarks(eventWise, tag_idxs, quark_pdgids=[-5, 5]):
    """
    Find the quark that is most strongly assocated with each tag particle.

    Parameters
    ----------
    eventWise : EventWise
        dataset containing locations of particles and jets
    tag_idx : iterable of ints
        the idx of the tag particles as found in the EventWise
    quark_pdgids : iterable of ints
        list of mcpids considred to be quarks
         (Default value = [-5, 5])

    Returns
    -------
    quark_parents : numpy array of ints
        the idx of the quark parents as found in the EventWise

    """
    assert eventWise.selected_index is not None
    if np.all([pid in quark_pdgids for pid in eventWise.MCPID[tag_idxs]]):
        # check if the tags are already quarks
        return tag_idxs
    # fetch the angular variables for speed
    rapidity = eventWise.Rapidity
    phi = eventWise.Phi
    pids = eventWise.MCPID
    parents = eventWise.Parents
    children = eventWise.Children
    # fetch any quarks in the tag's parents
    quark_parents = []
    quark_distances = []
    for tag_idx in tag_idxs:
        this_qparents = []
        parent_stack = [tag_idx]
        while parent_stack:
            idx = parent_stack.pop()
            if pids[idx] in quark_pdgids:
                this_qparents.append(idx)
            else:
                parent_stack += parents[idx].tolist()
        this_qparents = list(set(this_qparents))
        # if there are multiple parents, abandon any that have b-quark decendants
        if len(this_qparents) > 1:
            last_b = [5 not in np.abs(pids[children[idx]]) for idx in this_qparents]
            if sum(last_b):  # there must be at least 1 remaining
                this_qparents = [this_qparents[i] for i, l in enumerate(last_b) if l]
        quark_parents.append(this_qparents)
        # if there is more than one quark on a tag calculate the deltaR to that quark
        # if these is just one parent, call the deltaR 0
        if len(this_qparents) == 1:
            quark_distances.append(np.zeros(1))
        elif len(this_qparents) > 1:
            distances = np.sqrt((rapidity[this_qparents] - rapidity[tag_idx])**2 +
                                Components.angular_distance(phi[this_qparents], phi[tag_idx])**2)
            quark_distances.append(distances.tolist())
        else:
            raise RuntimeError(f"Why does this tag have no quark parents? event_n = {eventWise.selected_index}, tag_idx = {tag_idx}.")
    # go through the quark distances, assigning each quark to its closest tag
    remaining = list(range(len(quark_parents)))
    while remaining:
        tag_num = np.argmin([np.min(quarks) for quarks in quark_distances])
        remaining.remove(tag_num)  # we found this one
        quark_distances[tag_num] = [np.inf]
        quark_num = np.argmin(quark_distances[tag_num])
        quark_idx = quark_parents[tag_num].pop(quark_num)
        quark_parents[tag_num] = quark_idx
        # chekc if this quark_idx is found elsewhere
        for num in remaining:
            try:
                quark_num2 = quark_parents[num].index(quark_idx)
            except ValueError:
                # it wasn't in this list, that's fine
                pass
            else:
                quark_parents[num].pop(quark_num2)
                quark_distances[num].pop(quark_num2)
                # check if there is now only one parent left at num
                if len(quark_distances[num]) == 1:
                    quark_distances[num][0] = 0. # set this distance to 0
    assert np.all([isinstance(q, int) for q in quark_parents])
    return quark_parents


