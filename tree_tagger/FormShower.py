""" Tools to turn clusters of particles into showers """
from ipdb import set_trace as st
import networkx
from tree_tagger import PDGNames, ReadSQL, ReadHepmc, DrawTrees
import itertools
from matplotlib import pyplot as plt
import numpy as np

class Shower:
    """
    Object to hold a shower of particles
    
    only keeps a list of the particle particle_idxs, parents, children and PDGparticle_idxs.

    Parameters
    ----------

    Returns
    -------

    """
    def __init__(self, particle_idxs, parents, children, labels):
        self.amalgam = False
        self.particle_idxs = particle_idxs
        self.parents = parents
        self.children = children
        self.labels = labels
        self.ranks = None  # exspensive, create as needed with find_ranks()
        self._find_roots()

    def __len__(self):
        return len(self.particle_idxs)

    def amalgamate(self, other_shower):
        """
        

        Parameters
        ----------
        other_shower :
            

        Returns
        -------

        """
        self.amalgam = True
        total_particle_idxs = len(set(self.particle_idxs).union(set(other_shower.particle_idxs)))
        next_free_sIndex = len(self.particle_idxs)
        self.particle_idxs.resize(total_particle_idxs)
        self.labels.resize(total_particle_idxs)
        for oIndex, oID in enumerate(other_shower.particle_idxs):
            if oID in self.particle_idxs:  #check for agreement
                sIndex = list(self.particle_idxs).index(oID)
                assert self.parents[sIndex] == other_shower.parents[oIndex]
                assert self.children[sIndex] == other_shower.children[oIndex]
                assert self.labels[sIndex] == other_shower.labels[oIndex]
            else:  # add it on
                sIndex = next_free_sIndex
                next_free_sIndex += 1
                self.particle_idxs[sIndex] = oID
                self.parents.append(other_shower.parents[oIndex])
                self.children.append(other_shower.children[oIndex])
                self.labels[sIndex] = other_shower.labels[oIndex]
        self._find_roots()

    @property
    def n_particles(self):
        """int: the number of particles at all points of the shower"""
        return len(self.particle_idxs)

    def _find_roots(self):
        """
        Demand the shower identify it's root.
        This is stored as an internal variable.

        Parameters
        ----------

        Returns
        -------

        """
        root_idxs = get_roots(self.particle_idxs, self.parents)
        if not self.amalgam:
            assert len(root_idxs) == 1, "There should only be one root to a shower"
        self.root_idxs = root_idxs
        list_idxs = list(self.particle_idxs)
        self.root_local_idxs = [list_idxs.index(r) for r in root_idxs]

    @property
    def roots(self):
        """ """
        msg = "changed to root_idxs or root_local_idxs for clarity"
        raise AttributeError(msg)


    def find_ranks(self):
        """
        Demand the shower identify the rang of each particle.
        The rank of a particle is the length of the shortest distance to the root.

        Parameters
        ----------

        Returns
        -------

        
        """
        # put the first rank in
        current_rank = self.root_local_idxs
        rank_n = 0
        ranks = np.full_like(self.particle_idxs, -1, dtype=int)
        ranks[current_rank] = rank_n
        list_particle_idxs = list(self.particle_idxs)  # somtimes this is an array
        has_decendants = True
        while has_decendants:
            rank_n += 1
            decendant_particle_idxs = [child for index in current_rank
                                       for child in self.children[index]
                                       if child in self.particle_idxs]
            current_rank = []
            for child in decendant_particle_idxs:
                index = list_particle_idxs.index(child)
                # don't overwite a rank, so in a loop the lowers rank stands
                # also needed to prevent perpetual loops
                if ranks[index] == -1:
                    current_rank.append(index)
            ranks[current_rank] = rank_n
            has_decendants = len(current_rank) > 0
        assert -1 not in ranks
        # finally, promote all end state particles to the highest rank
        ranks[self.ends] = rank_n
        self.ranks = ranks
        return ranks

    def graph(self):
        """Turn the shower into a dotgraph"""
        assert len(self.particle_idxs) == len(self.parents)
        assert len(self.particle_idxs) == len(self.children)
        assert len(self.particle_idxs) == len(self.labels)
        return DrawTrees.DotGraph(self)

    @property
    def outside_connections(self):
        """ """
        raise AttributeError("you want outside_connection_idxs, but check you are useing particle_idx not local shower index")

    @property
    def outside_connection_idxs(self):
        """
        Function that anouches which particles have perantage from outside this shower
        Includes the root

        Parameters
        ----------

        Returns
        -------

        """
        outside_idxs = []
        for idx, parents_here in zip(self.particle_idxs, self.parents):
            if not np.all([m in self.particle_idxs for m in parents_here]):
                outside_idxs.append(idx)
        return outside_idxs

    @property
    def ends(self):
        """ """
        _ends = []
        for i, children_here in enumerate(self.children):
            if np.all([child is None for child in children_here]):
                _ends.append(i)
        return _ends

    @property
    def flavour(self):
        """ """
        flavours = self.labels[self.root_local_idxs]
        return '+'.join(flavours)
        

def get_showers(eventWise, exclude_pids=[2212, 25, 35]):
    """
    From each root split the decendants into showers
    Each particle can only belong to a single shower.

    Parameters
    ----------
    databaseName : string
        Path and file name of database
    exclude_MCPparticle_idxs : list like of ints
        Pparticle_idxs that will be cut out before splitting into showers
        (Default value = [2212, 25, 35])
    eventWise :
        
    exclude_pids :
         (Default value = [2212)
    25 :
        
    35] :
        

    Returns
    -------

    
    """
    # remove any stop pids
    mask = [p not in exclude_pids for p in eventWise.PID]
    particle_idxs = np.where(mask)[0]
    parent_ids = eventWise.Parents[mask]
    child_ids = eventWise.Children[mask]
    pids = eventWise.PID[mask]
    # check that worked
    remaining_pids = set(pids)
    for exclude in exclude_pids:
        assert exclude not in remaining_pids
    # now where possible convert labels to names
    pdg = PDGNames.IDConverter()
    labels = np.array([pdg[x] for x in pids])
    # now we have values for the whole event,
    # but we want to split the event into showers
    # at start all particles are allocated to a diferent shower
    showers = []
    root_gids = get_roots(particle_idxs, parent_ids)
    print(f"Found {len(root_gids)} root_gids")
    list_particle_idxs = list(particle_idxs)
    for i, root_gid in enumerate(root_gids):
        root_idx = list_particle_idxs.index(root_gid)
        shower_indices = [root_idx]
        to_follow = [root_idx]
        while len(to_follow) > 0:
            next_gen = []
            for index in to_follow:
                # be careful not to include forbiden particles
                next_gen += [list_particle_idxs.index(child) for child in child_ids[index]
                             if child in particle_idxs]
            # prevent loops
            next_gen = list(set(next_gen))
            next_gen = [i for i in next_gen if i not in shower_indices]
            shower_indices += next_gen
            # print(f"Indices in shower = {len(shower_indices)}", flush = True)
            to_follow = next_gen
        assert len(set(shower_indices)) == len(shower_indices)
        new_shower = Shower(particle_idxs[shower_indices],
                            [parent_ids[s] for s in shower_indices], 
                            [child_ids[s] for s in shower_indices],
                            labels[shower_indices])
        showers.append(new_shower)
    return showers


def get_roots(particle_ids, parents):
    """
    From a list of particle particle_idxs and a list of parent particle_idxs determin root particles
    
    A root particle is one whos parents are both from outside the particle list.

    Parameters
    ----------
    particle_ids : numpy array of ints
        The unique id of each particle
    parents : 2D numpy array of ints
        Each row contains the particle_idxs of two parents of each particle in particle_idxs
        These can be none

    Returns
    -------

    
    """
    roots = []
    for gid, parents_here in zip(particle_ids, parents):
        if not np.any([m in particle_ids for m in parents_here]):
            roots.append(gid)
    return roots


# ignore not working
def make_tree(particle_idxs, parents, children, labels):
    """
    It's possible this is working better than I think it is ...
    Just the data was screwy
    Anyway it has been supassed by dot files.

    Parameters
    ----------
    particle_idxs : numpy array of ints
        The unique id of each particle
    parents : 2D numpy array of ints
        Each row contains the particle_idxs of two parents of each particle in particle_idxs
        These can be none
    children : 2D numpy array of ints
        Each row contains the particle_idxs of two children of each particle in particle_idxs
        These can be none
    labels : numpy array of ints
        the MC PDG ID of each particle in particle_idxs

    Returns
    -------

    
    """
    graph =  networkx.Graph()
    graph.add_nodes_from(particle_idxs)
    for this_id, this_parents, this_children in zip(particle_idxs, parents, children):
        parent_edges = [(par_id, this_id) for par_id in this_parents if par_id in particle_idxs]
        graph.add_edges_from(parent_edges)
        child_edges = [(this_id, chi_id) for chi_id in this_children if chi_id in particle_idxs]
        graph.add_edges_from(child_edges)
    label_dict = {i:l for i, l in zip(particle_idxs, labels)}
    networkx.relabel_nodes(graph, label_dict)
    return graph


# don't use - hits recursion limit
def recursive_grab(seed_id, particle_idxs, relatives):
    """
    Not a good idea in python

    Parameters
    ----------
    seedID : int
        root particle ID
    particle_idxs : numpy array of ints
        particle_idxs of all particles considered
    relatives : 2D numpy array of ints
        particle_idxs of relatives of all particles in the ID list
    seed_id :
        

    Returns
    -------

    
    """
    try:
        index = np.where(particle_idxs == seed_id)[0][0]
    except IndexError:
        print(f"could not find seed_id {seed_id}")
        return []
    all_indices = [index]
    our_rels = relatives[index]
    for relative in our_rels[our_rels is not None]:
        all_indices += recursive_grab(relative, particle_idxs, relatives)
    return all_indices

