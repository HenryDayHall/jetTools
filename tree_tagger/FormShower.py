""" Tools to turn clusters of particles into showers """
from ipdb import set_trace as st
import networkx
from tree_tagger import PDGNames, ReadSQL, ReadHepmc, DrawTrees
import itertools
from matplotlib import pyplot as plt
import numpy as np

class Shower:
    """ Object to hold a shower of particles
    
    only keeps a list of the particle global_ids, parents, childs and PDGglobal_ids.
    """
    def __init__(self, global_ids, parents, childs, labels):
        self.amalgam = False
        self.global_ids = global_ids
        self.parents = parents
        self.childs = childs
        self.labels = labels
        self.ranks = None  # exspensive, create as needed with find_ranks()
        self._find_roots()

    def amalgamate(self, other_shower):
        self.amalgam = True
        total_global_ids = len(set(self.global_ids).union(set(other_shower.global_ids)))
        next_free_sIndex = len(self.global_ids)
        self.global_ids.resize(total_global_ids)
        self.labels.resize(total_global_ids)
        for oIndex, oID in enumerate(other_shower.global_ids):
            if oID in self.global_ids:  #check for agreement
                sIndex = list(self.global_ids).index(oID)
                assert self.parents[sIndex] == other_shower.parents[oIndex]
                assert self.childs[sIndex] == other_shower.childs[oIndex]
                assert self.labels[sIndex] == other_shower.labels[oIndex]
            else:  # add it on
                sIndex = next_free_sIndex
                next_free_sIndex += 1
                self.global_ids[sIndex] = oID
                self.parents.append(other_shower.parents[oIndex])
                self.childs.append(other_shower.childs[oIndex])
                self.labels[sIndex] = other_shower.labels[oIndex]
        self._find_roots()

    @property
    def n_particles(self):
        """ int: the number of particles at all points of the shower """
        return len(self.global_ids)

    def _find_roots(self):
        """ Demand the shower identify it's root.
        This is stored as an internal variable. """
        root_gids = get_roots(self.global_ids, self.parents)
        if not self.amalgam:
            assert len(root_gids) == 1, "There should only be one root to a shower"
        self.root_gids = root_gids
        list_gids = list(self.global_ids)
        self.root_idxs = [list_gids.index(r) for r in root_gids]

    @property
    def roots(self):
        msg = "changed to root_gids or root_idxs for clarity"
        raise AttributeError(msg)


    def find_ranks(self):
        """ Demand the shower identify the rang of each particle.
        The rank of a particle is the length of the shortest distance to the root.

        Returns
        -------
        ranks : numpy array of ints
            mimics the structure of the ID list
        """
        # put the first rank in
        current_rank = self.root_idxs
        rank_n = 0
        ranks = np.full_like(self.global_ids, -1, dtype=int)
        ranks[current_rank] = rank_n
        list_global_ids = list(self.global_ids)  # somtimes this is an array
        has_decendants = True
        while has_decendants:
            rank_n += 1
            decendant_global_ids = [child for index in current_rank
                                    for child in self.childs[index]
                                    if child in self.global_ids]
            current_rank = []
            for child in decendant_global_ids:
                index = list_global_ids.index(child)
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
        """ Turn the shower into a dotgraph
        
        Returns
        -------
        dotgraph : DotGraph
            an object that can produce a string that would be in a dot file for this graph
        """
        assert len(self.global_ids) == len(self.parents)
        assert len(self.global_ids) == len(self.childs)
        assert len(self.global_ids) == len(self.labels)
        return DrawTrees.DotGraph(self)

    @property
    def outside_connections(self):
        raise AttributeError("you want outside_connection_gids, but check you are useing global_id not index")

    @property
    def outside_connection_gids(self):
        """
        Function that anouches which particles have perantage from outside this shower
        Includes the root """
        outside_gids = []
        for gid, parents_here in zip(self.global_ids, self.parents):
            if not np.all([m in self.global_ids for m in parents_here]):
                outside_gids.append(gid)
        return outside_gids

    @property
    def ends(self):
        _ends = []
        for i, childs_here in enumerate(self.childs):
            if np.all([child is None for child in childs_here]):
                _ends.append(i)
        return _ends

    @property
    def flavour(self):
        flavours = self.labels[self.roots]
        return '+'.join(flavours)
        

def get_showers(eventWise, event_n, exclude_pids=[2212, 25, 35]):
    """ From each root split the decendants into showers
    Each particle can only belong to a single shower.

    Parameters
    ----------
    databaseName : string
        Path and file name of database

    exclude_MCPglobal_ids : list like of ints
        Pglobal_ids that will be cut out before splitting into showers
         (Default value = [2212, 25, 35])


    Returns
    -------
    showers : list of Showers
        showers found in the database

    """
    # remove any stop pids
    mask = [p not in exclude_pids for p in eventWise.PID[event_n]]
    global_ids = np.where(mask)[0]
    parent_ids = eventWise.Parents[event_n, mask]
    child_ids = eventWise.Children[event_n, mask]
    pids = eventWise.PID[event_n, mask]
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
    root_gids = get_roots(global_ids, parent_ids)
    print(f"Found {len(root_gids)} root_gids")
    list_global_ids = list(global_ids)
    for i, root_gid in enumerate(root_gids):
        root_idx = list_global_ids.index(root_gid)
        print(f"Working on root {i}.", flush=True)
        shower_indices = [root_idx]
        to_follow = [root_idx]
        while len(to_follow) > 0:
            next_gen = []
            for index in to_follow:
                # be careful not to include forbiden particles
                next_gen += [list_global_ids.index(child) for child in child_ids[index]
                             if child in global_ids]
            # prevent loops
            next_gen = list(set(next_gen))
            next_gen = [i for i in next_gen if i not in shower_indices]
            shower_indices += next_gen
            # print(f"Indices in shower = {len(shower_indices)}", flush = True)
            to_follow = next_gen
        assert len(set(shower_indices)) == len(shower_indices)
        new_shower = Shower(global_ids[shower_indices],
                            [parent_ids[s] for s in shower_indices], 
                            [child_ids[s] for s in shower_indices],
                            labels[shower_indices])
        showers.append(new_shower)
    return showers


def get_roots(global_ids, parents):
    """From a list of particle global_ids and a list of parent global_ids determin root particles

    A root particle is one whos parents are both from outside the particle list.

    Parameters
    ----------
    global_ids : numpy array of ints
        The unique id of each particle
        
    parents : 2D numpy array of ints
        Each row contains the global_ids of two parents of each particle in global_ids
        These can be none

    Returns
    -------
    roots : numpy array of ints
        the global_ids of the root particles

    """
    roots = []
    for gid, parents_here in zip(global_ids, parents):
        if not np.any([m in global_ids for m in parents_here]):
            roots.append(gid)
    return roots


# ignore not working
def make_tree(global_ids, parents, childs, labels):
    """
    It's possible this is working better than I think it is ...
    Just the data was screwy
    Anyway it has been supassed by dot files.
    

    Parameters
    ----------
    global_ids : numpy array of ints
        The unique id of each particle
        
    parents : 2D numpy array of ints
        Each row contains the global_ids of two parents of each particle in global_ids
        These can be none
        
    childs : 2D numpy array of ints
        Each row contains the global_ids of two childs of each particle in global_ids
        These can be none
        
    labels : numpy array of ints
        the MC PDG ID of each particle in global_ids
        

    Returns
    -------
    graph : networkx Graph
        The tree as a graph

    """
    graph =  networkx.Graph()
    graph.add_nodes_from(global_ids)
    for this_id, this_parents, this_childs in zip(global_ids, parents, childs):
        parent_edges = [(par_id, this_id) for par_id in this_parents if par_id in global_ids]
        graph.add_edges_from(parent_edges)
        child_edges = [(this_id, chi_id) for chi_id in this_childs if chi_id in global_ids]
        graph.add_edges_from(child_edges)
    label_dict = {i:l for i, l in zip(global_ids, labels)}
    networkx.relabel_nodes(graph, label_dict)
    return graph


# don't use - hits recursion limit
def recursive_grab(seed_id, global_ids, relatives):
    """
    Not a good idea in python

    Parameters
    ----------
    seedID : int
        root particle ID

    global_ids : numpy array of ints
        global_ids of all particles considered

    relatives : 2D numpy array of ints
        global_ids of relatives of all particles in the ID list


    Returns
    -------
    all_indices : list of ints
        list of the indices of all particles found to be in the shower of the seedID

    """
    try:
        index = np.where(global_ids == seed_id)[0][0]
    except IndexError:
        print(f"could not find seed_id {seed_id}")
        return []
    all_indices = [index]
    our_rels = relatives[index]
    for relative in our_rels[our_rels is not None]:
        all_indices += recursive_grab(relative, global_ids, relatives)
    return all_indices

