""" Tools to turn clusters of particles into showers """
from ipdb import set_trace as st
import networkx
import PDGNames
import ReadSQL
import ReadHepmc
import DrawTrees
import itertools
from matplotlib import pyplot as plt
import numpy as np

class Shower:
    """ Object to hold a shower of particles
    
    only keeps a list of the particle global_ids, mothers, daughters and PDGglobal_ids.
    """
    def __init__(self, global_ids, mothers, daughters, labels):
        self.amalgam = False
        self.global_ids = global_ids
        self.mothers = mothers
        self.daughters = daughters
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
                assert self.mothers[sIndex] == other_shower.mothers[oIndex]
                assert self.daughters[sIndex] == other_shower.daughters[oIndex]
                assert self.labels[sIndex] == other_shower.labels[oIndex]
            else:  # add it on
                sIndex = next_free_sIndex
                next_free_sIndex += 1
                self.global_ids[sIndex] = oID
                self.mothers.append(other_shower.mothers[oIndex])
                self.daughters.append(other_shower.daughters[oIndex])
                self.labels[sIndex] = other_shower.labels[oIndex]

    @property
    def n_particles(self):
        """ int: the number of particles at all points of the shower """
        return len(self.global_ids)

    def _find_roots(self):
        """ Demand the shower identify it's root.
        This is stored as an internal variable. """
        roots = getRoots(self.global_ids, self.mothers)
        if not self.amalgam:
            assert len(roots) == 1, "There should only be one root to a shower"
        self.roots = roots


    def find_ranks(self):
        """ Demand the shower identify the rang of each particle.
        The rank of a particle is the length of the shortest distance to the root.

        Returns
        -------
        ranks : numpy array of ints
            mimics the structure of the ID list
        """
        # put the first rank in
        current_rank = self.roots
        rank_n = 0
        ranks = np.full_like(self.global_ids, -1, dtype=int)
        ranks[current_rank] = rank_n
        list_global_ids = list(self.global_ids)  # somtimes this is an array
        has_decendants = True
        while has_decendants:
            rank_n += 1
            decendant_global_ids = [daughter for index in current_rank
                                    for daughter in self.daughters[index]
                                    if daughter in self.global_ids]
            current_rank = []
            for daughter in decendant_global_ids:
                index = list_global_ids.index(daughter)
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
        assert len(self.global_ids) == len(self.mothers)
        assert len(self.global_ids) == len(self.daughters)
        assert len(self.global_ids) == len(self.labels)
        return DrawTrees.DotGraph(self)

    @property
    def outside_connections(self):
        """
        Function that anouches which particles have perantage from outside this shower
        Includes the root """
        outside_indices = []
        for i, mothers_here in enumerate(self.mothers):
            if not np.all([m in self.global_ids for m in mothers_here]):
                outside_indices.append(i)
        return outside_indices

    @property
    def ends(self):
        _ends = []
        for i, daughters_here in enumerate(self.daughters):
            if np.all([daughter is None for daughter in daughters_here]):
                _ends.append(i)
        return _ends

    @property
    def flavour(self):
        flavours = self.labels[self.roots]
        return '+'.join(flavours)
        

def get_showers(particle_collection, exclude_pids=[2212, 25, 35]):
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
    mask = [p not in exclude_pids for p in particle_collection.pids]
    global_ids = particle_collection.global_ids[mask]
    mother_ids = [p.mother_ids for p in particle_collection.particle_list]
    mother_ids = list(itertools.compress(mother_ids, mask))
    daughter_ids = [p.daughter_ids for p in particle_collection.particle_list]
    daughter_ids = list(itertools.compress(daughter_ids, mask))
    pids = particle_collection.pids[mask]
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
    roots = getRoots(global_ids, mother_ids)
    print(f"Found {len(roots)} roots")
    list_global_ids = list(global_ids)
    for i, root in enumerate(roots):
        print(f"Working on root {i}.", flush=True)
        shower_indices = [root]
        to_follow = [root]
        while len(to_follow) > 0:
            next_gen = []
            for index in to_follow:
                # be careful not to include forbiden particles
                next_gen += [list_global_ids.index(daughter) for daughter in daughter_ids[index]
                             if daughter in global_ids]
            # prevent loops
            next_gen = list(set(next_gen))
            next_gen = [i for i in next_gen if i not in shower_indices]
            shower_indices += next_gen
            # print(f"Indices in shower = {len(shower_indices)}", flush = True)
            to_follow = next_gen
        assert len(set(shower_indices)) == len(shower_indices)
        new_shower = Shower(global_ids[shower_indices],
                            [mother_ids[s] for s in shower_indices], 
                            [daughter_ids[s] for s in shower_indices],
                            labels[shower_indices])
        showers.append(new_shower)
    return showers


def getRoots(global_ids, mothers):
    """From a list of particle global_ids and a list of mother global_ids determin root particles

    A root particle is one whos mothers are both from outside the particle list.

    Parameters
    ----------
    global_ids : numpy array of ints
        The unique id of each particle
        
    mothers : 2D numpy array of ints
        Each row contains the global_ids of two mothers of each particle in global_ids
        These can be none

    Returns
    -------
    roots : numpy array of ints
        the global_ids of the root particles

    """
    roots = []
    for i, mothers_here in enumerate(mothers):
        if not np.any([m in global_ids for m in mothers_here]):
            roots.append(i)
    return roots


# ignore not working
def makeTree(global_ids, mothers, daughters, labels):
    """
    It's possible this is working better than I think it is ...
    Just the data was screwy
    Anyway it has been supassed by dot files.
    

    Parameters
    ----------
    global_ids : numpy array of ints
        The unique id of each particle
        
    mothers : 2D numpy array of ints
        Each row contains the global_ids of two mothers of each particle in global_ids
        These can be none
        
    daughters : 2D numpy array of ints
        Each row contains the global_ids of two daughters of each particle in global_ids
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
    for this_id, this_mothers, this_daughters in zip(global_ids, mothers, daughters):
        mother_edges = [(par_id, this_id) for par_id in this_mothers if par_id in global_ids]
        graph.add_edges_from(mother_edges)
        daughter_edges = [(this_id, chi_id) for chi_id in this_daughters if chi_id in global_ids]
        graph.add_edges_from(daughter_edges)
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

