import networkx as nx
from matplotlib import pyplot as plt
from ReadSQL import readSelected
import numpy as np
from ipdb import set_trace as st
from networkx.drawing.nx_agraph import write_dot


class Shower:
    """ Object to hold a shower of particles
    
    only keeps a list of the particle IDs, parents, children and PDGIDs.
    """
    def __init__(self):
        self.IDs = []
        self.parents = []
        self.children = []
        self.labels = []

    @property
    def nParticles(self):
        """ int: the number of particles at all points of the shower """
        return len(self.IDs)

    def findRoot(self):
        """ Demand the shower identify it's root. 
        This is stored as an internal variable. """
        roots = getRoots(self.IDs, self.parents)
        assert len(roots) == 1, "There should only be one root to a shower"
        self.rootIndex = roots[0]


    def findRanks(self):
        """ Demand the shower identify the rang of each particle.
        The rank of a particle is the length of the shortest distance to the root.

        Returns
        -------
        ranks : numpy array of ints
            mimics the structure of the ID list
        """
        self.findRoot()
        # put the first rank in
        currentRank = [self.rootIndex]
        rankN = 0
        ranks = np.full_like(self.IDs, -1, dtype=int)
        ranks[currentRank] = rankN
        listIDs = list(self.IDs)  # somtimes this is an array
        hasDecendants = True
        st()
        while hasDecendants:
            rankN += 1
            decendantIDs = [child for index in currentRank for child in self.children[index]
                            if child in self.IDs]
            currentRank = []
            for child in decendantIDs:
                index = listIDs.index(child)
                # don't overwite a rank, so in a loop the lowers rank stands
                # also needed to prevent perpetual loops
                if ranks[index] == -1:
                    currentRank.append(index)
            ranks[currentRank] = rankN
            hasDecendants = len(currentRank) > 0
        assert -1 not in ranks
        self.ranks = ranks
        return ranks

    def graph(self):
        """ Turn the shower into a dotgraph
        
        Returns
        -------
        dotgraph : DotGraph
            an object that can produce a string that would be in a dot file for this graph
        """
        assert len(self.IDs) == len(self.parents)
        assert len(self.IDs) == len(self.children)
        assert len(self.IDs) == len(self.labels)
        return DotGraph(self)

    @property
    def outsideConnections(self):
        """ Not implemented
        Function that anouches which particles have perantage from outside this shower
        Includes the root """
        raise NotImplementedError


def getRoots(IDs, parents):
    """From a list of particle IDs and a list of parent IDs determin root particles

    A root particle is one whos parents are both from outside the particle list.

    Parameters
    ----------
    IDs : numpy array of ints
        The unique id of each particle
        
    parents : 2D numpy array of ints
        Each row contains the IDs of two mothers of each particle in IDs
        These can be none

    Returns
    -------
    roots : numpy array of ints
        the IDs of the root particles

    """
    roots = []
    listIDs = list(IDs)  # somtimes this is an array
    for i, pars in enumerate(parents):
        if not (pars[0] in IDs or pars[1] in IDs):
            roots.append(i)
    return roots


# ignore not working
def makeTree(IDs, parents, children, labels):
    """
    It's possible this is working better than I think it is ...
    Just the data was screwy
    Anyway it has been supassed by dot files.
    

    Parameters
    ----------
    IDs : numpy array of ints
        The unique id of each particle
        
    parents : 2D numpy array of ints
        Each row contains the IDs of two mothers of each particle in IDs
        These can be none
        
    children : 2D numpy array of ints
        Each row contains the IDs of two daughters of each particle in IDs
        These can be none
        
    labels : numpy array of ints
        the MC PDG ID of each particle in IDs
        

    Returns
    -------
    graph : networkx Graph
        The tree as a graph

    """
    graph =  nx.Graph()
    graph.add_nodes_from(IDs)
    for this_id, this_parents, this_children in zip(IDs, parents, children):
        parent_edges = [(par_id, this_id) for par_id in this_parents if par_id in IDs]
        graph.add_edges_from(parent_edges)
        child_edges = [(this_id, chi_id) for chi_id in this_children if chi_id in IDs]
        graph.add_edges_from(child_edges)
    label_dict = {i:l for i, l in zip(IDs, labels)}
    nx.relabel_nodes(graph, label_dict)
    return graph


class DotGraph:
    """ A class that allows a dot graph to be built and can represent it as a string.

    Parameters
    ----------
    shower : Shower
        an object that holds all infomration about the structure of a particle shower
        optional
    graph : string
        is it a graph or a digraph (directed graph)
        default is digraph
    strict : bool
        does the graph forbid identical connections?
        default is True
    name : string
        name the graph
        optional
    """
    def __init__(self, shower=None, **kwargs):
        # start by checking the kwargs
        graph = kwargs.get('graph', 'digraph')
        strict = kwargs.get('strict', True)
        graphName = kwargs.get('name', None)
        # set up
        self.__start = ""
        if strict:
            self.__start += "strict "
        if graphName is not None:
            self.__start += graphName + " "
        self.__start += " {\n"
        self.__end = "}\n"
        self.__nodes = ""
        self.__edges = ""
        self.__ranks = ""
        # if a shower is given make edges from that
        if shower is not None:
            self.fromShower(shower)

    def fromShower(self, shower):
        """ Construct the graph from a shower object
        

        Parameters
        ----------
        shower : Shower
            an object that holds all infomration about the structure of a particle shower
            
        """
        # add the edges
        for this_id, this_parents in zip(shower.IDs, shower.parents):
            for parent in this_parents:
                if parent in shower.IDs:
                    self.addEdge(parent, this_id)
        # add the labels
        for this_id, label in zip(shower.IDs, shower.labels):
            self.addLabel(this_id, label)
        # add the ranks
        ranks = shower.findRanks()
        rankKeys = set(ranks)
        for key in rankKeys:
            mask = ranks == key
            rankIDs = np.array(shower.IDs)[mask]
            self.addRank(rankIDs)

    def addEdge(self, ID1, ID2):
        """ Add an edge to this graph
        

        Parameters
        ----------
        ID1 : int
            start node ID
            
        ID2 : int
            end node ID

        """
        self.__edges += f"\t{ID1} -> {ID2}\n"

    def addLabel(self, ID, label):
        """ Add a label to this graph
        

        Parameters
        ----------
        ID : int
            node ID to get this label
            
        label : string
            label for node

        """
        self.__nodes += f'\t{ID} [label="{label}"]\n'

    def addRank(self, IDs):
        """ Specify set of IDs that sit on the same rank
            

        Parameters
        ----------
        IDs : list like of ints
            IDs on a rank

        """
        id_string = '; '.join(IDs) + ';'
        self.__ranks += f'\t{{rank = same; {id_string}}}'

    def __str__(self):
        fullString = self.__start + self.__nodes + self.__edges + self.__ranks + self.__end
        return fullString

def getShowers(databaseName, exclude_MCPIDs=[2212, 25, 35]):
    """ From each root split the decendants into showers
    Each particle can only belong to a single shower. 

    Parameters
    ----------
    databaseName : string
        Path and file name of database
        
    exclude_MCPIDs : list like of ints
        PIDs that will be cut out before splitting into showers
         (Default value = [2212, 25, 35])
        

    Returns
    -------
    showers : list of Showers
        showers found in the database

    """
    fields = ["ID", "M1", "M2", "D1", "D2", "MCPID"]
    fromDatabase = readSelected(databaseName, fields)
    IDs = np.array([d[0] for d in fromDatabase])
    parents = np.array([d[1:3] for d in fromDatabase])
    children = np.array([d[3:5] for d in fromDatabase])
    labels = np.array([d[5] for d in fromDatabase])
    # remove any stop IDs
    for exclude in exclude_MCPIDs:
        mask = labels != exclude
        IDs = IDs[mask]
        parents = parents[mask]
        children = children[mask]
        labels = labels[mask]
    # check that worked
    remainingPIDs = set(labels)
    for exclude in exclude_MCPIDs:
        assert exclude not in remainingPIDs
    all_relatives = np.hstack((parents, children))
    # now we have values for the whole event,
    # but we want to split the event into showers
    # at start all particles are allocated to a diferent shower
    allocated = np.full_like(IDs, False)
    showers = []
    roots = getRoots(IDs, parents)
    print(f"Found {len(roots)} roots")
    listIDs = list(IDs)
    # not working, showers way too small
    for i, root in enumerate(roots):
        print(f"Working on root {i}.", flush=True)
        showerIndices = [root]
        to_follow = [root]
        n_taken = 0
        while len(to_follow) > 0:
            next_gen = []
            for index in to_follow:
                # be careful to prevent loops
                next_gen += [listIDs.index(child) for child in children[index]
                             if child in IDs]
            # check it's not alread allocated
            n_taken += sum(allocated[next_gen])
            # also prevent loops
            next_gen = [i for i in next_gen if (not allocated[i] and i not in showerIndices)]
            showerIndices += next_gen
            assert len(set(showerIndices)) == len(showerIndices)
            # print(f"Indices in shower = {len(showerIndices)}", flush = True)
            to_follow = next_gen
        allocated[showerIndices] = True
        newShower = Shower()
        newShower.IDs = IDs[showerIndices]
        newShower.parents = parents[showerIndices]
        newShower.children = children[showerIndices]
        newShower.labels = labels[showerIndices]
        showers.append(newShower)
        print(f"Shower missed {n_taken} taken indices")
    return showers



# don't use - hits recursion limit
def recursiveGrab(seedID, IDs, relatives):
    """
    Not a good idea in python

    Parameters
    ----------
    seedID : int
        root particle ID
        
    IDs : numpy array of ints
        IDs of all particles considered
        
    relatives : 2D numpy array of ints
        IDs of relatives of all particles in the ID list
        

    Returns
    -------
    all_indices : list of ints
        list of the indices of all particles found to be in the shower of the seedID

    """
    try:
        index = np.where(IDs==seedID)[0][0]
    except IndexError:
        print(f"could not find seedID {seedID}")
        return []
    all_indices = [index]
    our_rels = relatives[index]
    for relative in our_rels[our_rels!=None]:
        all_indices += recursiveGrab(relative, IDs, relatives)
    return all_indices

     
    
def main():
    """ Launch file, makes and saves a dot graph """
    databaseName = "/home/henry/lazy/tag_1_delphes_events.db"
    showers = getShowers(databaseName)
    particles_in_shower = [s.nParticles for s in showers]
    graph = showers[particles_in_shower.index(41)].graph()
    dotName = databaseName.split('.')[0] + ".dot"
    with open(dotName, 'w') as dotFile:
        dotFile.write(str(graph))

if __name__ == '__main__':
    main()
