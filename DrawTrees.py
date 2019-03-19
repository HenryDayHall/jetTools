import networkx as nx
from matplotlib import pyplot as plt
from ReadSQL import readSelected, ParticleDatabase
from ReadHepmc import Hepmc_event
import numpy as np
from ipdb import set_trace as st
from networkx.drawing.nx_agraph import write_dot
from itertools import compress
from PDGNames import IDConverter


class Shower:
    """ Object to hold a shower of particles
    
    only keeps a list of the particle IDs, mothers, daughters and PDGIDs.
    """
    def __init__(self):
        self.IDs = []
        self.mothers = []
        self.daughters = []
        self.labels = []
        self.__makesTrack = np.array([])
        self.__makesTower = np.array([])

    @property
    def nParticles(self):
        """ int: the number of particles at all points of the shower """
        return len(self.IDs)

    def findRoot(self):
        """ Demand the shower identify it's root. 
        This is stored as an internal variable. """
        roots = getRoots(self.IDs, self.mothers)
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
        while hasDecendants:
            rankN += 1
            decendantIDs = [daughter for index in currentRank for daughter in self.daughters[index]
                            if daughter in self.IDs]
            currentRank = []
            for daughter in decendantIDs:
                index = listIDs.index(daughter)
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
        assert len(self.IDs) == len(self.mothers)
        assert len(self.IDs) == len(self.daughters)
        assert len(self.IDs) == len(self.labels)
        return DotGraph(self)

    @property
    def outsideConnections(self):
        """ 
        Function that anouches which particles have perantage from outside this shower
        Includes the root """
        outside_indices = []
        for i, mothers_here in enumerate(self.mothers):
            if not np.all([m in self.IDs for m in mothers_here]):
                outside_indices.append(i)
        return outside_indices

    @property
    def ends(self):
        _ends = []
        for i, daughters_here in enumerate(self.daughters):
            if np.all([d == None for d in daughters_here]):
                _ends.append(i)
        return _ends

    @property
    def makesTrack(self):
        if len(self.__makesTrack) > len(self.IDs):
            raise RuntimeError("How are there more track declarations than tracks?")
        elif len(self.__makesTrack) < len(self.IDs):
            to_add = len(self.IDs) - len(self.__makesTrack)
            self.__makesTrack = np.hstack((self.__makesTrack, np.zeros(to_add)))
        return self.__makesTrack

    @property
    def makesTower(self):
        if len(self.__makesTower) > len(self.IDs):
            raise RuntimeError("How are there more tower declarations than tracks?")
        elif len(self.__makesTower) < len(self.IDs):
            to_add = len(self.IDs) - len(self.__makesTower)
            self.__makesTower = np.hstack((self.__makesTower, np.zeros(to_add)))
        return self.__makesTower
        
def addTracksTowers(databaseName, shower):
    trackParticles = readSelected(databaseName, ["Particle"], tableName="Tracks")
    for p in trackParticles:
        index = np.where(shower.IDs==p)[0]
        if len(index) == 1:
            shower.makesTrack[index[0]] = 1
    return shower

def getRoots(IDs, mothers):
    """From a list of particle IDs and a list of mother IDs determin root particles

    A root particle is one whos mothers are both from outside the particle list.

    Parameters
    ----------
    IDs : numpy array of ints
        The unique id of each particle
        
    mothers : 2D numpy array of ints
        Each row contains the IDs of two mothers of each particle in IDs
        These can be none

    Returns
    -------
    roots : numpy array of ints
        the IDs of the root particles

    """
    roots = []
    for i, mothers_here in enumerate(mothers):
        if not np.any([m in IDs for m in mothers_here]):
            roots.append(i)
    return roots


# ignore not working
def makeTree(IDs, mothers, daughters, labels):
    """
    It's possible this is working better than I think it is ...
    Just the data was screwy
    Anyway it has been supassed by dot files.
    

    Parameters
    ----------
    IDs : numpy array of ints
        The unique id of each particle
        
    mothers : 2D numpy array of ints
        Each row contains the IDs of two mothers of each particle in IDs
        These can be none
        
    daughters : 2D numpy array of ints
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
    for this_id, this_mothers, this_daughters in zip(IDs, mothers, daughters):
        mother_edges = [(par_id, this_id) for par_id in this_mothers if par_id in IDs]
        graph.add_edges_from(mother_edges)
        daughter_edges = [(this_id, chi_id) for chi_id in this_daughters if chi_id in IDs]
        graph.add_edges_from(daughter_edges)
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
        self.__start += graph + " {\n"
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
        for this_id, this_mothers in zip(shower.IDs, shower.mothers):
            for mother in this_mothers:
                if mother in shower.IDs:
                    self.addEdge(mother, this_id)
        # add the labels
        outsiders = shower.outsideConnections
        shower.findRoot()
        root = shower.rootIndex
        ends = shower.ends
        for i, (this_id, label) in enumerate(zip(shower.IDs, shower.labels)):
            colour = "darkolivegreen1"
            if i == root:
                colour="darkolivegreen3"
            elif i in outsiders:
                colour="gold"
            shape = None
            if i in ends:
                shape = "diamond"
            if shower.makesTower[i] > 0:
                label += f" Tw{shower.makesTower[i]}"
                colour = "cadetblue"
            if shower.makesTrack[i] == 1:
                colour = "deepskyblue1"

            self.addNode(this_id, label, colour=colour, shape=shape)
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

    def addNode(self, ID, label, colour=None, shape=None):
        """ Add a label to this graph
        

        Parameters
        ----------
        ID : int
            node ID to get this label
            
        label : string
            label for node

        """
        self.__nodes += f'\t{ID} [label="{label}"'
        if colour is not None:
            self.__nodes += f' style=filled fillcolor={colour}'
        if shape is not None:
            self.__nodes += f' shape={shape}'
        self.__nodes += ']\n'

    def addRank(self, IDs):
        """ Specify set of IDs that sit on the same rank
            

        Parameters
        ----------
        IDs : list like of ints
            IDs on a rank

        """
        ID_strings = [str(ID) for ID in IDs]
        id_string = '; '.join(ID_strings) + ';'
        self.__ranks += f'\t{{rank = same; {id_string}}}\n'

    def __str__(self):
        fullString = self.__start + self.__nodes + self.__edges + self.__ranks + self.__end
        return fullString

def getShowers(particleSource, exclude_MCPIDs=[2212, 25, 35]):
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
    # remove any stop IDs
    mask = [p not in exclude_MCPIDs for p in particleSource.MCPIDs]
    IDs = np.array(particleSource.IDs)[mask]
    mothers = list(compress(particleSource.mothers, mask))
    daughters = list(compress(particleSource.daughters, mask))
    labels = np.array(particleSource.MCPIDs)[mask]
    # check that worked
    remainingPIDs = set(labels)
    for exclude in exclude_MCPIDs:
        assert exclude not in remainingPIDs
    # now where possible convert labels to names
    pdg = IDConverter()
    labels = np.array([pdg[x] for x in labels])
    # now we have values for the whole event,
    # but we want to split the event into showers
    # at start all particles are allocated to a diferent shower
    showers = []
    roots = getRoots(IDs, mothers)
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
                # be careful not to include forbiden particles
                next_gen += [listIDs.index(daughter) for daughter in daughters[index]
                             if daughter in IDs]
            # prevent loops
            next_gen = list(set(next_gen))
            next_gen = [i for i in next_gen if i not in showerIndices]
            showerIndices += next_gen
            assert len(set(showerIndices)) == len(showerIndices)
            # print(f"Indices in shower = {len(showerIndices)}", flush = True)
            to_follow = next_gen
        newShower = Shower()
        newShower.IDs = IDs[showerIndices]
        newShower.mothers = [mothers[i] for i in showerIndices]
        newShower.daughters = [daughters[i] for i in showerIndices]
        newShower.labels = labels[showerIndices]
        showers.append(newShower)
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
    databaseName = "/home/henry/lazy/29delphes_events.db"
    hepmc_name = "/home/henry/lazy/29pythia8_events.hepmc"
    hepmc = Hepmc_event()
    hepmc.read_file(hepmc_name)
    hepmc.assign_heritage()
    showers = getShowers(hepmc)
    # rootDB = ParticleDatabase(databaseName)
    # showers = getShowers(rootDB)
    for i, shower in enumerate(showers):
        if 'b' not in shower.labels:
            continue
        addTracksTowers(databaseName, shower)
        print(f"Drawing shower {i}")
        graph = shower.graph()
        dotName = hepmc_name.split('.')[0] + str(i) + ".dot"
        with open(dotName, 'w') as dotFile:
            dotFile.write(str(graph))

if __name__ == '__main__':
    main()
