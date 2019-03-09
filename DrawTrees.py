import networkx as nx
from matplotlib import pyplot as plt
from ReadSQL import readSelected
import numpy as np
from ipdb import set_trace as st
from networkx.drawing.nx_agraph import write_dot


class Shower:
    def __init__(self):
        self.IDs = []
        self.parents = []
        self.children = []
        self.labels = []

    @property
    def nParticles(self):
        return len(self.IDs)

    def findRoot(self):
        roots = getRoots(self.IDs, self.parents)
        assert len(roots) == 1, "There should only be one root to a shower"
        self.rootIndex = roots[0]


    def findRanks(self):
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
        assert len(self.IDs) == len(self.parents)
        assert len(self.IDs) == len(self.children)
        assert len(self.IDs) == len(self.labels)
        return DotGraph(self)


def getRoots(IDs, parents):
    roots = []
    listIDs = list(IDs)  # somtimes this is an array
    for i, pars in enumerate(parents):
        if not (pars[0] in IDs or pars[1] in IDs):
            roots.append(i)
    return roots


# ignore not working
def makeTree(IDs, parents, children, labels):
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
        self.__edges += f"\t{ID1} -> {ID2}\n"

    def addLabel(self, ID, label):
        self.__nodes += f'\t{ID} [label="{label}"]\n'

    def addRank(self, IDs):
        id_string = '; '.join(IDs) + ';'
        self.__ranks += f'\t{{rank = same; {id_string}}}'

    def __str__(self):
        fullString = self.__start + self.__nodes + self.__edges + self.__ranks + self.__end
        return fullString

def getShowers(databaseName, exclude_MCPIDs=[2212, 25, 35]):
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
    shower_allocation = np.arange(len(IDs))
    connected = list(np.full_like(IDs, False))
    ID_list = list(IDs)
    # we go through the list of particles making them share allocations if they are decendants
    while False in connected:
        next_unconnected = connected.index(False)
        connected[next_unconnected] = True
        rels = all_relatives[next_unconnected]
        relative_indices = [ID_list.index(rel) for rel
                            in rels if rel in ID_list]
        for i in relative_indices:
            mask = shower_allocation == shower_allocation[i]
            shower_allocation[mask] = shower_allocation[next_unconnected]
    shower_IDs = list(set(shower_allocation))
    print(f"Found {len(shower_IDs)} showers")
    showers = []
    for s_ID in shower_IDs:
        mask = np.where(shower_allocation == s_ID)
        new_shower = Shower()
        new_shower.IDs = IDs[mask]
        new_shower.children = children[mask]
        new_shower.parents = parents[mask]
        new_shower.labels = labels[mask]
        showers.append(new_shower)
    return showers



# don't use - hits recursion limit
def recursiveGrab(seedID, IDs, relatives):
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
    databaseName = "/home/henry/lazy/tag_1_delphes_events.db"
    showers = getShowers(databaseName)
    particles_in_shower = [s.nParticles for s in showers]
    graph = showers[particles_in_shower.index(41)].graph()
    dotName = databaseName.split('.')[0] + ".dot"
    with open(dotName, 'w') as dotFile:
        dotFile.write(str(graph))

if __name__ == '__main__':
    main()
