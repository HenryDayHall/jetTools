import networkx as nx
from matplotlib import pyplot as plt
from ReadSQL import readSelected, ParticleDatabase
from ReadHepmc import Hepmc_event
import numpy as np
from ipdb import set_trace as st
from networkx.drawing.nx_agraph import write_dot
from itertools import compress
from PDGNames import IDConverter


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
        self.__legend = ""
        # if a shower is given make edges from that
        if shower is not None:
            self.fromShower(shower)

    def fromShower(self, shower, jet=None):
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
        roots = shower.roots
        ends = shower.ends
        # set up a legened
        ID = 0
        internal_particle = "darkolivegreen1"
        self.addLegendNode(ID, "Internal particle", colour=internal_particle)
        ID += 1
        root_particle = "darkolivegreen3"
        self.addLegendNode(ID, "Root particle", colour=root_particle)
        ID += 1
        outside_particle = "gold"
        self.addLegendNode(ID, "Connected to other shower", colour=outside_particle)
        ID += 1
        end_shape = "diamond"
        tower_particle = "cadetblue"
        self.addLegendNode(ID, "Particle in tower Tw#", colour=tower_particle, shape=end_shape)
        ID += 1
        track_particle = "deepskyblue1"
        self.addLegendNode(ID, "Particle creates track", colour=track_particle, shape=end_shape)
        ID += 1
        for i, (this_id, label) in enumerate(zip(shower.IDs, shower.labels)):
            colour = internal_particle
            if i in roots:
                colour=root_particle
            elif i in outsiders:
                colour=outside_particle
            shape = None
            if i in ends:
                shape = end_shape
            if shower.makesTower[i] > 0:
                label += f" Tw{shower.makesTower[i]}"
                colour = tower_particle
            if shower.makesTrack[i] == 1:
                colour = track_particle

            self.addNode(this_id, label, colour=colour, shape=shape)
        # add the ranks
        ranks = shower.findRanks()
        rankKeys = sorted(list(set(ranks)))
        for key in rankKeys:
            mask = ranks == key
            rankIDs = np.array(shower.IDs)[mask]
            self.addRank(rankIDs)
        if jet is not None:
            # add the highest shower rank t all the jet ranks
            jet_cluster = "coral"
            self.addLegendNode(ID, f"Jet cluster", colour=jet_cluster)
            jet.addObsLevel()
            jet.addGlobalID(max(shower.IDs))
            adjusted_jet_rank = jet.ranks + max(ranks)
            for index, (this_id, this_mother) in enumerate(zip(jet.IDs, jet.mothers)):
                # add the edges
                if this_mother >= 0:
                    self.addEdge(this_id, this_mother)
                # if needed add the node
                if jet.IDs[index] in shower.IDs:
                    continue
                else:
                    self.addNode(this_id, f"jet", colour=jet_cluster, shape=None)
            # add the ranks
            rankKeys = sorted(list(set(adjusted_jet_rank)))
            for key in rankKeys:
                if key == max(ranks):
                    continue  # skip this as it's in the shower
                mask = adjusted_jet_rank == key
                rankIDs = jet.IDs[mask]
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
        id_string = ' '.join(ID_strings)
        self.__ranks += f'\t{{rank = same; {id_string}}}\n'

    def addLegendNode(self, ID, label, colour=None, shape=None):
        """ Add a label to this graph

        Parameters
        ----------
        ID : int
            node ID to get this label
            
        label : string
            label for node

        """
        self.__legend += f'\t{ID} [label="{label}"'
        if colour is not None:
            self.__legend += f' style=filled fillcolor={colour}'
        if shape is not None:
            self.__legend += f' shape={shape}'
        self.__legend += ']\n'

    def __str__(self):
        fullString = self.__start + self.__nodes + self.__edges + self.__ranks + self.__end
        return fullString

    @property
    def legend(self):
        return self.__start + self.__legend + self.__end


def quickPlot():
    import JetCluster
    diag_hits, showers, matched_jets = JetCluster.make_showerClusterGrid();
    shower1 = showers[-1]; shower2 = showers[-2]; jet = matched_jets[-1];
    #shower1.amalgamate(shower2)
    graph = DotGraph()
    graph.fromShower(shower1, jet)
    graph.fromShower(shower1)
    with open("singleShowerJet.dot", 'w') as f:
        f.write(str(graph))

    
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
            pass #continue
        max_daughters = max([len(d) for d in shower.daughters])
        addTracksTowers(databaseName, shower)
        print(f"Drawing shower {i}, has {max_daughters} max daughters. Daughters to particles ratio = {max_daughters/len(shower.daughters)}")
        graph = shower.graph()
        dotName = hepmc_name.split('.')[0] + str(i) + ".dot"
        legendName = hepmc_name.split('.')[0] + str(i) + "_ledg.dot"
        with open(dotName, 'w') as dotFile:
            dotFile.write(str(graph))
        with open(legendName, 'w') as dotFile:
            dotFile.write(graph.legend)

if __name__ == '__main__':
    pass #main()
