from tree_tagger import Components, InputTools
import numpy as np
from ipdb import set_trace as st
from networkx.drawing.nx_agraph import write_dot
from itertools import compress
from tree_tagger.PDGNames import IDConverter


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
            eventWise = kwargs.get('eventWise', None)
            assert isinstance(eventWise.selected_index, int), "Must specify one event in the eventWise object"
            use_TracksTowers = kwargs.get('use_TracksTowers', False)
            jet_name = kwargs.get('jet_name', None)
            jet_num = kwargs.get('jet_num', None)
            self.fromShower(shower, eventWise, jet_name, jet_num, use_TracksTowers)

    def fromShower(self, shower, eventWise=None, jet_name=None, jet_num=None, use_TracksTowers=False):
        """ Construct the graph from a shower object
        

        Parameters
        ----------
        shower : Shower
            an object that holds all infomration about the structure of a particle shower
            
        """
        # add the edges
        for this_id, this_parents in zip(shower.particle_idxs, shower.parents):
            for parent in this_parents:
                if parent in shower.particle_idxs:
                    self.addEdge(parent, this_id)
        # add the labels
        shower_leaf_particle_idxs = shower.particle_idxs[shower.ends]
        # set up a legened
        legened_id = 0
        internal_particle = "darkolivegreen1"
        self.addLegendNode(legened_id, "Internal particle", colour=internal_particle)
        legened_id += 1
        root_particle = "darkolivegreen3"
        self.addLegendNode(legened_id, "Root particle", colour=root_particle)
        legened_id += 1
        outside_particle = "gold"
        self.addLegendNode(legened_id, "Connected to other shower", colour=outside_particle)
        legened_id += 1
        # add the shower particles in
        for this_id, label in zip(shower.particle_idxs, shower.labels):
            colour = internal_particle
            if this_id in shower.root_idxs:
                colour=root_particle
            elif this_id in shower.outside_connection_idxs:
                colour=outside_particle
            shape = None
            self.addNode(this_id, label, colour=colour, shape=shape)
        # add the ranks
        ranks = shower.find_ranks()
        rank_keys = sorted(list(set(ranks)))
        for key in rank_keys:
            mask = ranks == key
            rank_ids = np.array(shower.particle_idxs)[mask]
            self.addRank(rank_ids)
        first_obs_id = np.max(shower.particle_idxs) + 1
        obs_rank = max(ranks) + 1
        obs_draw_map = {}
        if eventWise is not None:
            # make an observable layer
            # observable part of legend
            observable_shape = "diamond"
            if use_TracksTowers:
                tower_particle = "cadetblue"
                self.addLegendNode(legened_id, "Tower", colour=tower_particle, shape=observable_shape)
                legened_id += 1
                track_particle = "deepskyblue1"
                self.addLegendNode(legened_id, "Track", colour=track_particle, shape=observable_shape)
                legened_id += 1
            else:
                self.addLegendNode(legened_id, "Observable", colour=internal_particle, shape=observable_shape)
                legened_id += 1
            # start by deciding which observabels are actually used
            jet_tracks, jet_towers, jet_particles = [], [], []
            if jet_name is not None:
                self.addLegendNode(legened_id, "Jet")
                # this woudl need changign if the jets started clustering somethign besides 
                # MC Particles
                # if the pseudojet lacks a child1 (or 2) it is observable
                jet_ends_internalIdx = np.where(getattr(eventWise, jet_name+"_Child1")[jet_num] < 0)[0]
                jet_ends_inputIdx = getattr(eventWise, jet_name+"_InputIdx")[jet_num][jet_ends_internalIdx]
                jet_particles = eventWise.JetInputs_SourceIdx[jet_ends_inputIdx]
                if use_TracksTowers:
                    jet_tracks = [eventWise.Particle_Track[i] for i in jet_particles
                                  if eventWise.Particle_Track[i]!=-1]
                    jet_towers = [eventWise.Particle_Tower[i] for i in jet_particles
                                  if eventWise.Particle_Tower[i]!=-1]
            if use_TracksTowers:
                draw_id = first_obs_id
                for track_idx, particle_idx in enumerate(eventWise.Track_Particle):
                    if (particle_idx not in shower_leaf_particle_idxs and
                        track_idx not in jet_tracks):
                        continue
                    draw_id = track_idx + first_obs_id
                    obs_draw_map[track_idx] = draw_id
                    if particle_idx in shower.particle_idxs:
                        self.addEdge(particle_idx, draw_id)
                    self.addNode(draw_id, "Track", track_particle, observable_shape)
                next_free_obs_id = draw_id
                for tower_idx, particle_idxs in enumerate(eventWise.Tower_Particles):
                    if tower_idx not in jet_towers:
                        try:
                            # have a look to see if there is a particle in the shower
                            inside_paticle = next(p for p in particle_idxs if p in shower_leaf_particle_idxs)
                        except StopIteration:
                            continue  # there isn't
                    draw_id = tower_idx + next_free_obs_id
                    obs_draw_map[tower_idx] = draw_id
                    for particle_idx in particle_idxs:
                        if particle_idx in shower_leaf_particle_idxs:
                            self.addEdge(particle_idx, draw_id)
                    self.addNode(draw_id, "Tower", tower_particle, observable_shape)
            else:  # not using tracks towers
                # in the shower and the tracks
                root_particles = np.where(eventWise.is_root)[0]
                for particle_idx in root_particles:
                    if particle_idx not in shower_leaf_particle_idxs:
                        continue
                    draw_id = particle_idx + first_obs_id
                    obs_draw_map[particle_idx] = draw_id
                    self.addEdge(particle_idx, draw_id)
                    self.addNode(draw_id, "Observable", internal_particle, observable_shape)
            self.addRank(list(obs_draw_map.values()))
        first_jet_id = int(np.max(list(obs_draw_map.values()),
                                  initial=first_obs_id)) + 1
             
        if jet_name is not None:
            inputIdx = getattr(eventWise, jet_name+"_InputIdx")[jet_num]
            distance = getattr(eventWise, jet_name+"_JoinDistance")[jet_num]
            for inpIdx, dis in zip(inputIdx, distance):
                label = "input"
                if dis != 0:
                    label = f"jet {dis:.1e}"
                self.addNode(inpIdx + first_jet_id, label)
            # connect the jet to the particles
            for j_idx, p_idx in zip(jet_ends_inputIdx, jet_particles):
                self.addEdge(p_idx, j_idx + first_jet_id)
            parents = getattr(eventWise, jet_name+"_Parent")[jet_num]
            for parent, child in zip(parents, inputIdx):
                if parent == -1:
                    continue
                self.addEdge(child+first_jet_id, parent+first_jet_id)
            # add the ranks
            ranks = getattr(eventWise, jet_name+"_Rank")[jet_num]
            list_ranks = sorted(set(ranks))
            for rank in list_ranks:
                rank_ids = [i+first_jet_id for i, r in zip(inputIdx, ranks)
                            if r == rank]
                self.addRank(rank_ids)

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

    
def main():
    """ Launch file, makes and saves a dot graph """
    repeat = True
    eventWise = Components.EventWise.from_file("/home/henry/lazy/dataset2/h1bBatch2_particles.awkd")
    while repeat:
        from tree_tagger import FormShower
        eventWise.selected_index = int(input("Event number: "))
        showers = FormShower.get_showers(eventWise)
        jet_name = "HomeJets"
        chosen_showers = []
        for i, shower in enumerate(showers):
            shower_roots = [shower.labels[i] for i in shower.root_local_idxs]
            if 'b' not in shower_roots and 'bbar' not in shower_roots:
                continue
            chosen_showers.append(shower)
            print(f"Shower roots {shower.root_idxs}")
            max_children = max([len(d) for d in shower.children])
            end_ids = [shower.particle_idxs[e] for e in shower.ends]
            print(f"Drawing shower {i}, has {max_children} max children. Daughters to particles ratio = {max_children/len(shower.children)}")
            # pick the jet with largest overlap
            largest_overlap = 0
            picked_jet = 0
            for i in range(len(eventWise.HomeJets_Parent)):
                is_external = getattr(eventWise, jet_name + "_Child1")[i] < 0
                input_idx = getattr(eventWise, jet_name + "_InputIdx")[i][is_external]
                jet_particles = eventWise.JetInputs_SourceIdx[input_idx]
                matches_here = sum([p in end_ids for p in jet_particles])
                if matches_here > largest_overlap:
                    largest_overlap = matches_here
                    picked_jet = i
            print(f"A jet contains {largest_overlap} out of {len(end_ids)} end products")
            graph = DotGraph(shower=shower, eventWise=eventWise,
                             jet_name=jet_name, jet_num=picked_jet,
                             use_TracksTowers=True)
            base_name = f"event{eventWise.selected_index}_plot"
            dotName = base_name + str(i) + ".dot"
            legendName = base_name + str(i) + "_ledg.dot"
            with open(dotName, 'w') as dotFile:
                dotFile.write(str(graph))
            with open(legendName, 'w') as dotFile:
                dotFile.write(graph.legend)
        #amalgam_shower = chosen_showers[0]
        #if len(chosen_showers)>1:
        #    for shower in chosen_showers[1:]:
        #        amalgam_shower.amalgamate(shower)
        #print("Drawing the amalgam of all b showers")
        #graph = DotGraph(shower=amalgam_shower, observables=obs)
        #dotName = f"event{eventWise.selected_index}_mixing_plot.dot"
        #legendName ="mixing_ledg.dot"
        #with open(dotName, 'w') as dotFile:
        #    dotFile.write(str(graph))
        #with open(legendName, 'w') as dotFile:
        #    dotFile.write(graph.legend)
        repeat = InputTools.yesNo_question("Again? ")


if __name__ == '__main__':
    main()
