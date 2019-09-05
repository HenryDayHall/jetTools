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
            observables = kwargs.get('observables', None)
            jet = kwargs.get('jet', None)
            self.fromShower(shower, observables, jet)

    def fromShower(self, shower, observables=None, jet=None):
        """ Construct the graph from a shower object
        

        Parameters
        ----------
        shower : Shower
            an object that holds all infomration about the structure of a particle shower
            
        """
        # add the edges
        for this_id, this_mothers in zip(shower.global_ids, shower.mothers):
            for mother in this_mothers:
                if mother in shower.global_ids:
                    self.addEdge(mother, this_id)
        # add the labels
        shower_leaf_global_ids = shower.global_ids[shower.ends]
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
        for this_id, label in zip(shower.global_ids, shower.labels):
            colour = internal_particle
            if this_id in shower.root_gids:
                colour=root_particle
            elif this_id in shower.outside_connection_gids:
                colour=outside_particle
            shape = None
            self.addNode(this_id, label, colour=colour, shape=shape)
        # add the ranks
        ranks = shower.find_ranks()
        rank_keys = sorted(list(set(ranks)))
        for key in rank_keys:
            mask = ranks == key
            rank_ids = np.array(shower.global_ids)[mask]
            self.addRank(rank_ids)
        first_obs_id = np.max(shower.global_ids) + 1
        obs_rank = max(ranks) + 1
        obs_draw_map = {}
        if observables is not None:
            # make an observable layer
            # observable part of legend
            observable_shape = "diamond"
            if observables.has_tracksTowers:
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
            used_obs_ids = []
            # in the shower 
            used_obs_ids += [oid for gid, oid in observables.global_to_obs.items()
                             if gid in shower_leaf_global_ids]
            if jet is not None:
                used_obs_ids += [oid for oid in jet.global_obs_ids if oid > -1]
            for oid, obj in zip(observables.global_obs_ids, observables.objects):
                if oid not in used_obs_ids:
                    continue
                draw_id = oid + first_obs_id
                obs_draw_map[oid] = draw_id
                name = "Observable"
                colour = internal_particle
                shape = observable_shape
                if type(obj) == Components.MyTrack:
                    name = "Track"
                    colour = track_particle
                    if obj.global_id in shower.global_ids:
                        self.addEdge(obj.global_id, draw_id)
                    self.addNode(draw_id, name, colour, shape)
                elif type(obj) == Components.MyTower:
                    name = "Tower"
                    colour = tower_particle
                    for global_id in obj.global_ids:
                        if global_id in shower.global_ids:
                            self.addEdge(global_id, draw_id)
                    self.addNode(draw_id, name, colour, shape)
                else:
                    self.addEdge(obj.global_id, draw_id)
                    self.addNode(draw_id, name, colour, shape)
            
            self.addRank(list(obs_draw_map.values()))
        first_jet_id = int(np.max(list(obs_draw_map.values()),
                                  initial=first_obs_id)) + 1
             
        if jet is not None:
            # jet ids must be created,
            # some will corrispond to the observations (may not be the full set)
            # others will be generated statig from the max of the obs ids
            free_id = first_jet_id
            jet_draw_map = {}
            for global_jet_id, obs_id in zip(jet.global_jet_ids, jet.global_obs_ids):
                if obs_id == -1:
                    jet_draw_map[global_jet_id] = free_id
                    free_id += 1
                else:
                    jet_draw_map[global_jet_id] = obs_draw_map[obs_id]
            jet_draw_ids = np.array(list(jet_draw_map.values()))
            
            # add the highest shower rank t all the jet ranks
            jet_cluster = "coral"
            self.addLegendNode(legened_id, f"PsudoJet (join distance)", colour=jet_cluster)
            for index, (draw_id, mother_jid, distance) in enumerate(zip(jet_draw_ids, jet.mothers, jet.distances)):
                # add the edges
                if mother_jid >= 0:
                    mother_draw_id = jet_draw_map[mother_jid]
                    if (mother_draw_id in jet_draw_ids
                        or mother_draw_id in obs_draw_map.values()):
                        self.addEdge(draw_id, mother_draw_id)
                # if needed add the node
                if draw_id in obs_draw_map.values():
                    continue
                else:
                    self.addNode(draw_id, f"jet {distance:.2e}", colour=jet_cluster, shape=None)
            # add the ranks
            adjusted_jet_rank = jet.ranks + max(ranks)
            rank_keys = sorted(list(set(adjusted_jet_rank)))
            for key in rank_keys:
                if key == max(ranks):
                    continue  # skip draw as it's in the obs
                mask = adjusted_jet_rank == key
                rank_ids = jet_draw_ids[mask]
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
    while repeat:
        event_n = int(input("Event number: "))
        from tree_tagger import ReadSQL
        event, tracks, towers, obs = ReadSQL.main(event_n)
        from tree_tagger import FormShower
        from tree_tagger import FormJets
        showers = FormShower.get_showers(event)
        jets = FormJets.PsudoJets(obs)
        jets.assign_mothers()
        jets = jets.split()
        chosen_showers = []
        for i, shower in enumerate(showers):
            shower_roots = [shower.labels[i] for i in shower.root_idxs]
            if 'b' not in shower_roots and 'bbar' not in shower_roots:
                continue
            chosen_showers.append(shower)
            print(f"Shower roots {shower.root_gids}")
            max_daughters = max([len(d) for d in shower.daughters])
            end_ids = [shower.global_ids[e] for e in shower.ends]
            print(f"Drawing shower {i}, has {max_daughters} max daughters. Daughters to particles ratio = {max_daughters/len(shower.daughters)}")
            # pick the jet with largest overlap
            largest_overlap = 0
            picked_jet = jets[0]
            for jet in jets:
                obs_ids = [oid for oid in jet.global_obs_ids if not oid == -1]
                matches_here = 0
                for ob_oid, ob in zip(obs.global_obs_ids, obs.objects):
                    if isinstance(ob, Components.MyTrack):
                        # it's a track obs
                                                          # the track is in the jet
                        if (ob_oid in obs_ids  
                            and ob.global_id in end_ids):  # the track is in the shower
                            matches_here += 1
                    else: # it's a tower
                        if ob_oid in obs_ids:   # the tower is in the jet
                            for gid in ob.global_ids:     # for each particle in the tower
                                if gid in end_ids:       # the particle is in the shower
                                    matches_here += 1
                if matches_here > largest_overlap:
                    largest_overlap = matches_here
                    picked_jet = jet
            print(f"A jet contains {largest_overlap} out of {len(end_ids)} end products")
            graph = DotGraph(shower=shower, jet=picked_jet, observables=obs)
            base_name = f"event{event_n}_plot"
            dotName = base_name + str(i) + ".dot"
            legendName = base_name + str(i) + "_ledg.dot"
            with open(dotName, 'w') as dotFile:
                dotFile.write(str(graph))
            with open(legendName, 'w') as dotFile:
                dotFile.write(graph.legend)
        amalgam_shower = chosen_showers[0]
        if len(chosen_showers)>1:
            for shower in chosen_showers[1:]:
                amalgam_shower.amalgamate(shower)
        print("Drawing the amalgam of all b showers")
        graph = DotGraph(shower=amalgam_shower, observables=obs)
        dotName = f"event{event_n}_mixing_plot.dot"
        legendName ="mixing_ledg.dot"
        with open(dotName, 'w') as dotFile:
            dotFile.write(str(graph))
        with open(legendName, 'w') as dotFile:
            dotFile.write(graph.legend)
        repeat = InputTools.yesNo_question("Again? ")


if __name__ == '__main__':
    main()
