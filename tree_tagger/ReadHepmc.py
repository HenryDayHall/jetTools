import numpy as np
import awkward
import os
from collections import Counter
from tree_tagger import Components
import csv
from ipdb import set_trace as st


class Hepmc(Components.EventWise):
    def __init__(self, dir_name, save_name, start=0, stop=np.inf, **kwargs):
        self.event_information_cols = ["Event_n", "N_multi_particle_inter",
                                       "Event_scale", "Alpha_QCD", "Alpha_QED",
                                       "Signal_process_id", "Barcode_for_signal_process",
                                       "N_vertices_in_event", "Barcode_beam_particle1",
                                       "Barcode_beam_particle2", "Len_random_state_list",
                                       "Random_state_ints", "Len_weight_list",
                                       "Weight_list"]
        self.weight_cols = ["N_weight_names",  # must == "len_weight_list"
                            "Weight_names"]
        self.units_cols = ["Momentum", "Length"]
        self.cross_section_cols = ["Cross_section_pb", "Cross_section_error_pb"]
        self.vertex_cols  = ["Vertex_barcode", "Id", "X", "Y", "Z", "Ctau",
                             "N_orphans", "N_out", "N_vertex_weights"]
        self.vertex_convertions  = [int, int, float, float, float, float,
                                    int, int, int]
        self.particle_cols = ["Particle_barcode", "MCPID", "Px", "Py", "Pz", "Energy", "Generated_mass",
                              "Status_code", "Polarization_theta", "Polarization_phi",
                              "End_vertex_barcode", "N_flow_codes", "Flow_codes", "Antiflow_codes",
                              "Start_vertex_barcode",
                              "Parents", "Children", "Is_root", "Is_leaf"]
        self.particle_convertions = [int, int, float, float, float, float, float,
                                     int, int, float,
                                     int, int]  # cannot include the list of flow codes
                                     # so do't include anythng after them
        expected_columns = self.event_information_cols + self.weight_cols + self.units_cols + \
                           self.cross_section_cols + self.vertex_cols + self.particle_cols
        if 'columns' in kwargs: 
            assert kwargs['columns'] == expected_columns, f'Expected; {expected_columns},'+\
                                                          f' in input; {kwargs["columns"]}'
            assert 'contents' in kwargs, 'Initilised with columns but no content.'
        else:  # readign from hepmc
            self.columns = expected_columns
            self._init_column_contents()
            # there are more but unfortunatly I cannot be bothered to add them right now
            # parse the event
            filepath = os.path.join(dir_name, save_name)
            self._parse_events(filepath, start, stop)
            # figure out which particles created which other particles
            self._assign_heritage()
            self._fix_column_contents()
            # self.check_colour_flow()  # never worked... colour flow doesn't seem to be conserved
            # change the savename to indicate the processing
            save_name = save_name.split('.', 1)[0] + '_hepmc.awkd'
        super().__init__(dir_name, save_name, **kwargs)
        self.n_particles = sum([len(evt) for evt in self.Particle_barcode])
        self.n_vertices = sum([len(evt) for evt in self.Vertex_barcode])
        self.n_events = len(self.Event_n)

    def _init_column_contents(self):
        """ initilise coloum contents just as a dict of lists,
            assumption is they need to be filled out so keep them mutable """
        self._column_contents = {name : [] for name in self.columns}

    def _fix_column_contents(self):
        """ to be called once the collumns are filed out """
        for name in self._column_contents:
            self._column_contents[name] = awkward.fromiter(self._column_contents[name])

    def __str__(self):
        return f"Hepmc file {self.save_name} for event {self.Event_n[0]} to {self.Event_n[-1]}"

    def __repr__(self):
        return self.__str__()

    def _assign_heritage(self):
        # this section requires numpy indexing on the barcodes
        self.Start_vertex_barcode = awkward.fromiter(self.Start_vertex_barcode)
        self.End_vertex_barcode = awkward.fromiter(self.End_vertex_barcode)
        self.Vertex_barcode = awkward.fromiter(self.Vertex_barcode)
        # each event will have completely seperate heritage
        for event_n, (barcodes, start_barcodes, end_barcodes) in enumerate(zip(self.Vertex_barcode,
                                                                               self.Start_vertex_barcode,
                                                                               self.End_vertex_barcode)):
            children = []
            parents = []
            for particle_n, (start_b, end_b) in enumerate(zip(start_barcodes, end_barcodes)):
                if start_b in barcodes:
                    indices = np.where(end_barcodes == start_b)[0]
                    parents.append(indices.tolist())
                else:
                    raise RuntimeError("This shouldn't happen... all particles sit below a vertex")
                if end_b in barcodes:
                    indices = np.where(start_barcodes == end_b)[0]
                    children.append(indices.tolist())
                else:
                    children.append([])
            # now particles that are the beam particles are their own mothers
            b1_idx = next(i for i, particle_b in enumerate(self.Particle_barcode[event_n])
                          if particle_b == self.Barcode_beam_particle1[event_n])
            b2_idx = next(i for i, particle_b in enumerate(self.Particle_barcode[event_n])
                          if particle_b == self.Barcode_beam_particle2[event_n])
            assert parents[b1_idx] == [b1_idx]
            assert parents[b2_idx] == [b2_idx]
            # so tidy this up
            parents[b1_idx] = []
            children[b1_idx].remove(b1_idx)
            parents[b2_idx] = []
            children[b2_idx].remove(b2_idx)
            # put these result into the arrays
            self.Parents[event_n] = parents
            self.Children[event_n] = children
            self.Is_leaf[event_n] = [c==[] for c in children]
            self.Is_root[event_n] = [p==[] for p in parents]

    def check_colour_flow(self):
        for event_n, vertex_barcodes in enumerate(self.Vertex_barcode):
            start_barcodes = self.Start_vertex_barcode[event_n]
            end_barcodes = self.End_vertex_barcode[event_n]
            for vertex_b in vertex_barcodes:
                # find the list of particles bearing this barcode
                in_indices = np.where(start_barcodes == vertex_b)[0]
                out_indices = np.where(end_barcodes == vertex_b)[0]
                # tally the colour flows in and out
                link_counter = Counter()
                for in_index in in_indices:
                    for flow in self.Flow_codes[event_n, in_index]:
                        link_counter[flow] += 1
                    for antiflow in self.Antiflow_codes[event_n, in_index]:
                        link_counter[antiflow] -= 1
                for out_index in out_indices:
                    for flow in self.Flow_codes[event_n, out_index]:
                        link_counter[flow] -= 1
                    for antiflow in self.Antiflow_codes[event_n, out_index]:
                        link_counter[antiflow] += 1
                del link_counter[None]
                for flow in link_counter:
                    try:
                        assert link_counter[flow] == 0, f"Vertex barcode {vertex_b} " + \
                                                        f"has {link_counter[flow]} outgoing colour " +\
                                                        f"flows for colour flow {flow}"
                    except Exception:
                        st()


    def _parse_events(self, filepath, start=0, stop=np.inf):
        assert os.path.exists(filepath), f"Can't see that file; {filepath}"
        with open(filepath, 'r') as this_file:
            csv_reader = csv.reader(this_file, delimiter=' ', quotechar='"')
            event_reached = 0
            event_line = None
            # move to the start of the start event
            for line in csv_reader:
                if len(line) == 0:
                    continue
                if line[0] == "E":
                    if event_reached >= start:
                        event_line = line
                        break
                    event_reached += 1
            assert event_line is not None, "Didn't find any events!"
            # continue till the stop event
            while event_line is not None:
                # the event parser will hand us the next event line
                event_line = self._parse_event(event_reached, event_line, csv_reader)
                event_reached += 1
                if event_reached >= stop:
                    break

    def _parse_event(self, event_n, event_line, csv_reader):
        # start by adding default entries, incase anythign dosn't get content
        add_row = self.particle_cols + self.vertex_cols
        for name in add_row:
            table = self.__getattr__(name)
            table.append([])
        add_default = [c for c in self.columns if c not in add_row]
        for name in add_default:
            table = self.__getattr__(name)
            table.append(np.nan)
        # first process the event line
        self._process_event_line(event_n, event_line)
        # lines start with a key that indicates their content
        # the next few lines might be any of N, U C, H or F
        header_keys = ['N', 'U', 'C', 'H', 'F']
        try:
            next_line = next(csv_reader)
        except StopIteration: # reached eof
            return
        if next_line[0] == 'E':  # reach next event
            return next_line

        while next_line[0] in header_keys:
            self._process_header_line(event_n, next_line)
            try:
                next_line = next(csv_reader)
            except StopIteration: # reached eof
                return
            if next_line[0] == 'E':  # reach next event
                return next_line
        # now the next line should be a vertex
        # then add an new row for the particles and vertices
        assert next_line[0] == 'V'
        # get indices ahead of time
        vertex_indices = {name: i+1 for i, name in enumerate(self.vertex_cols)} 
        vertex_barcode_index = self.vertex_cols.index("Vertex_barcode") + 1
        particle_indices = {name: i+1 for i, name in enumerate(self.particle_cols)
                            if name not in ["Flow_codes", "Antiflow_codes", "Start_vertex_barcode"]  # these are exceptional
                            and name not in ["Parents", "Children", "Is_root", "Is_leaf"]}  # and these will be determined later
        n_flow_index = self.particle_cols.index("N_flow_codes") + 1
        first_flow_index = self.particle_cols.index("Flow_codes") + 1
        # for speed reason avpid function calls
        last_vertex_barcode = int(next_line[vertex_barcode_index])
        for convertion, name in zip(self.vertex_convertions, vertex_indices):
            self.__getattr__(name)[-1].append(convertion(next_line[vertex_indices[name]]))
        # now everythng should be particles and vertices
        for line in csv_reader:
            if line[0] == 'E':  # reached next event
                return line
            elif line[0] == 'V':  # new vertex
                last_vertex_barcode = int(line[vertex_barcode_index])
                for convertion, name in zip(self.vertex_convertions, vertex_indices):
                    self.__getattr__(name)[-1].append(convertion(line[vertex_indices[name]]))
            elif line[0] == 'P':  # new particle
                for convertion, name in zip(self.particle_convertions, particle_indices):
                    try:
                        self.__getattr__(name)[-1].append(convertion(line[particle_indices[name]]))
                    except Exception:
                        st()
                # now deal with the two speciel cases
                self.Start_vertex_barcode[-1].append(last_vertex_barcode)
                if len(line) > len(particle_indices):  # there are flow codes
                    last_flow_index = first_flow_index + int(line[n_flow_index])*2  # *2 becuase there are flow and antiflow
                    self.Flow_codes[-1].append([int(x) for x in line[first_flow_index:last_flow_index+1:2]])
                    self.Antiflow_codes[-1].append([int(x) for x in line[first_flow_index+1:last_flow_index+1:2]])
                else:
                    assert int(line[n_flow_index]) == 0

    def _process_event_line(self, event_n, event_line):
        assert event_line[0][0] == 'E'
        i = 1
        self.Event_n[event_n] = int(event_line[i])
        i += 1
        self.N_multi_particle_inter[event_n] = int(event_line[i])
        i += 1
        self.Event_scale[event_n] = float(event_line[i])
        i += 1
        self.Alpha_QCD[event_n] = float(event_line[i])
        i += 1
        self.Alpha_QED[event_n] = float(event_line[i])
        i += 1
        self.Signal_process_id[event_n] = int(event_line[i])
        i += 1
        self.Barcode_for_signal_process[event_n] = int(event_line[i])
        i += 1
        self.N_vertices_in_event[event_n] = int(event_line[i])
        i += 1
        self.Barcode_beam_particle1[event_n] = int(event_line[i])
        i += 1
        self.Barcode_beam_particle2[event_n] = int(event_line[i])
        i += 1
        self.Len_random_state_list[event_n] = int(event_line[i])
        i += 1
        self.Random_state_ints[event_n] = [float(x) for x in
                                           event_line[i:i+self.Len_random_state_list[-1]]]
        i += self.Len_random_state_list[-1]
        self.Len_weight_list[event_n] = int(event_line[i])
        i += 1
        self.Weight_list[event_n] = [float(x) for x in
                                     event_line[i:i+self.Len_weight_list[-1]]]

    def _process_header_line(self, event_n, header_line):
        if header_line[0] == 'N':
            i = 1 # start from 1 because the first item is the key
            self.N_weight_names[event_n] = int(header_line[i])
            i += 1
            self.Weight_names[event_n] = header_line[i:]
        elif header_line[0] == 'U':
            i = 1
            self.Momentum[event_n] = header_line[i]
            i += 1
            self.Length[event_n] = header_line[i]
        elif header_line[0] == 'C':
            i = 1
            self.Cross_section_pb[event_n] = float(header_line[i])
            i += 1
            self.Cross_section_error_pb[event_n] = float(header_line[i])


def main():
    filepath = "/home/henry/lazy/h1bBatch2.hepmc"
    event = Hepmc(filepath, 0, 1)
    return event

