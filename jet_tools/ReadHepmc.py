import os
import csv
import gzip

import collections
#from ipdb import set_trace as st
import numpy as np
import awkward

from jet_tools import Components, InputTools

class Hepmc(Components.EventWise):
    """ """
    event_information_cols = ["Event_n", "N_multi_particle_inter",
                              "Event_scale", "Alpha_QCD", "Alpha_QED",
                              "Signal_process_id", "Barcode_for_signal_process",
                              "N_vertices_in_event", "Barcode_beam_particle1",
                              "Barcode_beam_particle2", "Len_random_state_list",
                              "Random_state_ints", "Len_weight_list",
                              "Weight_list"]
    event_information_conversions = [int, int,
                                     float, float, float,
                                     int, int,
                                     int, int,
                                     int, int,
                                     int, int, 
                                     float]
    weight_cols = ["N_weight_names",  # must == "len_weight_list"
                   "Weight_names"]
    units_cols = ["Momentum", "Length"]
    cross_section_cols = ["Cross_section_pb", "Cross_section_error_pb"]
    vertex_cols  = ["Vertex_barcode", "Id", "X", "Y", "Z", "Ctau",
                    "N_orphans", "N_out", "N_vertex_weights"]
    vertex_convertions  = [int, int, float, float, float, float,
                           int, int, int]
    # parents and children hold list indices
    particle_cols = ["Particle_barcode", "MCPID", "Px", "Py", "Pz", "Energy", "Generated_mass",
                     "Status_code", "Polarization_theta", "Polarization_phi",
                     "End_vertex_barcode", "N_flow_codes", "Flow_codes", "Antiflow_codes",
                     "Start_vertex_barcode",
                     "Parents", "Children", "Is_root", "Is_leaf"]
    particle_convertions = [int, int, float, float, float, float, float,
                            int, float, float,
                            int, int, int, int]  # cannot include the list of flow codes
                            # so do't include anythng after them
    def __init__(self, file_name, start=0, stop=np.inf, **kwargs):
        expected_columns = Hepmc.event_information_cols + Hepmc.weight_cols + Hepmc.units_cols + \
                           Hepmc.cross_section_cols + Hepmc.vertex_cols + Hepmc.particle_cols
        dir_name, save_name = os.path.split(file_name)
        if 'columns' in kwargs: 
            assert kwargs['columns'] == expected_columns, f'Expected; {expected_columns},'+\
                                                          f' in input; {kwargs["columns"]}'
            assert 'contents' in kwargs, 'Initilised with columns but no content.'
        else:  # readign from hepmc
            self.columns = expected_columns
            self.prepared_contents = {name: [] for name in expected_columns}
            # there are more but unfortunatly I cannot be bothered to add them right now
            # parse the event
            print("Parsing events")
            self._parse_events(file_name, start, stop)
            # figure out which particles created which other particles
            print("Assigning heritage")
            self._assign_heritage()
            print("Fixing column contents")
            self._fix_column_contents()
            # self.check_colour_flow()  # never worked... colour flow doesn't seem to be conserved
            # change the savename to indicate the processing
            save_name = save_name.split('.', 1)[0] + '_hepmc.awkd'
            kwargs["columns"] = self.columns
            kwargs["contents"] = self.prepared_contents
        super().__init__(file_name, **kwargs)
        self.n_particles = sum([len(evt) for evt in self.Particle_barcode])
        self.n_vertices = sum([len(evt) for evt in self.Vertex_barcode])
        self.n_events = len(self.Event_n)

    def _fix_column_contents(self):
        """to be called once the collumns are filed out"""
        for name in self.prepared_contents:
            self.prepared_contents[name] = awkward.fromiter(self.prepared_contents[name])
        self._loaded_contents = {}

    def __str__(self):
        return f"Hepmc file {self.save_name} for event {self.Event_n[0]} to {self.Event_n[-1]}"

    def __repr__(self):
        return self.__str__()

    def _assign_heritage(self):
        """ """
        # this section requires numpy indexing on the barcodes
        self.prepared_contents["Start_vertex_barcode"] = awkward.fromiter(self.prepared_contents["Start_vertex_barcode"])
        self.prepared_contents["End_vertex_barcode"] = awkward.fromiter(self.prepared_contents["End_vertex_barcode"])
        self.prepared_contents["Vertex_barcode"] = awkward.fromiter(self.prepared_contents["Vertex_barcode"])
        n_events = len(self.prepared_contents["Vertex_barcode"])
        # each event will have completely seperate heritage
        for event_index, (barcodes, start_barcodes, end_barcodes) in enumerate(zip(self.prepared_contents["Vertex_barcode"],
                                                                               self.prepared_contents["Start_vertex_barcode"],
                                                                               self.prepared_contents["End_vertex_barcode"])):
            if event_index % 100 == 0:
                print(f"{event_index/n_events:.1%}", end='\r', flush=True)
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
            b1_idx = next(i for i, particle_b in enumerate(self.prepared_contents["Particle_barcode"][event_index])
                          if particle_b == self.prepared_contents["Barcode_beam_particle1"][event_index])
            b2_idx = next(i for i, particle_b in enumerate(self.prepared_contents["Particle_barcode"][event_index])
                          if particle_b == self.prepared_contents["Barcode_beam_particle2"][event_index])
            assert parents[b1_idx] == [b1_idx]
            assert parents[b2_idx] == [b2_idx]
            # so tidy this up
            parents[b1_idx] = []
            children[b1_idx].remove(b1_idx)
            parents[b2_idx] = []
            children[b2_idx].remove(b2_idx)
            # put these result into the arrays
            self.prepared_contents["Parents"][event_index] = parents
            self.prepared_contents["Children"][event_index] = children
            self.prepared_contents["Is_leaf"][event_index] = [c==[] for c in children]
            self.prepared_contents["Is_root"][event_index] = [p==[] for p in parents]

    def check_colour_flow(self):
        """ """
        for event_index, vertex_barcodes in enumerate(self.prepared_contents["Vertex_barcode"]):
            start_barcodes = self.prepared_contents["Start_vertex_barcode"][event_index]
            end_barcodes = self.prepared_contents["End_vertex_barcode"][event_index]
            for vertex_b in vertex_barcodes:
                # find the list of particles bearing this barcode
                in_indices = np.where(start_barcodes == vertex_b)[0]
                out_indices = np.where(end_barcodes == vertex_b)[0]
                # tally the colour flows in and out
                link_counter = collections.Counter()
                for in_index in in_indices:
                    for flow in self.prepared_contents["Flow_codes"][event_index, in_index]:
                        link_counter[flow] += 1
                    for antiflow in self.prepared_contents["Antiflow_codes"][event_index, in_index]:
                        link_counter[antiflow] -= 1
                for out_index in out_indices:
                    for flow in self.prepared_contents["Flow_codes"][event_index, out_index]:
                        link_counter[flow] -= 1
                    for antiflow in self.prepared_contents["Antiflow_codes"][event_index, out_index]:
                        link_counter[antiflow] += 1
                del link_counter[None]
                for flow in link_counter:
                    assert link_counter[flow] == 0, f"Vertex barcode {vertex_b} " + \
                                                    f"has {link_counter[flow]} outgoing colour " +\
                                                    f"flows for colour flow {flow}"

    
    def _open_hep(self, filepath):
        if filepath.endswith('.gz'):
            f = gzip.open(filepath, mode='rt')
        else:
            f = open(filepath, 'r')
        return f


    def _parse_events(self, filepath, start=0, stop=np.inf):
        """
        

        Parameters
        ----------
        filepath :
            param start: (Default value = 0)
        stop :
            Default value = np.inf)
        start :
            (Default value = 0)

        Returns
        -------

        
        """
        assert os.path.exists(filepath), f"Can't see that file; {filepath}"
        with self._open_hep(filepath) as this_file:
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
                if event_reached % 100 == 0:
                    print(f"reached event {event_reached}", end='\r', flush=True)
                # the event parser will hand us the next event line
                event_line = self._parse_event(event_reached-start, event_line, csv_reader)
                event_reached += 1
                if event_reached >= stop:
                    break

    def _parse_event(self, event_index, event_line, csv_reader):
        """
        

        Parameters
        ----------
        event_index :
            param event_line:
        csv_reader :
            
        event_line :
            

        Returns
        -------

        
        """
        # start by adding default entries, incase anythign dosn't get content
        add_row = Hepmc.particle_cols + Hepmc.vertex_cols
        for name in add_row:
            table = self.prepared_contents[name]
            table.append([])
        add_default = [c for c in self.columns if c not in add_row]
        for name in add_default:
            table = self.prepared_contents[name]
            table.append(np.nan)
        # first process the event line
        self._process_event_line(event_index, event_line)
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
            self._process_header_line(event_index, next_line)
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
        vertex_indices = {name: i+1 for i, name in enumerate(Hepmc.vertex_cols)} 
        vertex_barcode_index = Hepmc.vertex_cols.index("Vertex_barcode") + 1
        particle_indices = {name: i+1 for i, name in enumerate(Hepmc.particle_cols)
                            if name not in ["Flow_codes", "Antiflow_codes", "Start_vertex_barcode"]  # these are exceptional
                            and name not in ["Parents", "Children", "Is_root", "Is_leaf"]}  # and these will be determined later
        n_flow_index = Hepmc.particle_cols.index("N_flow_codes") + 1
        first_flow_index = Hepmc.particle_cols.index("Flow_codes") + 1
        # for speed reason avpid function calls
        last_vertex_barcode = int(next_line[vertex_barcode_index])
        for convertion, name in zip(Hepmc.vertex_convertions, vertex_indices):
            self.prepared_contents[name][-1].append(convertion(next_line[vertex_indices[name]]))
        # now everythng should be particles and vertices
        for line in csv_reader:
            if line[0] == 'E':  # reached next event
                return line
            elif line[0] == 'V':  # new vertex
                last_vertex_barcode = int(line[vertex_barcode_index])
                for convertion, name in zip(Hepmc.vertex_convertions, vertex_indices):
                    self.prepared_contents[name][-1].append(convertion(line[vertex_indices[name]]))
            elif line[0] == 'P':  # new particle
                for convertion, name in zip(Hepmc.particle_convertions, particle_indices):
                    self.prepared_contents[name][-1].append(convertion(line[particle_indices[name]]))
                # now deal with the two speciel cases
                self.prepared_contents["Start_vertex_barcode"][-1].append(last_vertex_barcode)
                if len(line) > len(particle_indices):  # there are flow codes
                    last_flow_index = first_flow_index + int(line[n_flow_index])*2  # *2 becuase there are flow and antiflow
                    self.prepared_contents["Flow_codes"][-1].append([int(x) for x in line[first_flow_index:last_flow_index+1:2]])
                    self.prepared_contents["Antiflow_codes"][-1].append([int(x) for x in line[first_flow_index+1:last_flow_index+1:2]])
                else:
                    assert int(line[n_flow_index]) == 0

    def _process_event_line(self, event_index, event_line):
        """
        

        Parameters
        ----------
        event_index :
            param event_line:
        event_line :
            

        Returns
        -------

        
        """
        assert event_line[0][0] == 'E'
        multi_item_pairs = {"Random_state_ints": "Len_random_state_list",
                            "Weight_list": "Len_weight_list"}
        i = 1
        for name, convert in zip(Hepmc.event_information_cols, Hepmc.event_information_conversions):
            if name in multi_item_pairs:
                num_items = self.prepared_contents[multi_item_pairs[name]][event_index]
                items = [convert(event_line[i + n]) for n in range(num_items)]
                i += num_items
                self.prepared_contents[name][event_index] = items
            else:
                self.prepared_contents[name][event_index] = convert(event_line[i])
                i += 1

    def _process_header_line(self, event_index, header_line):
        """
        

        Parameters
        ----------
        event_index :
            param header_line:
        header_line :
            

        Returns
        -------

        
        """
        if header_line[0] == 'N':
            i = 1 # start from 1 because the first item is the key
            self.prepared_contents["N_weight_names"][event_index] = int(header_line[i])
            i += 1
            self.prepared_contents["Weight_names"][event_index] = header_line[i:]
        elif header_line[0] == 'U':
            i = 1
            self.prepared_contents["Momentum"][event_index] = header_line[i]
            i += 1
            self.prepared_contents["Length"][event_index] = header_line[i]
        elif header_line[0] == 'C':
            i = 1
            self.prepared_contents["Cross_section_pb"][event_index] = float(header_line[i])
            i += 1
            self.prepared_contents["Cross_section_error_pb"][event_index] = float(header_line[i])


def main():
    """ """
    input_file = InputTools.get_file_name("Input file? ", 'hepmc').strip()
    new_name = InputTools.get_file_name("Output file? ", 'awkd').strip()
    dir_name, save_name = os.path.split(input_file)
    new_dir_name, new_save_name = os.path.split(new_name)
    print(f"Reading {save_name}")
    eventWise = Hepmc(dir_name, save_name, 0, 10000)
    eventWise.dir_name = new_dir_name
    eventWise.save_name = new_save_name
    print(f"Writing {eventWise.save_name}")
    eventWise.write()


if __name__ == '__main__':
    if InputTools.yesNo_question("Do you want to read that hepmc file and write it? "):
        if InputTools.yesNo_question("Really? Did you close everything else? "):
            main()

