import numpy as np
import os
from collections import Counter
import Components
import csv
from ipdb import set_trace as st


class Hepmc_event:
    def __init__(self, filepath, lines):
        self.filepath = filepath
        self.event_information = {"event_n": None, "n_multi_particle_inter": None,
                                  "event_scale": None, "alpha_QCD": None, "alpha_QED": None,
                                  "signal_process_id": None, "barcode_for_signal_process": None,
                                  "n_vertices_in_event": None, "barcode_beam_particle1": None,
                                  "barcode_beam_particle2": None, "len_random_state_list": None,
                                  "random_state_ints": None, "len_weight_list": None,
                                  "weight_list": None}
        self.weight_names = {"n_entries" : None,  # must == "len_weight_list"
                             "weight_names" : None}
        self.units = {"momentum" : None, "length" : None}
        self.cross_section = {"cross_section(pb)" : None, "error(pb)": None}
        # there are more but unfortunatly I cannot be bothered to add them right now
        self._init_tables()
        self._parse_event(lines)

    def __str__(self):
        event_n =self.event_information['event_n'] 
        cross_sec = self.cross_section['cross_section(pb)']
        return f"Event; event number={event_n}, cross section={cross_sec}, number particles={self.n_particles}"

    def __repr__(self):
        return self.__str__()

    def _init_tables(self):
        self.n_particles = 0
        self.n_vertices = 0
        self._new_tables()

    def _new_tables(self):
        self.colourFlow = [None for _ in range(self.n_particles)]
        self.antiColourFlow = [None for _ in range(self.n_particles)]


    def assign_heritage(self):
        self.daughters = []
        self.mothers = []
        vertexCodes = [v.hepmc_barcode for v in self.vertex_list]
        for i, (start, end) in enumerate(zip(self.particles.start_vertex_barcodes,
                                             self.particles.end_vertex_barcodes)):
            if start in vertexCodes:
                ms = list(np.nonzero(self.particles.end_vertex_barcodes == start)[0])
            else:
                print("This shouldn't happend... all particles sit below a vertex")
                ms = []
            self.mothers.append(ms)
            if end in vertexCodes:
                ds = list(np.nonzero(self.particles.start_vertex_barcodes == end)[0])
            else:
                ds = []
            self.daughters.append(ds)
            assert len(ms) + len(ds) > 0, f"Particle {i} isolated"
        # now particles that are the beam particles are their own mothers
        b1_code = self.event_information["barcode_beam_particle1"]
        b2_code = self.event_information["barcode_beam_particle2"]
        b1_idx = np.where(self.particles.hepmc_barcode == b1_code)[0]
        b2_idx = np.where(self.particles.hepmc_barcode == b2_code)[0]
        assert self.mothers[b1_idx] == [b1_code]
        assert self.mothers[b2_idx] == [b2_code]
        # so tidy this up
        self.mothers[b1_idx] = []
        self.daughters[b1_idx].remove(b1_code)
        self.mothers[b2_idx] = []
        self.daughters[b2_idx].remove(b2_code)
        unused_barcode = max(self.particles.hepmc_barcode) + 1
        #self.intParticles[b1, self.intParticle_columns.index("start_vertex_barcode")] = unused_barcode
        #self.intParticles[b2, self.intParticle_columns.index("start_vertex_barcode")] = unused_barcode
        

    def check_colour_flow(self):
        exempt_particles = [np.nonzero(self.particles.hepmc_barcode
                                      == self.event_information["barcode_beam_particle1"]),
                            np.nonzero(self.particles.hepmc_barcode
                                      == self.event_information["barcode_beam_particle2"])]
        for vertex in self.vertex_list:
            # find the list of particles bearing this barcode
            in_indices = np.nonzero(self.particles.start_vertex_barcodes == vertex.hepmc_barcode)[0]
            out_indices = np.nonzero(self.particles.end_vertex_barcodes == vertex.hepmc_barcode)[0]
            # tally the colour flows in and out
            link_counter = Counter()
            for in_index in in_indices:
                link_counter[self.colourFlow[in_index]] += 1
                link_counter[self.antiColourFlow[in_index]] -= 1
            for out_index in out_indices:
                link_counter[self.colourFlow[out_index]] -= 1
                link_counter[self.antiColourFlow[out_index]] += 1
            del link_counter[None]
            for flow in link_counter:
                assert link_counter[flow] % 2 == 0, f"Vertex barcode {vertex.hepmc_barcode}" + \
                                                    f"has {link_counter[flow]} outgoing colour" +\
                                                    f"flows for colour flow {flow}"


    @property
    def IDs(self):
        return self.particles.global_ids

    @property
    def MCPIDs(self):
        return self.particles.pids

    @property
    def energy(self):
        return self.particles.es

    @property
    def mass(self):
        return self.particles.ms

    @property
    def px(self):
        return self.particles.pxs

    @property
    def py(self):
        return self.particles.pys

    @property
    def pz(self):
        return self.particles.pzs

    def _parse_event(self, event_lines):
        # now we have a set of lines of the desired event broken into elements
        e_line=[];n_line=[]; u_line=[]; c_line=[]; h_line=[]; f_line=[]
        # they each have their own start key
        start_lines = ['E', 'N', 'U', 'C', 'H', 'F']
        # the first one will be the event line
        assert event_lines[0][0] == 'E'
        e_line = event_lines.pop(0)
        i = 1
        self.event_information["event_n"] = int(e_line[i])
        i += 1
        self.event_information["n_multi_particle_inter"] = int(e_line[i])
        i += 1
        self.event_information["event_scale"] = float(e_line[i])
        i += 1
        self.event_information["alpha_QCD"] = float(e_line[i])
        i += 1
        self.event_information["alpha_QED"] = float(e_line[i])
        i += 1
        self.event_information["signal_process_id"] = int(e_line[i])
        i += 1
        self.event_information["barcode_for_signal_process"] = int(e_line[i])
        i += 1
        self.event_information["n_vertices_in_event"] = int(e_line[i])
        i += 1
        self.event_information["barcode_beam_particle1"] = int(e_line[i])
        i += 1
        self.event_information["barcode_beam_particle2"] = int(e_line[i])
        i += 1
        self.event_information["len_random_state_list"] = int(e_line[i])
        i += 1
        self.event_information["random_state_ints"] = [float(x) for x in
                e_line[i:i+self.event_information["len_random_state_list"]]]
        i += self.event_information["len_random_state_list"]
        self.event_information["len_weight_list"] = int(e_line[i])
        i += 1
        self.event_information["weight_list"] = [float(x) for x in
                e_line[i:i+self.event_information["len_weight_list"]]]
        # the next few lines might be any of N, U C, H or F
        while(event_lines[0][0] in start_lines):
            key = event_lines[0][0]
            if key == 'E':
                raise ValueError("Two event lines found")
            elif key == 'N':
                n_line = event_lines.pop(0)
            elif key == 'U':
                u_line =  event_lines.pop(0)
            elif key == 'C':
                c_line = event_lines.pop(0)
            elif key == 'H':
                h_line = event_lines.pop(0)
            elif key == 'F':
                f_line = event_lines.pop(0)
        # now the next line should be a vertex or particle
        assert event_lines[0][0] in ['V', 'P']
        i = 1 # start from 1 because the first item is the key
        self.weight_names["n_entries"] = int(n_line[i])
        i += 1
        self.weight_names["weight_names"] = n_line[i:]
        i = 1
        self.units["momentum"] = u_line[i]
        i += 1
        self.units["length"] = u_line[i]
        i = 1
        self.cross_section["cross_section(pb)"] = float(c_line[i])
        i += 1
        self.cross_section["error(pb)"] = float(c_line[i])
        # will nned to add the H and f lines when I get round to it
        # now everythng should eb particles and vertices
        n_particles = 0
        n_vertices = 0
        for line in event_lines:
            n_particles += line[0] == 'P'
            n_vertices += line[0] == 'V'
        assert n_particles + n_vertices == len(event_lines)
        # not the number of events is know we can set up the tables
        self.n_particles = n_particles
        self.n_vertices = n_vertices
        self._new_tables()
        # now go through and set things up
        # for speed reasons get lists of indices ahead of time
        particle_list = []
        self.vertex_list = []
        # vertex
        v_file_columns = ["V", "barcode", "id", "x", "y", "z", "ctau",
                          "n_orphans", "n_out", "n_weights"]
        v_file_dict = {name : v_file_columns.index(name) for name in v_file_columns}
        v_barcode_index = v_file_columns.index("barcode")
        #particle
        p_file_columns = ["P", "barcode", "MCPID", "px", "py", "pz", "energy", "generated_mass",
                          "status_code", "polarization_theta", "polarization_phi",
                          "end_vertex_barcode", "n_flow_codes"]
        p_file_dict = {name : p_file_columns.index(name) for name in p_file_columns}
        # at the end of each particle row there are the colour flow indices
        p_colour_start = len(p_file_columns)
        last_vertex_barcode = None
        for line in event_lines:
            if line[0] == 'V':  # new vertex
                last_vertex_barcode = int(line[v_barcode_index])
                this_vertex = Components.MyVertex(float(line[v_file_dict['x']]), float(line[v_file_dict['y']]), float(line[v_file_dict['z']]),
                                                  float(line[v_file_dict['ctau']]), hepmc_barcode=int(line[v_file_dict['barcode']]),
                                                  global_id=len(self.vertex_list), n_out=int(line[v_file_dict['n_out']]),
                                                  n_orphans=int(line[v_file_dict['n_orphans']]), n_weights=int(line[v_file_dict['n_weights']]))
                self.vertex_list.append(this_vertex)
            elif line[0] == 'P':
                particle_reached = len(particle_list)
                this_particle = Components.MyParticle(float(line[p_file_dict['px']]), float(line[p_file_dict['py']]),
                                                      float(line[p_file_dict['pz']]), float(line[p_file_dict['energy']]),
                                                      pid=int(line[p_file_dict['MCPID']]), hepmc_barcode=int(line[p_file_dict['barcode']]),
                                                      global_id=particle_reached,
                                                      start_vertex_barcode=last_vertex_barcode,
                                                      end_vertex_barcode=int(line[p_file_dict['end_vertex_barcode']]),
                                                      status=int(line[p_file_dict['status_code']]),
                                                      generated_mass=float(line[p_file_dict['generated_mass']]))

                particle_list.append(this_particle)
                if len(line) > p_colour_start:
                    colour_pairs = zip(line[p_colour_start::2], line[p_colour_start+1::2])
                    for code_index, colour_code in colour_pairs:
                        if code_index == 1:
                            self.colourFlow[particle_reached] = colour_code
                        elif code_index == 2:
                            self.antiColourFlow[particle_reached] = colour_code
        # finally, but all particles in a collection object
        self.particles = Components.ParticleCollection(particle_list)


def read_file(filepath, start=0, stop=np.inf):
    assert os.path.exists(filepath), f"Can't see that file; {filepath}"
    with open(filepath, 'r') as this_file:
        csv_reader = csv.reader(this_file, delimiter=' ', quotechar='"')
        event_reached = 0
        # move to the start of the start event
        for line in csv_reader:
            if len(line) == 0:
                continue
            if line[0] == "E":
                event_reached += 1
                if event_reached >= start:
                    event_lines =[line]
                    break
        assert len(event_lines) == 1, "Diddnt find any events!"
        # continue till the stop event
        events = []
        for line in csv_reader:
            if len(line) == 0:
                continue
            if line[0] == "E":
                event = Hepmc_event(filepath, event_lines)
                events.append(event)
                event_reached += 1
                event_lines = [line]
            else:
                event_lines.append(line)
            if event_reached > stop:
                break
    return events

