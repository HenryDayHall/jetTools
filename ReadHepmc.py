import numpy as np
import csv
from ipdb import set_trace as st


class Hepmc_event:
    def __init__(self, filepath=None):
        self.filepath=filepath
        self._init_columns()
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

    def _init_tables(self):
        self.n_particles = 0
        self.n_vertices = 0
        self._new_tables()

    def _new_tables(self):
        self.intVertices = np.full((self.n_vertices, len(self.intVertex_columns)),
                                   np.nan, dtype=int)
        self.floatVertices = np.full((self.n_vertices, len(self.floatVertex_columns)),
                                     np.nan, dtype=float)
        self.intParticles = np.full((self.n_particles, len(self.intParticle_columns)),
                                   np.nan, dtype=int)
        self.floatParticles = np.full((self.n_particles, len(self.floatParticle_columns)),
                                     np.nan, dtype=float)
        self.colourFlow = [[] for _ in range(self.n_particles)]

    def _init_columns(self):
        self.intVertex_columns = ["barcode", "id", "n_orphans", "n_out", "n_weights"]
        self.floatVertex_columns = ["x", "y", "z", "ctau"]
        self.intParticle_columns = ["barcode", "MCPID", "status_code",
                                    "end_vertex_barcode", "start_vertex_barcode",
                                    "n_flow_codes"]
        self.floatParticle_columns = ["px", "py", "pz", "energy", "generated_mass",
                                      "polarization_theta", "polarization_phi"]


    def assign_heritage(self):
        self.daughters = []
        self.mothers = []
        startCol = self.intParticles[:, self.intParticle_columns.index("start_vertex_barcode")]
        endCol = self.intParticles[:, self.intParticle_columns.index("end_vertex_barcode")]
        vertexCodes = self.intVertices[:, self.intVertex_columns.index("barcode")]
        for i, (start, end) in enumerate(zip(startCol,endCol)):
            if start in vertexCodes:
                ms = list(np.nonzero(endCol == start)[0])
            else:
                print("This shouldn't happend... all particles sit below a vertex")
                ms = []
            self.mothers.append(ms)
            if end in vertexCodes:
                ds = list(np.nonzero(startCol == end)[0])
            else:
                ds = []
            self.daughters.append(ds)
            assert len(ms) + len(ds) > 0, f"Particle {i} isolated"
        # now particles that are the beam particles are their own mothers
        barcodes = list(self.intParticles[:, self.intParticle_columns.index("barcode")])
        b1 = barcodes.index(self.event_information["barcode_beam_particle1"])
        b2 = barcodes.index(self.event_information["barcode_beam_particle2"])
        assert self.mothers[b1] == [b1]
        assert self.mothers[b2] == [b2]
        # so tidy this up
        self.mothers[b1] = []
        self.daughters[b1].remove(b1)
        self.mothers[b2] = []
        self.daughters[b2].remove(b2)
        unused_barcode = max(barcodes) + 1
        self.intParticles[b1, self.intParticle_columns.index("start_vertex_barcode")] = unused_barcode
        self.intParticles[b2, self.intParticle_columns.index("start_vertex_barcode")] = unused_barcode
        


    def read_file(self, filepath=None, event_n=0):
        if filepath is not None:
            self.filepath = filepath
        else:
            assert self.filepath is not None
        with open(filepath, 'r') as this_file:
            event_lines = []
            event_reached = -1
            csv_reader = csv.reader(this_file, delimiter=' ', quotechar='"')
            for line in csv_reader:
                if len(line) == 0:
                    continue
                event_reached += line[0] == "E"
                if event_reached > event_n:
                    break
                elif event_reached == event_n:
                    event_lines.append(line)
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
                e_line = event_lines.pop(0)
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
        particle_reached = 0
        vertex_reached = 0
        # vertex
        v_file_columns = ["V", "barcode", "id", "x", "y", "z", "ctau",
                          "n_orphans", "n_out", "n_weights"]
        v_int_file_indices = [v_file_columns.index(name) for name in self.intVertex_columns
                              if name in v_file_columns]
        v_float_file_indices = [v_file_columns.index(name) for name in self.floatVertex_columns
                                if name in v_file_columns]
        v_int_table_indices = [self.intVertex_columns.index(name) for name in v_file_columns
                                if name in self.intVertex_columns]
        v_float_table_indices = [self.floatVertex_columns.index(name) for name in v_file_columns
                                if name in self.floatVertex_columns]
        v_barcode_index = v_file_columns.index("barcode")
        #particle
        p_file_columns = ["P", "barcode", "MCPID", "px", "py", "pz", "energy", "generated_mass",
                          "status_code", "polarization_theta", "polarization_phi",
                          "end_vertex_barcode", "n_flow_codes"]
        p_int_file_indices = [p_file_columns.index(name) for name in self.intParticle_columns
                              if name in p_file_columns]
        p_float_file_indices = [p_file_columns.index(name) for name in self.floatParticle_columns
                                if name in p_file_columns]
        p_int_table_indices = [self.intParticle_columns.index(name) for name in p_file_columns
                                if name in self.intParticle_columns]
        p_float_table_indices = [self.floatParticle_columns.index(name) for name in p_file_columns
                                if name in self.floatParticle_columns]
        p_barcode_index = self.intParticle_columns.index("start_vertex_barcode")
        # at the end of each particle row there are the colour flow indices
        p_colour_start = max(p_int_file_indices + p_float_file_indices) + 1
        last_vertex_barcode = None
        for line in event_lines:
            if line[0] == 'V':
                last_vertex_barcode = int(line[v_barcode_index])
                self.intVertices[vertex_reached, v_int_table_indices] = [int(line[i]) for i in v_int_file_indices]
                self.floatVertices[vertex_reached, v_float_table_indices] = [float(line[i]) for i in v_float_file_indices]
                vertex_reached += 1
            elif line[0] == 'P':
                self.intParticles[particle_reached, p_barcode_index] = last_vertex_barcode
                self.intParticles[particle_reached, p_int_table_indices] = [int(line[i]) for i in p_int_file_indices]
                self.floatParticles[particle_reached, p_float_table_indices] = [float(line[i]) for i in p_float_file_indices]
                self.colourFlow[particle_reached] = [int(x) for x in line[p_colour_start:]]
                particle_reached += 1



