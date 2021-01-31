from jet_tools import ReadHepmc, Components
import awkward
##from ipdb import set_trace as st
import os
import numpy as np

format_specifiers = {int: 'd', float: 'e', str: 's'}

def create_particle_line(eventWise, particle_idx):
    # this ends in a list of low codes
    start_key = 'P '
    col_types = list(ReadHepmc.Hepmc.particle_convertions)
    n_particle_cols = len(col_types)
    # some of the particle columns in Hepmc are created additions
    particle_cols = ReadHepmc.Hepmc.particle_cols[:n_particle_cols]
    items = [getattr(eventWise, name)[particle_idx] for name in particle_cols]
    # the last three items are flow codes
    assert particle_cols[-3] == "N_flow_codes"
    assert particle_cols[-2] == "Flow_codes"
    assert particle_cols[-1] == "Antiflow_codes"
    num_flow_codes = items[-3]
    col_types = col_types[:-2] + [col_types[-2]]*num_flow_codes + [col_types[-1]]*num_flow_codes
    items = items[:-2] + [*items[-2], *items[-1]]
    # compose the string format
    line = start_key + ' '.join(["{:" + format_specifiers[t] + "}" for t in col_types])
    line = line.format(*items)
    return line


def create_vertex_line(eventWise, vertex_idx):
    start_key = 'V '
    col_types = list(ReadHepmc.Hepmc.vertex_convertions)
    vertex_cols = ReadHepmc.Hepmc.vertex_cols
    items = [getattr(eventWise, name)[vertex_idx] for name in vertex_cols]
    # the last item is vertex weights, but we don't stor them becuase they never exist
    assert vertex_cols[-1] == "N_vertex_weights"
    assert np.allclose(items[-1], 0)
    # compose the string format
    line = start_key + ' '.join(["{:" + format_specifiers[t] + "}" for t in col_types])
    line = line.format(*items)
    return line


def create_unit_line(eventWise):
    start_key = 'U '
    units_cols = ReadHepmc.Hepmc.units_cols
    items = [getattr(eventWise, name) for name in units_cols]
    # compose the string format
    line = start_key + ' '.join([f"{v}" for v in items])
    return line


def create_event_line(eventWise):
    start_key = 'E'
    col_types = list(ReadHepmc.Hepmc.event_information_conversions)
    event_information_cols = ReadHepmc.Hepmc.event_information_cols
    items = [getattr(eventWise, name) for name in event_information_cols]
    multi_item_pairs = {"Random_state_ints": "Len_random_state_list",
                        "Weight_list": "Len_weight_list"}
    line = start_key
    for i, (name, converter) in enumerate(zip(event_information_cols, col_types)):
        fmt = " {:" + format_specifiers[converter] + "}"
        if name in multi_item_pairs:
            len_col = event_information_cols.index(multi_item_pairs[name])
            assert len(items[i]) == int(items[len_col])
            line += ''.join([fmt.format(x) for x in items[i]])
        else:
            line += fmt.format(items[i])
    return line


def create_crosssection_line(eventWise):
    start_key = 'E '
    cross_section_cols = ReadHepmc.Hepmc.cross_section_cols
    items = [getattr(eventWise, name) for name in cross_section_cols]
    fmt = "{:" + format_specifiers[float] + "}"
    line = start_key + ' '.join([fmt.format(x) for x in items])
    return line


def create_weight_line(eventWise):
    # this ends in a list of low codes
    start_key = 'N '
    # some of the particle columns in Hepmc are created additions
    weight_cols = ReadHepmc.Hepmc.weight_cols
    items = [getattr(eventWise, name) for name in weight_cols]
    assert weight_cols[0] == "N_weight_names"
    assert weight_cols[1] == "Weight_names"
    items = [items[0], *items[1].tolist()]
    col_types = [int] + [str]*(len(items) - 1)
    # compose the string format
    line = start_key + ' '.join(["{:" + format_specifiers[t] + "}" for t in col_types])
    line = line.format(*items)
    return line


def write_Hepmc(eventWise, save_path):
    header = [create_event_line, create_weight_line, create_unit_line, create_crosssection_line]
    with open(save_path, 'w') as save_file:
        # open the block
        save_file.write("HepMC::IO_GenEvent-START_EVENT_LISTING")
        eventWise.selected_index = None
        n_events = len(eventWise.Vertex_barcode)
        lines = []
        for event_n in range(n_events):
            if (event_n+1) % 100 == 0:
                print(f"{event_n/n_events:.2%}", end='\r', flush = True)
                # periodic writing for speed
                save_file.write(os.linesep + os.linesep.join(lines))
                lines = []
            eventWise.selected_index = event_n
            lines += [line_creator(eventWise) for line_creator in header]
            # now go through adding each vertex then the particles the have come out of it
            particles_found = 0
            for vertex_n, barcode in enumerate(eventWise.Vertex_barcode):
                lines.append(create_vertex_line(eventWise, vertex_n))
                children = np.where(eventWise.Start_vertex_barcode == barcode)[0]
                lines += [create_particle_line(eventWise, particle_n) for 
                          particle_n in children]
                particles_found += len(children)
            # check we found all the particles
            assert particles_found == len(eventWise.Start_vertex_barcode)
        # finished events
        save_file.write(os.linesep + os.linesep.join(lines))
        # close the block
        save_file.write(os.linesep + "HepMC::IO_GenEvent-END_EVENT_LISTING")
                

def zip_eventWises(main_eventWise, inserted_eventWise, same_beamparticles=False):
    main_eventWise.selected_index = None
    inserted_eventWise.selected_index = None
    n_events = len(main_eventWise.Vertex_barcode)
    if n_events != len(inserted_eventWise.Vertex_barcode):
        print("Warning eventWise objects differ in length")
    n_events = min(len(main_eventWise.Vertex_barcode), len(inserted_eventWise.Vertex_barcode))
    # we need to change the particle and vertex barcodes in the insert
    free_vertex_barcode = awkward.fromiter([np.min(event)-1 for event in
                                            main_eventWise.Vertex_barcode])
    new_vertex_barcode = inserted_eventWise.Vertex_barcode - free_vertex_barcode
    new_end_barcode = inserted_eventWise.End_vertex_barcode - free_vertex_barcode
    new_start_barcode = inserted_eventWise.Start_vertex_barcode - free_vertex_barcode
    free_particle_barcode = awkward.fromiter([np.max(event)+1 for event in
                                              main_eventWise.Particle_barcode])
    new_particle_barcode = inserted_eventWise.Particle_barcode + free_particle_barcode
    inserted_eventWise.append(Vertex_barcode = new_vertex_barcode,
                              End_vertex_barcode = new_end_barcode,
                              Start_vertex_barcode = new_start_barcode,
                              Particle_barcode = new_particle_barcode)
    # for ram reasons go throught the columns in small sets
    columns = set(main_eventWise.columns).intersection(inserted_eventWise.columns)
    new_content = {}
    for i, column in enumerate(columns):
        main = getattr(main_eventWise, column)
        insert = getattr(inserted_eventWise, column)
        new_column = [a.tolist() + b.tolist() for a, b in zip(main, insert)]
        new_content[column] = awkward.fromiter(new_column)
    main_eventWise.append(**new_content)
        






    


def merge_Hepmc(path1, path2, out_path=None,
                same_beamparticles=False, batch_len=None):
    pass


if __name__ == '__main__':
    path = "megaIgnore/join/tag_1_pythia8_events_pileup.hepmc"
    hepmc = ReadHepmc.Hepmc(*os.path.split(path), 0, 10)
    new_path = path + ".copy"
    write_Hepmc(hepmc, new_path)
