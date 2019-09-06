""" Marry a HepMC and root eventwise object """
from ipdb import set_trace as st
from tree_tagger import Components, ReadHepmc
import os
import numpy as np
import awkward

def marry(hepmc, root_particles):
    if isinstance(root_particles, str):
        root_particles = Components.RootReadout(*os.path.split(root_particles),
                                                ['Particle', 'Track', 'Tower'])
    if isinstance(hepmc, str):
        hepmc = ReadHepmc.Hepmc.from_file(hepmc)
    # first we assert that they both contain the same number of events
    n_events = len(root_particles.PID)
    assert n_events == len(hepmc.MCPID), "Files cotain doferent number of events"
    # this being extablished we compare them eventwise
                     #  (hepmc name, root name)
    precicely_equivalent = [('MCPID', 'PID'),
                            ('Status_code', 'Status')]
    close_equivalent = [('Px', 'Px'),
                        ('Py', 'Py'),
                        ('Pz', 'Pz'),
                        ('Energy', 'Energy'),
                        ('Generated_mass', 'Mass')]
    for event_n in range(n_events):
        # the particles are expected to have the same order in both files
        for hepmc_name, root_name in precicely_equivalent:
            assert np.all(root_particles.__getattr__(root_name)[event_n]
                          == hepmc.__getattr__(hepmc_name)[event_n]), \
                                  f"{root_name} in root file not equal to {hepmc_name}"+\
                                  " in hepmc file"
        for hepmc_name, root_name in close_equivalent:
            np.testing.assert_allclose(root_particles.__getattr__(root_name)[event_n],
                                       hepmc.__getattr__(hepmc_name)[event_n],
                                       err_msg=f"{root_name} in root file not close" +
                                       "to {hepmc_name} in hepmc file")
    # we will keep all of the root columns, but only a selection of the hepmc columns
    columns = root_particles.columns
    contents = root_particles._column_contents
    per_event_hepmc_cols = hepmc.event_information_cols + hepmc.weight_cols + \
                           hepmc.units_cols + hepmc.cross_section_cols
    columns += sorted(per_event_hepmc_cols)
    for name in per_event_hepmc_cols:
        contents[name] = hepmc.__getattr__(name)
    # some get renamed
    per_vertex_hepmc_cols = {'Vertex_barcode': 'Vertex_barcode',
                             'X': 'Vertex_X',
                             'Y': 'Vertex_Y',
                             'Z': 'Vertex_Z',
                             'Ctau': 'Vertex_Ctau'}
    columns += sorted(per_vertex_hepmc_cols.values())
    for name, new_name in per_vertex_hepmc_cols.items():
        contents[new_name] = hepmc.__getattr__(name)
    per_particle_hepmc_cols = ['End_vertex_barcode', 'Start_vertex_barcode',
                               'Parents', 'Children', 'Is_root', 'Is_leaf']
    columns += sorted(per_particle_hepmc_cols)
    for name in per_particle_hepmc_cols:
        contents[name] = hepmc.__getattr__(name)
    # record what has what level of granularity
    contents['per_event'] = awkward.fromiter(per_event_hepmc_cols)
    contents['per_vertex'] = awkward.fromiter(per_vertex_hepmc_cols.values())
    contents['per_particle'] = awkward.fromiter(root_particles.columns + per_particle_hepmc_cols)
    # make the new object and save it
    save_name = root_particles.save_name.split('.', 1)[0] + '_particles.awkd'
    dir_name = root_particles.dir_name
    try:
        new_eventWise = Components.EventWise(dir_name, save_name, columns, contents)
        new_eventWise.write()
    except Exception:
        st()
        new_eventWise = Components.EventWise(dir_name, save_name, columns, contents)
        return new_eventWise

