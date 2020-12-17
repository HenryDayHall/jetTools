""" Marry a HepMC and root eventwise object """
#from ipdb import set_trace as st
from jet_tools.tree_tagger import Components, ReadHepmc
import os
import numpy as np
import awkward

def marry(hepmc, root_particles):
    """
    Combine the information in a hepmc file and a root file
    in one eventWise, to have richer information about the particles

    Parameters
    ----------
    hepmc : str or EventWise
        if a str should be the path to the hepmc file on disk
        if an EventWise should be a dataset with the
        hepmc event data
    root_particles : str or EventWise
        if a str should be the path to the root file on disk
        if an EventWise should be a dataset with the
        root event data

    Returns
    -------
    new_eventWise : EventWise
        dataset containing matched data from teh root file and
        the hepmc file.
    
    """
    if isinstance(root_particles, str):
        root_particles = Components.RootReadout(root_particles,
                                                ['Particle', 'Track', 'Tower'])
    if isinstance(hepmc, str):
        if hepmc.endswith('awkd'):
            hepmc = ReadHepmc.Hepmc.from_file(hepmc)
        else:
            hepmc = ReadHepmc.Hepmc(hepmc)
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
        hepmc.selected_index = event_n
        root_particles.selected_index = event_n
        # the particles are expected to have the same order in both files
        for hepmc_name, root_name in precicely_equivalent:
            assert np.all(root_particles.__getattr__(root_name)
                          == hepmc.__getattr__(hepmc_name)), \
                                  f"{root_name} in root file not equal to {hepmc_name}"+\
                                  " in hepmc file"
        for hepmc_name, root_name in close_equivalent:
            np.testing.assert_allclose(root_particles.__getattr__(root_name),
                                       hepmc.__getattr__(hepmc_name),
                                       err_msg=f"{root_name} in root file not close" +
                                       "to {hepmc_name} in hepmc file")
    # remove all selected indices
    hepmc.selected_index = None
    root_particles.selected_index = None
    # we will keep all of the root columns, but only a selection of the hepmc columns
    # ensure a copy is made
    columns = [name for name in root_particles.columns]
    # forcefully read in here
    contents = {key: getattr(root_particles, key) for key in columns}
    per_event_hepmc_cols = hepmc.event_information_cols + hepmc.weight_cols + \
                           hepmc.units_cols + hepmc.cross_section_cols
    columns += sorted(per_event_hepmc_cols)
    for name in per_event_hepmc_cols:
        contents[name] = getattr(hepmc, name)
    # some get renamed
    per_vertex_hepmc_cols = {'Vertex_barcode': 'Vertex_barcode',
                             'X': 'Vertex_X',
                             'Y': 'Vertex_Y',
                             'Z': 'Vertex_Z',
                             'Ctau': 'Vertex_Ctau'}
    columns += sorted(per_vertex_hepmc_cols.values())
    for name, new_name in per_vertex_hepmc_cols.items():
        contents[new_name] = getattr(hepmc, name)
    per_particle_hepmc_cols = ['End_vertex_barcode', 'Start_vertex_barcode',
                               'Parents', 'Children', 'Is_root', 'Is_leaf']
    columns += sorted(per_particle_hepmc_cols)
    for name in per_particle_hepmc_cols:
        contents[name] = hepmc.__getattr__(name)
    # firgure out which of the root columns are per particle and which are per event
    per_particle_root_cols = []
    per_event_root_cols = []
    for name in root_particles.columns:
        values = getattr(root_particles, name)
        _, depth = Components.detect_depth(values[:])
        if depth == 0:
            per_event_root_cols.append(name)
        else:
            per_particle_root_cols.append(name)
    # record what has what level of granularity
    contents['per_event'] = awkward.fromiter(per_event_root_cols + per_event_hepmc_cols)
    contents['per_vertex'] = awkward.fromiter(per_vertex_hepmc_cols.values())
    contents['per_particle'] = awkward.fromiter(per_particle_root_cols + per_particle_hepmc_cols)
    # make the new object and save it
    save_name = root_particles.save_name.split('.', 1)[0] + '_particles.awkd'
    dir_name = root_particles.dir_name
    path_name = os.path.join(dir_name, save_name)
    new_eventWise = Components.EventWise(path_name, columns, contents)
    new_eventWise.write()
    return new_eventWise

