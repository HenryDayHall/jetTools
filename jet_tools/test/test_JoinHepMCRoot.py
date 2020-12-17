import numpy as np
from numpy import testing as tst
from jet_tools.tree_tagger import Components, JoinHepMCRoot, ReadHepmc
from jet_tools.test.tools import TempTestDir, generic_equality_comp, data_dir
import os
from ipdb import set_trace as st

def test_marry():
    # get a root file
    root_file = os.path.join(data_dir, "mini.root")
    components = ["Particle", "Track", "Tower"]
    rootreadout = Components.RootReadout(root_file, components)
    # decide how many events
    n_events = min(len(rootreadout.Energy), 4)

    with TempTestDir("marry") as dir_name:
        # create a cut down verson of the root file
        contents = {name: getattr(rootreadout, name)[:n_events] for name in rootreadout.columns}
        columns = list(contents.keys())
        restricted = Components.EventWise(os.path.join(dir_name, "restricted.awkd"), columns=columns, contents=contents)
        hepmc_file = os.path.join(data_dir, "mini.hepmc")
        hepmc = ReadHepmc.Hepmc(hepmc_file, 0, n_events)
        JoinHepMCRoot.marry(hepmc, restricted)
        married_name = 'restricted_particles.awkd'
        married = Components.EventWise.from_file(os.path.join(dir_name, married_name))
        # check that the married file contains the columns of the components
        for column in married.columns:
            values = getattr(married, column)
            assert len(values) == n_events
        # check that the number of particles in each event is consistent across all columns
        per_particle_cols = married._column_contents['per_particle']
        values0 = getattr(married, per_particle_cols[0])
        nparticles_col0 = np.array([len(event) for event in values0])
        for column in per_particle_cols[1:]:
            if  column.split('_', 1)[0] in ("Tower", "Track", "Vertex", "Weight"):
                continue  #  these do have diferent frequencies
            if column == 'Random_state_ints':
                continue
            values = getattr(married, column)
            nparticles_here = [len(event) for event in values]
            tst.assert_allclose(nparticles_col0, nparticles_here, err_msg=f"{per_particle_cols[0]} and {column} cound diferent particles")



