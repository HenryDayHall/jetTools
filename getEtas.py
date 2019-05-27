# coding: utf-8
get_ipython().run_line_magic('run', 'ReadHepmc.py')
events = read_file(fname, 0, 100)
fname = "../../workWithBilly/tag_1_pythia8_events.hepmc"
events = read_file(fname, 0, 100)
get_ipython().run_line_magic('run', 'Components.py')
observables = [Observables(e.particles) for e in events]
all_etas = np.hstack((o.etas for o in observables))
all_etas.shape
