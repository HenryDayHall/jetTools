""" A module to test the TrueTag module """
from ipdb import set_trace as st
import numpy as np
from tree_tagger import Components, FormJets, TrueTag
import awkward
from tools import TempTestDir
import numpy.testing as tst


def test_allocate():
    # try an empty event
    params = {}
    jet_name = "Jet"
    params['Jet_InputIdx'] = []
    params['Jet_RootInputIdx'] = []
    params['Jet_Phi'] = []
    params['Jet_Rapidity'] = []
    params['Phi'] = []
    params['Rapidity'] = []
    params = {key: [awkward.fromiter(v)] for key, v in params.items()}
    tag_idx = []
    valid_jets = []
    expected = np.array([]).reshape((0,0))
    with TempTestDir("tst") as dir_name:
        eventWise = Components.EventWise(dir_name, "tmp.awkd")
        eventWise.append(**params)
        eventWise.selected_index = 0
        closest = TrueTag.allocate(eventWise, jet_name, tag_idx, 0.1)
        tst.assert_allclose(closest, expected)
    # try a regular event
    params = {}
    jet_name = "Jet"
    params['Jet_InputIdx'] = [[0, 1, 2], [3, 4, 5]]
    params['Jet_RootInputIdx'] = [[0], [4]]
    params['Jet_Phi'] = [[0., 1., 1.], [1., np.pi, 1.]]
    params['Jet_Rapidity'] = [[0., 1., 1.], [1., 2., 1.]]
    params['Phi'] = [4., 0., -np.pi, -1., np.pi, 2.]
    params['Rapidity'] = [0., 0., -2., 10., 2., 0.]
    params = {key: [awkward.fromiter(v)] for key, v in params.items()}
    tag_idx = [1, 4]
    valid_jets = np.array([0, 1])
    expected = [0, 1]
    with TempTestDir("tst") as dir_name:
        eventWise = Components.EventWise(dir_name, "tmp.awkd")
        eventWise.append(**params)
        eventWise.selected_index = 0
        closest = TrueTag.allocate(eventWise, jet_name, tag_idx, 0.1, valid_jets)
        tst.assert_allclose(closest, expected)
    # try an event where one tag gets cut off by the max angle
    params = {}
    jet_name = "Jet"
    params['Jet_InputIdx'] = [[0], [1]]
    params['Jet_RootInputIdx'] = [[0], [1]]
    params['Jet_Phi'] = [[0.], [1.]]
    params['Jet_Rapidity'] = [[0.], [1.]]
    params['Phi'] = [4., 0., -np.pi, 1., np.pi, 2.]
    params['Rapidity'] = [0., 0., -2., 0., 2., 0.]
    params = {key: [awkward.fromiter(v)] for key, v in params.items()}
    tag_idx = [1, 3]
    valid_jets = None
    expected = [0, -1]
    with TempTestDir("tst") as dir_name:
        eventWise = Components.EventWise(dir_name, "tmp.awkd")
        eventWise.append(**params)
        eventWise.selected_index = 0
        closest = TrueTag.allocate(eventWise, jet_name, tag_idx, 0.9, valid_jets)
        tst.assert_allclose(closest, expected)
    # try an event with th closest et declated invalid
    params = {}
    jet_name = "Jet"
    params['Jet_InputIdx'] = [[0], [1]]
    params['Jet_RootInputIdx'] = [[0], [1]]
    params['Jet_Phi'] = [[0.], [1.]]
    params['Jet_Rapidity'] = [[0.], [1.]]
    params['Phi'] = [4., 0., -np.pi, 1., np.pi, 2.]
    params['Rapidity'] = [0., 0., -2., 0., 2., 0.]
    params = {key: [awkward.fromiter(v)] for key, v in params.items()}
    tag_idx = [1, 3]
    valid_jets = np.array([1])
    expected = [1, 1]
    with TempTestDir("tst") as dir_name:
        eventWise = Components.EventWise(dir_name, "tmp.awkd")
        eventWise.append(**params)
        eventWise.selected_index = 0
        closest = TrueTag.allocate(eventWise, jet_name, tag_idx, 2., valid_jets)
        tst.assert_allclose(closest, expected)


def test_tag_particle_indices():
    # will need an eventwise with Parents, Children, MCPID
    # try an empty event
    children = []
    parents =  []
    mcpid =    []
    expected = []
    with TempTestDir("tst") as dir_name:
        eventWise = Components.EventWise(dir_name, "tmp.awkd")
        eventWise.append(Children=[awkward.fromiter(children)],
                         Parents=[awkward.fromiter(parents)],
                         MCPID=[awkward.fromiter(mcpid)])
        eventWise.selected_index = 0
        tag_idx = TrueTag.tag_particle_indices(eventWise, tag_pids=np.array([5]))
        tst.assert_allclose(tag_idx, expected)
    # layer     -1  0       1    1      -1   2   2          3   3   3   -1
    # idx       0   1       2    3       4   5       6          7   8   9   10
    children = [[], [2, 3], [5], [6, 5], [], [],     [7, 8, 9], [], [], [], []]
    parents =  [[], [],     [1], [1],    [], [2, 3], [3],       [6],[6],[6],[]]
    mcpid =    [4,  -5,     5,   3,      2,  1,      -5,        -1, 7,  11, 12]
    expected = [2]
    with TempTestDir("tst") as dir_name:
        eventWise = Components.EventWise(dir_name, "tmp.awkd")
        eventWise.append(Children=[awkward.fromiter(children)],
                         Parents=[awkward.fromiter(parents)],
                         MCPID=[awkward.fromiter(mcpid)])
        eventWise.selected_index = 0
        tag_idx = TrueTag.tag_particle_indices(eventWise, tag_pids=np.array([5]))
        tst.assert_allclose(tag_idx, expected)
    #  a regular event with all the tag pids
    # layer     -1  0       1    1      -1   2   2          3   3   3   -1
    # idx       0   1       2    3       4   5       6          7   8   9   10
    children = [[], [2, 3], [5], [6, 5], [], [],     [7, 8, 9], [], [], [], []]
    parents =  [[], [],     [1], [1],    [], [2, 3], [3],       [6],[6],[6],[]]
    mcpid =    [1,  -5,     5101,10511,  5,  1,      -5,        -1, 7,  11, 12]
    expected = [2, 4, 6]
    with TempTestDir("tst") as dir_name:
        eventWise = Components.EventWise(dir_name, "tmp.awkd")
        eventWise.append(Children=[awkward.fromiter(children)],
                         Parents=[awkward.fromiter(parents)],
                         MCPID=[awkward.fromiter(mcpid)])
        eventWise.selected_index = 0
        tag_idx = TrueTag.tag_particle_indices(eventWise, tag_pids='hadrons')
        tst.assert_allclose(sorted(tag_idx), expected)


def test_add_tag_particles():
    pass


def test_add_tags():
    pass


def test_add_ctags():
    pass


def test_tags_to_quarks():
    pass


def test_percent_pos():
    pass

