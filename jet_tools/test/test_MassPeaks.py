""" A module to test the code in MassPeaks.py """
import pytest
from ipdb import set_trace as st
import numpy as np
import numpy.testing as tst
from test.tools import TempTestDir
import awkward
from tree_tagger import MassPeaks, Components, Constants

# WARNING; this was not written with consistent angular/linear momentums
# if tests her start breaking, it could be that the functions are fine,
# just need to match th elinear and angular measures in the tests

def test_order_tagged_jets():
    # need Tags, InputIdx and RootInputIdx, and a ranking_variable, by default PT
    # try an empty event
    params = {}
    jet_name = "Jet"
    params['Jet_InputIdx'] = []
    params['Jet_RootInputIdx'] = []
    params['Jet_Tags'] = []
    params['Jet_PT'] = []
    params['Jet_Child1'] = []
    params = {key: [awkward.fromiter(v)] for key, v in params.items()}
    with TempTestDir("tst") as dir_name:
        eventWise = Components.EventWise(dir_name, "tmp.awkd")
        eventWise.append(**params)
        eventWise.selected_index = 0
        filtered_event_idxs = []
        found = MassPeaks.order_tagged_jets(eventWise, jet_name, filtered_event_idxs)
        assert len(found) == 0
    # try a regular event
    params = {}
    params['Jet_InputIdx'] = [[0, 1, 2], [3, 4, 5], [7, 8, 9]]
    params['Jet_RootInputIdx'] = [[0], [4], [7]]
    params['Jet_Tags'] = [[0], [1], []]
    params['Jet_PT'] = [[60, 10, 40], [100, 40, 50], [40, 40, 40]]
    params['Jet_Bobble'] = [[-5, 1, 1], [0, 0, -10], [12, 20, 40]]
    params['Jet_Child1'] = [[1, -1, -1], [-1, 5, -1], [-1, -1, 7]]
    params = {key: [awkward.fromiter(v)] for key, v in params.items()}
    with TempTestDir("tst") as dir_name:
        eventWise = Components.EventWise(dir_name, "tmp.awkd")
        eventWise.append(**params)
        eventWise.selected_index = 0
        filtered_event_idxs = [0, 1]
        found = MassPeaks.order_tagged_jets(eventWise, jet_name, filtered_event_idxs)
        tst.assert_allclose(found, [1, 0])
        found = MassPeaks.order_tagged_jets(eventWise, jet_name, filtered_event_idxs, "Bobble")
        tst.assert_allclose(found, [0, 1])

def test_combined_jet_mass():
    # need jet Px, Py, Pz, Energy, InputIdx, RootInputIdx
    # try an empty event
    params = {}
    jet_name = "Jet"
    params['Jet_RootInputIdx'] = []
    params['Jet_InputIdx'] = []
    params['Jet_Px'] = []
    params['Jet_Py'] = []
    params['Jet_Pz'] = []
    params['Jet_Energy'] = []
    params = {key: [awkward.fromiter(v)] for key, v in params.items()}
    with TempTestDir("tst") as dir_name:
        eventWise = Components.EventWise(dir_name, "tmp.awkd")
        eventWise.append(**params)
        eventWise.selected_index = 0
        found = MassPeaks.combined_jet_mass(eventWise, jet_name, [])
        assert found == 0
    # try a regular event
    params = {}
    params['Jet_InputIdx'] = [[0, 1, 2], [3, 4, 5], [7, 8, 9], [10]]
    params['Jet_RootInputIdx'] = [[0], [4], [7], [10]]
    params['Jet_Px'] = [[0, 0, 0], [40, 0, 50], [40, 0, 0], [1]]
    params['Jet_Py'] = [[0, 0, 0], [100, 0, 50], [70, 0, 0], [0]]
    params['Jet_Pz'] = [[0, 0, 0], [100, 0, 50], [1, 0, 0], [0]]
    params['Jet_Energy'] = [[0, 0, 0], [10, 40, 10], [400, 40, 40], [10]]
    params = {key: [awkward.fromiter(v)] for key, v in params.items()}
    with TempTestDir("tst") as dir_name:
        eventWise = Components.EventWise(dir_name, "tmp.awkd")
        eventWise.append(**params)
        eventWise.selected_index = 0
        for i, expected in enumerate([0, 40, np.sqrt(400**2 - 1 - 70**2 - 40**2), np.sqrt(99)]):
            found = MassPeaks.combined_jet_mass(eventWise, jet_name, [i])
            tst.assert_allclose(found, expected)
            found = MassPeaks.combined_jet_mass(eventWise, jet_name, [0, i])
            tst.assert_allclose(found, expected)

    
def test_cluster_mass():
    # need Px, Py, Pz, Energy
    # try an empty event
    params = {}
    params['Px'] = []
    params['Py'] = []
    params['Pz'] = []
    params['Energy'] = []
    params = {key: [awkward.fromiter(v)] for key, v in params.items()}
    with TempTestDir("tst") as dir_name:
        eventWise = Components.EventWise(dir_name, "tmp.awkd")
        eventWise.append(**params)
        eventWise.selected_index = 0
        found = MassPeaks.cluster_mass(eventWise, [])
        assert found == 0
    # try a regular event
    params = {}
    params['Px'] = [0, 0, 40, 1]
    params['Py'] = [0, 0, -70, 0]
    params['Pz'] = [0, 0, 1, 0]
    params['Energy'] = [0, 40, 400, 10]
    params = {key: [awkward.fromiter(v)] for key, v in params.items()}
    with TempTestDir("tst") as dir_name:
        eventWise = Components.EventWise(dir_name, "tmp.awkd")
        eventWise.append(**params)
        eventWise.selected_index = 0
        for i, expected in enumerate([0, 40, np.sqrt(400**2 - 1 - 70**2 - 40**2), np.sqrt(99)]):
            found = MassPeaks.cluster_mass(eventWise, [i])
            tst.assert_allclose(found, expected)
            found = MassPeaks.cluster_mass(eventWise, [0, i])
            tst.assert_allclose(found, expected)
        found = MassPeaks.cluster_mass(eventWise, [0, 1, 2, 3])
        expected = np.sqrt(450**2 - 41**2 - 70**2 - 1)
        tst.assert_allclose(found, expected)


def test_inverse_condensed_indices():
    assert MassPeaks.inverse_condensed_indices(0, 0) is None
    assert set(MassPeaks.inverse_condensed_indices(0, 2)) == {0, 1}
    assert set(MassPeaks.inverse_condensed_indices(3, 4)) == {1, 2}


def test_smallest_angle_parings():
    params = {}
    jet_name = "Jet"
    params['Jet_RootInputIdx'] = []
    params['Jet_InputIdx'] = []
    params['Jet_Phi'] = []
    params['Jet_PT'] = []
    params['Jet_Rapidity'] = []
    params['Jet_Tags'] = []
    params['Jet_Energy'] = []
    params['Jet_Child1'] = []
    params = {key: [awkward.fromiter(v)] for key, v in params.items()}
    with TempTestDir("tst") as dir_name:
        eventWise = Components.EventWise(dir_name, "tmp.awkd")
        eventWise.append(**params)
        eventWise.selected_index = 0
        found = MassPeaks.smallest_angle_parings(eventWise, jet_name, [])
        assert len(found) == 0
    # try a regular event
    params = {}
    params['Jet_InputIdx'] = [[0, 1, 2], [3, 4, 5], [7, 8, 9], [10, 11]]
    params['Jet_RootInputIdx'] = [[0], [4], [7], [10]]
    params['Jet_Phi'] = [[0, 0, 0], [40, 0, 50], [40, 0, 0], [1, 3]]
    params['Jet_PT'] = [[10, 10, 10], [40, 50, 50], [40, 0, 0], [1, 7]]
    params['Jet_Rapidity'] = [[0, 0, 0], [100, 0, 50], [70, 0, 0], [0, 5]]
    params['Jet_Child1'] = [[1, -1, -1], [-1, 5, -1], [-1, -1, 7], [-1, -1]]
    params['Jet_Tags'] = [[0], [], [1], []]
    params = {key: [awkward.fromiter(v)] for key, v in params.items()}
    with TempTestDir("tst") as dir_name:
        eventWise = Components.EventWise(dir_name, "tmp.awkd")
        eventWise.append(**params)
        eventWise.selected_index = 0
        found = MassPeaks.smallest_angle_parings(eventWise, jet_name, [0, 1, 2])
        assert len(found) == 1
        assert set(found[0]) == {0, 2}
        found = MassPeaks.smallest_angle_parings(eventWise, jet_name, [1, 2])
        assert len(found) == 0
    params['Jet_Tags'] = [awkward.fromiter([[0], [5], [1], [7]])]
    with TempTestDir("tst") as dir_name:
        eventWise = Components.EventWise(dir_name, "tmp.awkd")
        eventWise.append(**params)
        eventWise.selected_index = 0
        found = MassPeaks.smallest_angle_parings(eventWise, jet_name, [0, 1, 2, 3])
        assert len(found) == 2
        found = [set(pair) for pair in found]
        assert {0, 1} in found
        assert {2, 3} in found


def test_all_smallest_angles():
    params = {}
    jet_name = "Jet"
    # event 0 is empty - no contribution
    params['Jet_RootInputIdx'] = [[]]
    params['Jet_InputIdx'] = [[]]
    params['Jet_Px'] = [[]]
    params['Jet_Py'] = [[]]
    params['Jet_Pz'] = [[]]
    params['Jet_Energy'] = [[]]
    params['Jet_Phi'] = [[]]
    params['Jet_PT'] = [[]]
    params['Jet_Rapidity'] = [[]]
    params['Jet_Tags'] = [[]]
    params['Jet_Child1'] = [[]]
    params['Jet_Parent'] = [[]]
    # event 1 has just 1 jet - no contribution
    params['Jet_RootInputIdx'] += [[[0]]]
    params['Jet_InputIdx'] += [[[0, 1, 2]]]
    params['Jet_Px'] += [[[1, 1, 1]]]
    params['Jet_Py'] += [[[1, 1, 1]]]
    params['Jet_Pz'] += [[[1, 1, 1]]]
    params['Jet_Energy'] += [[[10, 10, 10]]]
    params['Jet_Phi'] += [[[0, 0, 0]]]
    params['Jet_PT'] += [[[1, 1, 1]]]
    params['Jet_Rapidity'] += [[[1, 1, 1]]]
    params['Jet_Tags'] += [[[1]]]
    params['Jet_Child1'] += [[[1, -1, -1]]]
    params['Jet_Parent'] += [[[-1, 0, 0]]]
    # event 2 has one untagged and one tagged jet - no contribution
    params['Jet_RootInputIdx'] += [[[0], [3]]]
    params['Jet_InputIdx'] += [[[0, 1, 2], [3, 4, 5]]]
    params['Jet_Px'] += [[[1, 1, 1], [1, 1, 1]]]
    params['Jet_Py'] += [[[1, 1, 1], [1, 1, 1]]]
    params['Jet_Pz'] += [[[1, 1, 1], [1, 1, 1]]]
    params['Jet_Energy'] += [[[10, 10, 10], [10, 10, 10]]]
    params['Jet_Phi'] += [[[0, 0, 0], [0, 0, 0]]]
    params['Jet_PT'] += [[[1, 1, 1], [1, 1, 1]]]
    params['Jet_Rapidity'] += [[[1, 1, 1], [1, 1, 1]]]
    params['Jet_Tags'] += [[[1], []]]
    params['Jet_Child1'] += [[[1, -1, -1], [4, -1, -1]]]
    params['Jet_Parent'] += [[[-1, 0, 0], [-1, 3, 3]]]
    prep_params = {key: awkward.fromiter([awkward.fromiter(e) for e in v])for key, v in params.items()}
    # check this results in no masses
    with TempTestDir("tst") as dir_name:
        eventWise = Components.EventWise(dir_name, "tmp.awkd")
        eventWise.append(**prep_params)
        found = MassPeaks.all_smallest_angles(eventWise, jet_name, 0)
        assert len(found) == 0
    # now start adding things that will contribute
    # event 3 has one untagged and two tagged jets - two tagged jets should combine
    params['Jet_RootInputIdx'] += [[[0], [3], [7]]]
    params['Jet_InputIdx'] += [[[0, 1, 2], [3, 4, 5], [6, 7, 8]]]
    params['Jet_Px'] += [[[1, 1, 1], [1, 1, 1], [-1, -1, -1]]]
    params['Jet_Py'] += [[[1, 1, 1], [1, 1, 1], [-1, -1, -1]]]
    params['Jet_Pz'] += [[[1, 1, 1], [1, 1, 1], [-1, -1, -1]]]
    params['Jet_Energy'] += [[[10, 10, 10], [10, 10, 10], [5, 5, 5]]]
    params['Jet_Phi'] += [[[0, 0, 0], [0, 0, 0], [3, 3, 3]]]
    params['Jet_PT'] += [[[1, 1, 1], [1, 1, 1], [1, 1, 1]]]
    params['Jet_Rapidity'] += [[[1, 1, 1], [1, 1, 1], [-1, -1, -1]]]
    params['Jet_Tags'] += [[[1], [], [4]]]
    params['Jet_Child1'] += [[[1, -1, -1], [4, -1, -1], [-1, 8, -1]]]
    params['Jet_Parent'] += [[[-1, 0, 0], [-1, 3, 3], [7, -1, 7]]]
    expected_masses = [15.]
    # event 4 has two untagged and three tagged jets - two closest tagged jets should combine
    params['Jet_RootInputIdx'] += [[[0], [3], [7], [9], [12]]]
    params['Jet_InputIdx'] += [[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14]]]
    params['Jet_Px'] += [[[1, 1, 1], [1, 1, 1], [-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]]
    params['Jet_Py'] += [[[1, 1, 1], [1, 1, 1], [-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]]
    params['Jet_Pz'] += [[[1, 1, 1], [1, 1, 1], [-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]]
    params['Jet_Energy'] += [[[10, 10, 10], [10, 10, 10], [5, 5, 5], [6, 6, 6], [7, 7, 7]]]
    params['Jet_Phi'] += [[[0, 0, 0], [0, 0, 0], [3, 3, 3], [3, 3, 3], [3, 3, 3]]]
    params['Jet_PT'] += [[[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]]
    params['Jet_Rapidity'] += [[[1, 1, 1], [1, 1, 1], [-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]]
    params['Jet_Tags'] += [[[1], [], [4], [10, 11], []]]
    params['Jet_Child1'] += [[[1, -1, -1], [4, -1, -1], [-1, 8, -1], [10, -1, -1], [13, -1, -1]]]
    params['Jet_Parent'] += [[[-1, 0, 0], [-1, 3, 3], [7, -1, 7], [-1, 9, 9], [12, -1, 12]]]
    expected_masses += [np.sqrt(11**2 - 3*4)]
    # event 5 has one untagged and four tagged jets - all tagged jets will combine
    params['Jet_RootInputIdx'] += [[[0], [3], [7], [9], [12]]]
    params['Jet_InputIdx'] += [[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14]]]
    params['Jet_Px'] += [[[1, 1, 1], [1, 1, 1], [-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]]
    params['Jet_Py'] += [[[1, 1, 1], [1, 1, 1], [-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]]
    params['Jet_Pz'] += [[[1, 1, 1], [1, 1, 1], [-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]]
    params['Jet_Energy'] += [[[10, 10, 10], [10, 10, 10], [5, 5, 5], [6, 6, 6], [7, 7, 7]]]
    params['Jet_Phi'] += [[[0, 0, 0], [0, 0, 0], [3, 3, 3], [3, 3, 3], [3, 3, 3]]]
    params['Jet_PT'] += [[[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]]
    params['Jet_Rapidity'] += [[[1, 1, 1], [1, 1, 1], [-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]]
    params['Jet_Tags'] += [[[1], [3], [4], [10, 11], []]]
    params['Jet_Child1'] += [[[1, -1, -1], [4, -1, -1], [-1, 8, -1], [10, -1, -1], [13, -1, -1]]]
    params['Jet_Parent'] += [[[-1, 0, 0], [-1, 3, 3], [7, -1, 7], [-1, 9, 9], [12, -1, 12]]]
    expected_masses += [np.sqrt(20**2 - 3*4), np.sqrt(11**2 - 3*4)]
    prep_params = {key: awkward.fromiter([awkward.fromiter(e) for e in v])for key, v in params.items()}
    # check this results in the predicted masses
    with TempTestDir("tst") as dir_name:
        eventWise = Components.EventWise(dir_name, "tmp.awkd")
        eventWise.append(**prep_params)
        found = MassPeaks.all_smallest_angles(eventWise, jet_name, 0)
        tst.assert_allclose(sorted(found), sorted(expected_masses))


def test_all_jet_masses():
    params = {}
    jet_name = "Jet"
    # event 0 is empty - should be 0
    params['Jet_RootInputIdx'] = [[]]
    params['Jet_InputIdx'] = [[]]
    params['Jet_Px'] = [[]]
    params['Jet_Py'] = [[]]
    params['Jet_Pz'] = [[]]
    params['Jet_Energy'] = [[]]
    params['Jet_Phi'] = [[]]
    params['Jet_PT'] = [[]]
    params['Jet_Rapidity'] = [[]]
    params['Jet_Tags'] = [[]]
    params['Jet_Child1'] = [[]]
    params['Jet_Parent'] = [[]]
    expected_masses = [0.]
    # event 1 has just 1 jet
    params['Jet_RootInputIdx'] += [[[0]]]
    params['Jet_InputIdx'] += [[[0, 1, 2]]]
    params['Jet_Px'] += [[[1, 1, 1]]]
    params['Jet_Py'] += [[[1, 1, 1]]]
    params['Jet_Pz'] += [[[1, 1, 1]]]
    params['Jet_Energy'] += [[[10, 10, 10]]]
    params['Jet_Phi'] += [[[0, 0, 0]]]
    params['Jet_PT'] += [[[1, 1, 1]]]
    params['Jet_Rapidity'] += [[[1, 1, 1]]]
    params['Jet_Tags'] += [[[1]]]
    params['Jet_Child1'] += [[[1, -1, -1]]]
    params['Jet_Parent'] += [[[-1, 0, 0]]]
    expected_masses += [np.sqrt(97)]
    # event 2 has just 1 jet but it is untagged - should be 0
    params['Jet_RootInputIdx'] += [[[0]]]
    params['Jet_InputIdx'] += [[[0, 1, 2]]]
    params['Jet_Px'] += [[[1, 1, 1]]]
    params['Jet_Py'] += [[[1, 1, 1]]]
    params['Jet_Pz'] += [[[1, 1, 1]]]
    params['Jet_Energy'] += [[[10, 10, 10]]]
    params['Jet_Phi'] += [[[0, 0, 0]]]
    params['Jet_PT'] += [[[1, 1, 1]]]
    params['Jet_Rapidity'] += [[[1, 1, 1]]]
    params['Jet_Tags'] += [[[]]]
    params['Jet_Child1'] += [[[1, -1, -1]]]
    params['Jet_Parent'] += [[[-1, 0, 0]]]
    expected_masses += [0.]
    # event 3 has one untagged and one tagged jet
    params['Jet_RootInputIdx'] += [[[0], [3]]]
    params['Jet_InputIdx'] += [[[0, 1, 2], [3, 4, 5]]]
    params['Jet_Px'] += [[[1, 1, 1], [1, 1, 1]]]
    params['Jet_Py'] += [[[1, 1, 1], [1, 1, 1]]]
    params['Jet_Pz'] += [[[1, 1, 1], [1, 1, 1]]]
    params['Jet_Energy'] += [[[10, 10, 10], [10, 10, 10]]]
    params['Jet_Phi'] += [[[0, 0, 0], [0, 0, 0]]]
    params['Jet_PT'] += [[[1, 1, 1], [1, 1, 1]]]
    params['Jet_Rapidity'] += [[[1, 1, 1], [1, 1, 1]]]
    params['Jet_Tags'] += [[[1], []]]
    params['Jet_Child1'] += [[[1, -1, -1], [4, -1, -1]]]
    params['Jet_Parent'] += [[[-1, 0, 0], [-1, 3, 3]]]
    expected_masses += [np.sqrt(10**2 - 3)]
    # event 4 has one untagged and two tagged jets
    params['Jet_RootInputIdx'] += [[[0], [3], [7]]]
    params['Jet_InputIdx'] += [[[0, 1, 2], [3, 4, 5], [6, 7, 8]]]
    params['Jet_Px'] += [[[1, 1, 1], [1, 1, 1], [-1, -1, -1]]]
    params['Jet_Py'] += [[[1, 1, 1], [1, 1, 1], [-1, -1, -1]]]
    params['Jet_Pz'] += [[[1, 1, 1], [1, 1, 1], [-1, -1, -1]]]
    params['Jet_Energy'] += [[[10, 10, 10], [10, 10, 10], [5, 5, 5]]]
    params['Jet_Phi'] += [[[0, 0, 0], [0, 0, 0], [3, 3, 3]]]
    params['Jet_PT'] += [[[1, 1, 1], [1, 1, 1], [1, 1, 1]]]
    params['Jet_Rapidity'] += [[[1, 1, 1], [1, 1, 1], [-1, -1, -1]]]
    params['Jet_Tags'] += [[[1], [], [4]]]
    params['Jet_Child1'] += [[[1, -1, -1], [4, -1, -1], [-1, 8, -1]]]
    params['Jet_Parent'] += [[[-1, 0, 0], [-1, 3, 3], [7, -1, 7]]]
    prep_params = {key: awkward.fromiter([awkward.fromiter(e) for e in v])for key, v in params.items()}
    expected_masses += [15.]
    # check this results in the predicted masses
    with TempTestDir("tst") as dir_name:
        eventWise = Components.EventWise(dir_name, "tmp.awkd")
        eventWise.append(**prep_params)
        found = MassPeaks.all_jet_masses(eventWise, jet_name, 0)
        tst.assert_allclose(found, expected_masses)


def test_all_PT_pairs():
    params = {}
    jet_name = "Jet"
    # event 0 is empty - no pairs
    params['Jet_RootInputIdx'] = [[]]
    params['Jet_InputIdx'] = [[]]
    params['Jet_Px'] = [[]]
    params['Jet_Py'] = [[]]
    params['Jet_Pz'] = [[]]
    params['Jet_Energy'] = [[]]
    params['Jet_Phi'] = [[]]
    params['Jet_PT'] = [[]]
    params['Jet_Rapidity'] = [[]]
    params['Jet_Tags'] = [[]]
    params['Jet_Child1'] = [[]]
    params['Jet_Parent'] = [[]]
    # 0 1, 0 2, 0 3, 1 2, 1 3, 2 3
    expected_order = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    expected = [[] for _ in expected_order]
    # event 1 has one untagged and one tagged jet - no pairs
    params['Jet_RootInputIdx'] += [[[0], [3]]]
    params['Jet_InputIdx'] += [[[0, 1, 2], [3, 4, 5]]]
    params['Jet_Px'] += [[[1, 1, 1], [2, 1, 1]]]
    params['Jet_Py'] += [[[1, 1, 1], [2, 1, 1]]]
    params['Jet_Pz'] += [[[1, 1, 1], [2, 1, 1]]]
    params['Jet_Energy'] += [[[10, 10, 10], [11, 10, 10]]]
    params['Jet_Phi'] += [[[0, 0, 0], [0, 0, 0]]]
    params['Jet_PT'] += [[[1, 1, 1], [2, 1, 1]]]
    params['Jet_Rapidity'] += [[[1, 1, 1], [1, 1, 1]]]
    params['Jet_Tags'] += [[[1], []]]
    params['Jet_Child1'] += [[[1, -1, -1], [4, -1, -1]]]
    params['Jet_Parent'] += [[[-1, 0, 0], [-1, 3, 3]]]
    # event 2 has one untagged and two tagged jets - only the 0 1 pair
    params['Jet_RootInputIdx'] += [[[0], [3], [7], [9]]]
    params['Jet_InputIdx'] += [[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]]
    params['Jet_Px'] += [[[1, 1, 1], [3, 1, 1], [-1, -1, -1], [-1, -1, -1]]]
    params['Jet_Py'] += [[[1, 1, 1], [3, 1, 1], [-1, -1, -1], [-1, -1, -1]]]
    params['Jet_Pz'] += [[[1, 1, 1], [3, 1, 1], [-1, -1, -1], [-1, -1, -1]]]
    params['Jet_Energy'] += [[[10, 10, 10], [12, 12, 12], [5, 5, 5], [10, 10, 10]]]
    params['Jet_Phi'] += [[[0, 0, 0], [0, 0, 0], [3, 3, 3], [1, 3, 3]]]
    params['Jet_PT'] += [[[4, 5, 4], [10, 10, 10], [1, 1, 1], [7, 7, 7]]]
    params['Jet_Rapidity'] += [[[1, 1, 1], [1, 1, 1], [-1, -1, -1], [10, 10, 10]]]
    params['Jet_Tags'] += [[[1], [], [4], []]]
    params['Jet_Child1'] += [[[1, -1, -1], [4, -1, -1], [-1, 8, -1], [11, -1, -1]]]
    params['Jet_Parent'] += [[[-1, 0, 0], [-1, 3, 3], [7, -1, 7], [-1, 9, 9]]]
    expected[expected_order.index((0, 1))].append(15)
    # event 3 has five tagged jet - the 4 with highest PT will contribute to every combination
    params['Jet_RootInputIdx'] += [[[0], [3], [7], [10], [13]]]
    params['Jet_InputIdx'] += [[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14]]]
    params['Jet_Px'] += [[[1, 1, 1], [1, 1, 1], [-1, -1, -1], [1, 1, 1], [-1, -1, -1]]]
    params['Jet_Py'] += [[[1, 1, 1], [2, 2, 2], [-1, -1, -1], [1, 1, 1], [-1, -1, -1]]]
    params['Jet_Pz'] += [[[1, 1, 1], [1, 1, 1], [-1, -1, -1], [1, 1, 1], [3, 3, 3]]]
    params['Jet_Energy'] += [[[10, 10, 10], [10, 10, 10], [5, 5, 5], [10, 10, 10], [6, 6, 6]]]
    params['Jet_Phi'] += [[[0, 0, 0], [0, 0, 0], [3, 3, 3], [0, 0, 0], [3, 3, 3]]]
    params['Jet_PT'] += [[[2.5, 2, 2], [2, 2, 2], [3, 3, 3], [1, 1, 4], [2, 1.5, 1]]]
    params['Jet_Rapidity'] += [[[1, 1, 1], [1, 1, 1], [-1, -1, -1], [1, 1, 1], [-1, -1, -1]]]
    params['Jet_Tags'] += [[[1], [2], [4], [5, 6], [10]]]
    params['Jet_Child1'] += [[[1, -1, -1], [4, -1, -1], [-1, 8, -1], [-1, 11, -1], [-1, 14, -1]]]
    params['Jet_Parent'] += [[[-1, 0, 0], [-1, 3, 3], [7, -1, 7], [-1, 9, 9], [12, -1, 12]]]
    # PT order is 2, 0, 1, 4
    expected[expected_order.index((0, 1))].append(15)
    mass = np.sqrt(15**2 - 1)
    expected[expected_order.index((0, 2))].append(mass)
    mass = np.sqrt(11**2 - 4*3)
    expected[expected_order.index((0, 3))].append(mass)
    mass = np.sqrt(20**2 - 4*2 - 3**2)
    expected[expected_order.index((1, 2))].append(mass)
    mass = np.sqrt(16**2 - 4**2)
    expected[expected_order.index((1, 3))].append(mass)
    mass = np.sqrt(16**2 - 4**2 - 1)
    expected[expected_order.index((2, 3))].append(mass)
    prep_params = {key: awkward.fromiter([awkward.fromiter(e) for e in v])for key, v in params.items()}
    # check this results in the predicted masses
    with TempTestDir("tst") as dir_name:
        eventWise = Components.EventWise(dir_name, "tmp.awkd")
        eventWise.append(**prep_params)
        _, pairs, pair_masses = MassPeaks.all_PT_pairs(eventWise, jet_name, jet_pt_cut=0)
        err_start = f"All expected = {list(zip(expected_order, expected))}\n"
        err_start += f"All found = {list(zip(pairs, pair_masses))}\n"
        for found_pair, found_masses in zip(pairs, pair_masses):
            expected_mass = expected[expected_order.index(tuple(sorted(found_pair)))]
            err_msg = err_start + f"found pair = {found_pair}, found_masses = {found_masses}\n"
            err_msg = err_start + f"expected masses = {expected_mass}\n"
            tst.assert_allclose(found_masses, expected_mass, err_msg=err_msg)


def test_all_doubleTagged_jets():
    params = {}
    jet_name = "Jet"
    # event 0 is empty - no entries
    params['Jet_RootInputIdx'] = [[]]
    params['Jet_InputIdx'] = [[]]
    params['Jet_Px'] = [[]]
    params['Jet_Py'] = [[]]
    params['Jet_Pz'] = [[]]
    params['Jet_Energy'] = [[]]
    params['Jet_Phi'] = [[]]
    params['Jet_PT'] = [[]]
    params['Jet_Rapidity'] = [[]]
    params['Jet_Tags'] = [[]]
    params['Jet_Child1'] = [[]]
    params['Jet_Parent'] = [[]]
    # event 1 has one double tagged jet blow the pt cut and one tagged jet - no entries
    params['Jet_RootInputIdx'] += [[[0], [3]]]
    params['Jet_InputIdx'] += [[[0, 1, 2], [3, 4, 5]]]
    params['Jet_Px'] += [[[1, 1, 1], [2, 1, 1]]]
    params['Jet_Py'] += [[[1, 1, 1], [2, 1, 1]]]
    params['Jet_Pz'] += [[[1, 1, 1], [2, 1, 1]]]
    params['Jet_Energy'] += [[[10, 10, 10], [11, 10, 10]]]
    params['Jet_Phi'] += [[[0, 0, 0], [0, 0, 0]]]
    params['Jet_PT'] += [[[1, 1, 1], [2, 1, 1]]]
    params['Jet_Rapidity'] += [[[1, 1, 1], [1, 1, 1]]]
    params['Jet_Tags'] += [[[1, 2], []]]
    params['Jet_Child1'] += [[[1, -1, -1], [4, -1, -1]]]
    params['Jet_Parent'] += [[[-1, 0, 0], [-1, 3, 3]]]
    # event 2 has one single tagged and two double tagged jets - two entiries
    params['Jet_RootInputIdx'] += [[[0], [3], [7], [9]]]
    params['Jet_InputIdx'] += [[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]]
    params['Jet_Px'] += [[[1, 1, 1], [3, 1, 1], [-1, -1, -1], [-1, -1, -1]]]
    params['Jet_Py'] += [[[1, 1, 1], [3, 1, 1], [-1, -1, -1], [-1, -1, -1]]]
    params['Jet_Pz'] += [[[1, 1, 1], [3, 1, 1], [-1, -1, -1], [-1, -1, -1]]]
    params['Jet_Energy'] += [[[10, 10, 10], [12, 12, 12], [5, 5, 5], [10, 10, 10]]]
    params['Jet_Phi'] += [[[0, 0, 0], [0, 0, 0], [3, 3, 3], [1, 3, 3]]]
    params['Jet_PT'] += [[[4, 5, 4], [10, 10, 10], [11, 11, 11], [7, 7, 7]]]
    params['Jet_Rapidity'] += [[[1, 1, 1], [1, 1, 1], [-1, -1, -1], [10, 10, 10]]]
    params['Jet_Tags'] += [[[1, 12], [1], [4, 5], [3]]]
    params['Jet_Child1'] += [[[1, -1, -1], [4, -1, -1], [-1, 8, -1], [11, -1, -1]]]
    params['Jet_Parent'] += [[[-1, 0, 0], [-1, 3, 3], [7, -1, 7], [-1, 9, 9]]]
    expected = sorted([np.sqrt(10**2 - 3), np.sqrt(5**2 - 3)])
    prep_params = {key: awkward.fromiter([awkward.fromiter(e) for e in v])for key, v in params.items()}
    # check this results in the predicted masses
    with TempTestDir("tst") as dir_name:
        eventWise = Components.EventWise(dir_name, "tmp.awkd")
        eventWise.append(**prep_params)
        masses = MassPeaks.all_doubleTagged_jets(eventWise, jet_name, 2)
        tst.assert_allclose(sorted(masses), expected)


def test_descendants_masses():
    light_pid = 25
    heavy_pid = 35
    params = {}
    jet_name = "Jet"
    # event 0
    params['Px'] = [awkward.fromiter([])]
    params['Py'] = [awkward.fromiter([])]
    params['Pz'] = [awkward.fromiter([])]
    params['Energy'] = [awkward.fromiter([])]
    params['Children'] = [awkward.fromiter([])]
    params['Parents'] = [awkward.fromiter([])]
    params['MCPID'] = [awkward.fromiter([])]
    params['JetInputs_SourceIdx'] = [awkward.fromiter([])]
    expected_light = []
    expected_heavy = []
    with TempTestDir("tst") as dir_name:
        eventWise = Components.EventWise(dir_name, "tmp.awkd")
        eventWise.append(**params)
        with pytest.raises(AssertionError):
            heavy, light1, light2 = MassPeaks.descendants_masses(eventWise, False)
    # event 1
    params['JetInputs_SourceIdx'] = [awkward.fromiter(np.arange(8))]
    #                                        0   1    2    3    4   5    6             7   8   9   10
    params['Children'] = [awkward.fromiter([[], [3], [],  [6], [1, 10], [],  [2, 7, 8, 9], [], [], [], []])]
    params['Parents'] =  [awkward.fromiter([[], [4],  [6], [1], [], [3], [],           [6],[6],[6],[4]])]
    params['MCPID'] =    [awkward.fromiter([4,  25,  5,   3,   35,  1,  -5,           -1,  7,  11, 25])]
    params['Px'] =       [awkward.fromiter([1,  1,   1,   1,   1,  1,   1,            1,   1,  1,  1])]
    params['Py'] =       [awkward.fromiter([1,  1,   1,   1,   1,  1,   1,            1,   1,  1,  1])]
    params['Pz'] =       [awkward.fromiter([1,  1,   1,   1,   1,  1,   1,            1,   1,  1,  1])]
    params['Energy'] =   [awkward.fromiter([6,  6,   6,   6,   6,  6,   6,            6,   6,  6,  6])]
    expected_light = [np.sqrt((4*6)**2 - 3*4**2), np.sqrt(6**2 - 3)]
    expected_heavy = [np.sqrt((5*6)**2 - 3*5**2)]
    with TempTestDir("tst") as dir_name:
        eventWise = Components.EventWise(dir_name, "tmp.awkd")
        eventWise.append(**params)
        heavy, light1, light2 = MassPeaks.descendants_masses(eventWise, False)
        tst.assert_allclose(light1, max(expected_light))
        tst.assert_allclose(light2, min(expected_light))
        tst.assert_allclose(heavy, expected_heavy)
