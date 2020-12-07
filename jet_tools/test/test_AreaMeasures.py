""" Test the function in AreaMeasures.py """
from tree_tagger import AreaMeasures, Components, FormShower
import numpy as np
import awkward
import numpy.testing as tst
from test.tools import TempTestDir

def test_width():
    linear = np.array([0, np.inf])
    cyclic = np.array([0., -1.5*np.pi])
    found = AreaMeasures.width(linear, cyclic)
    assert np.isinf(found)
    assert found > 0
    linear = np.array([0,  0.])
    cyclic = np.array([0., -1.5*np.pi])
    found = AreaMeasures.width(linear, cyclic)
    tst.assert_allclose(found, 0.5*np.pi)
    linear = np.array([0, 1.,      ])
    cyclic = np.array([0., 2*np.pi])
    found = AreaMeasures.width(linear, cyclic)
    tst.assert_allclose(found, 1.)
    linear = np.array([0, -1.,      ])
    cyclic = np.array([-np.pi, np.pi])
    found = AreaMeasures.width(linear, cyclic)
    tst.assert_allclose(found, 1.)
    linear = np.array([0, 0.,     1.,   ])
    cyclic = np.array([0., np.pi, 2*np.pi])
    expected = np.array([0., np.pi, 1., np.sqrt(1 + np.pi**2)])
    found = AreaMeasures.width(linear, cyclic)
    tst.assert_allclose(found, np.max(expected))



def test_descendant_widths():
    # check a normal case
    jetinputs_sourceidx = [0, 5, 6, 7]
    #           0   1       2    3       4   5   6          7   8   9   10
    children = [[], [2, 3], [5], [6, 5], [], [], [7, 8, 9], [], [], [], []]
    rapidity = [0,  -1,    -1,  -1,     -1,  1,  0,         0, -1, -1,  -1]
    phi      = [0,  -1,    -1,  -1,     -1,  1,  0,         0, -1, -1,  -1]
    content = {"JetInputs_SourceIdx": jetinputs_sourceidx,
               "Children": children,
               "Rapidity": rapidity,
               "Phi": phi}
    content = {k: awkward.fromiter([v]) for k, v in content.items()}
    with TempTestDir("tst") as dir_name:
        eventWise = Components.EventWise(dir_name, "tmp.awkd")
        eventWise.append(**content)
        eventWise.selected_index = 0
        root_idxs = [0, 1, 4, 10]
        found = AreaMeasures.decendants_width(eventWise, *root_idxs)
        tst.assert_allclose(found, np.sqrt(2))
        # check what happens if no decndants are found
        content["JetInputs_SourceIdx"]= awkward.fromiter([[]])
        eventWise = Components.EventWise(dir_name, "tmp.awkd")
        eventWise.append(**content)
        eventWise.selected_index = 0
        root_idxs = [0, 1, 4, 10]
        found = AreaMeasures.decendants_width(eventWise, *root_idxs)
        tst.assert_allclose(found, 0.)
        



def test_append_b_shower_widths():
    # check a normal case
    jetinputs_sourceidx = [0, 5, 6, 7]
    bquakidx = [0, 1, 4, 10]
    #           0   1       2    3       4   5   6          7   8   9   10
    children = [[], [2, 3], [5], [6, 5], [], [], [7, 8, 9], [], [], [], []]
    rapidity = [0,  -1,    -1,  -1,     -1,  1,  0,         0, -1, -1,  -1]
    phi      = [0,  -1,    -1,  -1,     -1,  1,  0,         0, -1, -1,  -1]
    content = {"JetInputs_SourceIdx": jetinputs_sourceidx,
               "BQuarkIdx": bquakidx,
               "Children": children,
               "Rapidity": rapidity,
               "Phi": phi}
    content = {k: awkward.fromiter([v]) for k, v in content.items()}
    with TempTestDir("tst") as dir_name:
        eventWise = Components.EventWise(dir_name, "tmp.awkd")
        eventWise.append(**content)
        AreaMeasures.append_b_shower_widths(eventWise)
        eventWise.selected_index = None
        expected = [0., np.sqrt(2), 0., 0.]
        tst.assert_allclose(eventWise.BWidth[0], expected)
        

def test_append_b_jet_widths():
    # check a normal case
    detectabletag_leaves = [0, 2, 3]
    #           0   1       2    3       4   5   6          7   8   9   10
    jetinputs_sourceidx =\
               [0,   7,       8,   9,     1,   2, 10,        3,    4,  5, 6]
    child1 = [[  -1, 1,      9,   2,      -1, -1, 4,         -1, -1, -1, -1]]
    rapidity = [[0,  -1,    -1,  -1,     -1,  1,  0,         0, -1, -1,  -1]]
    phi      = [[0,  -1,    -1,  -1,     -1,  1,  0,         0, -1, -1,  -1]]
    tags = [[0]]
    content = {"JetInputs_SourceIdx": jetinputs_sourceidx,
               "DogJet_InputIdx": [jetinputs_sourceidx],  # if there is only one jet...
               "DetectableTag_Leaves": detectabletag_leaves,
               "DogJet_Child1": child1,
               "DogJet_Rapidity": rapidity,
               "DogJet_Tags": tags,
               "DogJet_Phi": phi}
    content = {k: awkward.fromiter([v]) for k, v in content.items()}
    with TempTestDir("tst") as dir_name:
        eventWise = Components.EventWise(dir_name, "tmp.awkd")
        eventWise.append(**content)
        AreaMeasures.append_b_jet_widths(eventWise, "DogJet", signal_only=False)
        AreaMeasures.append_b_jet_widths(eventWise, "DogJet", signal_only=True)
        eventWise.selected_index = None
        tst.assert_allclose(eventWise.DogJet_BWidth[0], np.sqrt(8))
        tst.assert_allclose(eventWise.DogJet_BSigWidth[0], np.sqrt(2))


