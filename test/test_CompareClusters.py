""" tests for the CompareClusters module """
import os
from ipdb import set_trace as st
import awkward
import pytest
import numpy as np
from numpy import testing as tst
from tools import TempTestDir
from tree_tagger import Components, CompareClusters, Constants
import unittest.mock


def test_get_best():
    with TempTestDir("tst") as dir_name:
        # this will raise a value error if given an empty eventWise
        save_name = "test.awkd"
        ew = Components.EventWise(dir_name, save_name)
        with pytest.raises(ValueError):
            CompareClusters.get_best(ew, "Duck")
        # it should also raise a value error if told to get the best of a jet not present
        params = dict(DogJet1_AveSignalMassRatio=0.2,
                      DogJet1_AveBGMassRatio=0.5,
                      PigJet_AveSignalMassRatio=np.nan,
                      PigJet_AveBGMassRatio=1.,
                      DogJet2_AveSignalMassRatio=0.5,
                      DogJet2_AveBGMassRatio=0.1)
        ew.append_hyperparameters(**params)
        with pytest.raises(ValueError):
            CompareClusters.get_best(ew, "Duck")
        # if asked for something wiht only nan scores it shoudl return one of them
        found = CompareClusters.get_best(ew, "Pig")
        assert found == "PigJet"
        # if asked for something with multiple vaild scores, chose the one with the
        # best ratio
        found = CompareClusters.get_best(ew, "Dog")
        assert found == "DogJet2"


def test_add_bg_mass():
    # will need BQuarkIdx JetInputs_SourceIdx, DetectableTag_Leaves, 
    # Energy Px Py Pz
    params = {}
    # an empty event should have 0 mass
    params["BQuarkIdx"] = [awkward.fromiter([3,4])]
    params["JetInputs_SourceIdx"] = [awkward.fromiter([])]
    params["DetectableTag_Leaves"] = [awkward.fromiter([])]
    params["Energy"] = [awkward.fromiter([])]
    params["Px"] = [awkward.fromiter([])]
    params["Py"] = [awkward.fromiter([])]
    params["Pz"] = [awkward.fromiter([])]
    # an event with no jet_inputs should also have no mass
    params["BQuarkIdx"] += [awkward.fromiter([1,2,3,4])]
    params["JetInputs_SourceIdx"] += [awkward.fromiter([])]
    params["DetectableTag_Leaves"] += [awkward.fromiter([[5, 6], [7,8]])]
    params["Energy"] += [awkward.fromiter([0, 0, 3, 0, 0, 0, 1, 10, 9])]
    params["Px"] += [awkward.fromiter([0, 0, 1, 0, 0, 0, 1, 0, -3])]
    params["Py"] += [awkward.fromiter([0, 0, 1, 1, 0, 0, -1, 0, 3])]
    params["Pz"] += [awkward.fromiter([0, 0, 0, 0, 0, 0, 1, 1, 3])]
    # a standard event can have mass
    params["BQuarkIdx"] += [awkward.fromiter([1,2,3,4])]
    params["JetInputs_SourceIdx"] += [awkward.fromiter([2, 6, 7])]
    params["DetectableTag_Leaves"] += [awkward.fromiter([[5, 6], [7,8]])]
    params["Energy"] += [awkward.fromiter([0, 0, 3, 0, 0, 0, 1, 10, 9])]
    params["Px"] += [awkward.fromiter([0, 0, 1, 0, 0, 0, 1, 0, -3])]
    params["Py"] += [awkward.fromiter([0, 0, 1, 1, 0, 0, -1, 0, 3])]
    params["Pz"] += [awkward.fromiter([0, 0, 0, 0, 0, 0, 1, 1, 3])]
    expected = [0, 0, np.sqrt(9 - 2)]
    with TempTestDir("tst") as dir_name:
        # this will raise a value error if given an empty eventWise
        save_name = "test.awkd"
        ew = Components.EventWise(dir_name, save_name)
        ew.append(**params)
        # settign a selected index should not screw things up
        ew.selected_index = 1
        CompareClusters.add_bg_mass(ew)
        ew.selected_index = None
        tst.assert_allclose(ew.DetectableBG_Mass, expected)


def test_get_detectable_comparisons():
    # will need BQuarkIdx JetInputs_SourceIdx, DetectableTag_Leaves, DetectableTag_Roots
    # DetectableTag_Energy, DetectableTag_Px, DetectableTag_Py, DetectableTag_Pz
    # Energy Px Py Pz
    # Jet_Parent, Jet_TagMass, Jet_InputIdx
    # Jet_Energy, Jet_Px, Jet_Py, Jet_Pz
    # Jet_MTags
    # will create Jet_DistanceRapidity Jet_DistancePT Jet_DistancePhi
    # Jet_SignalMassRatio Jet_BGMassRatio Jet_PercentFound
    # an empty event should be nan
    params = {}
    params["BQuarkIdx"] = [awkward.fromiter([])]
    params["JetInputs_SourceIdx"] = [awkward.fromiter([])]
    params["DetectableTag_Roots"] = [awkward.fromiter([])]
    params["DetectableTag_Leaves"] = [awkward.fromiter([])]
    params["DetectableTag_Energy"] = [awkward.fromiter([])]
    params["DetectableTag_Px"] = [awkward.fromiter([])]
    params["DetectableTag_Py"] = [awkward.fromiter([])]
    params["DetectableTag_Pz"] = [awkward.fromiter([])]
    params["Energy"] = [awkward.fromiter([])]
    params["Px"] = [awkward.fromiter([])]
    params["Py"] = [awkward.fromiter([])]
    params["Pz"] = [awkward.fromiter([])]
    params["Jet_Energy"] = [awkward.fromiter([])]
    params["Jet_Px"] = [awkward.fromiter([])]
    params["Jet_Py"] = [awkward.fromiter([])]
    params["Jet_Pz"] = [awkward.fromiter([])]
    params["Jet_Parent"] = [awkward.fromiter([])]
    params["Jet_TagMass"] = [awkward.fromiter([])]
    params["Jet_InputIdx"] = [awkward.fromiter([])]
    params["Jet_MTags"] = [awkward.fromiter([])]
    expected_distances = [[[], [], []]]
    expected_sig = [[]]
    expected_bg = [[]]
    expected_percent = [np.nan]
    # an event with tags but no jets
    params["BQuarkIdx"] += [awkward.fromiter([3, 4])]
    params["JetInputs_SourceIdx"] += [awkward.fromiter([])]
    params["DetectableTag_Roots"] += [awkward.fromiter([[3], [4]])]
    params["DetectableTag_Leaves"] += [awkward.fromiter([])]
    params["DetectableTag_Energy"] += [awkward.fromiter([0., 0.])]
    params["DetectableTag_Px"] += [awkward.fromiter([0., 0.])]
    params["DetectableTag_Py"] += [awkward.fromiter([0., 0.])]
    params["DetectableTag_Pz"] += [awkward.fromiter([0., 0.])]
    params["Energy"] += [awkward.fromiter([])]
    params["Px"] += [awkward.fromiter([])]
    params["Py"] += [awkward.fromiter([])]
    params["Pz"] += [awkward.fromiter([])]
    params["Jet_Energy"] += [awkward.fromiter([])]
    params["Jet_Px"] += [awkward.fromiter([])]
    params["Jet_Py"] += [awkward.fromiter([])]
    params["Jet_Pz"] += [awkward.fromiter([])]
    params["Jet_Parent"] += [awkward.fromiter([])]
    params["Jet_TagMass"] += [awkward.fromiter([])]
    params["Jet_InputIdx"] += [awkward.fromiter([])]
    params["Jet_MTags"] += [awkward.fromiter([])]
    expected_distances += [[[np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]]]
    expected_sig += [[np.nan, np.nan]]
    expected_bg += [[np.nan, np.nan]]
    expected_percent += [0.]
    # the only thing that changes in the next 3 events is the tags
    repeat = 4
    params["BQuarkIdx"] += [awkward.fromiter([3,4])] * repeat
    params["JetInputs_SourceIdx"] += [awkward.fromiter([1, 2, 4, 5, 6, 8])] * repeat
    params["DetectableTag_Roots"] += [awkward.fromiter([[3], [4]])] * repeat
    params["DetectableTag_Leaves"] += [awkward.fromiter([[2, 4], [5]])] * repeat
    params["DetectableTag_Energy"] += [awkward.fromiter([5., 10.])] * repeat
    params["DetectableTag_Px"] += [awkward.fromiter([0., 0.])] * repeat
    params["DetectableTag_Py"] += [awkward.fromiter([1., -1.])] * repeat
    params["DetectableTag_Pz"] += [awkward.fromiter([3., 4.])] * repeat
    # InputIdx                             0   1   2   3   4   5   6   7   8
    params["Energy"] += [awkward.fromiter([3., 4., 4., 2., 1.,10., 6., 9., 6.])] * repeat
    params["Px"] +=     [awkward.fromiter([0., 1.,-1., 2., 1., 0.,-1., 0., 5.])] * repeat
    params["Py"] +=     [awkward.fromiter([0., 0., 3., 0.,-2.,-1., 0., 0., 0.])] * repeat
    params["Pz"] +=     [awkward.fromiter([1., 3., 2.,-1., 1., 4., 1., 0., 0.])] * repeat
    params["Jet_Energy"] = [awkward.fromiter([[4., 1., 5.], [10.], [12., 6., 6.]])] * repeat
    params["Jet_Px"] = [awkward.fromiter([[-1., 1., 0.], [0.], [4., -1., 5.]])] * repeat
    params["Jet_Py"] = [awkward.fromiter([[3., -2., 1.], [-1.], [0., 0., 0.]])] * repeat
    params["Jet_Pz"] = [awkward.fromiter([[2., 1., 3.], [4.], [1., 1., 0.]])] * repeat
    params["Jet_Parent"] += [awkward.fromiter([[10, 10, -1], [-1], [-1, 40, 40]])] * repeat
    params["Jet_TagMass"] += [awkward.fromiter([[1., 0.], [0., 1.]])] * repeat
    params["Jet_InputIdx"] += [awkward.fromiter([[1, 2, 10], [3], [40, 4, 5]])] * repeat
    # a perfect event should return a perfect score
    params["Jet_MTags"] += [awkward.fromiter([[3], [4], []])]
    expected_distances += [[[0, 0], [0, 0], [0, 0]]]
    expected_sig += [[1, 1]]
    expected_bg += [[0., 0.]]
    expected_percent += [1.]
    # a perfectly imperfect event should return the worst score
    params["Jet_MTags"] += [awkward.fromiter([[], [], [3]])]
    
    # detectable tag is at [5, 0, 1, 3], matched jet is at [12, 4, 0, 1]
    (tag_phi, jet_phi), (tag_pt, jet_pt) = Components.pxpy_to_phipt(np.array([0., 4.]),
                                                                    np.array([1., 0.]))
    tag_rapidity, jet_rapidity = Components.ptpze_to_rapidity(np.array([tag_pt, jet_pt]), 
                                                              np.array([3., 1.]),
                                                              np.array([5., 12.]))
    phi_distance = Components.angular_distance(tag_phi, jet_phi)
    expected_distances += [[[phi_distance, np.nan], [abs(tag_pt-jet_pt), np.nan],
                            [abs(jet_rapidity-tag_rapidity), np.nan]]]
    expected_sig += [[0, 0]]
    # the total bg mass is made from indices 1, 6 and 8
    total_bg = np.sqrt(16**2 - 25 - 16)
    # the bg in tagged jets is divided by this
    mass_jet3 = np.sqrt(12**2 - 17)
    expected_bg += [[mass_jet3/total_bg, 0.]]
    expected_percent += [0.5]
    # a mixed event returns a score that can be calculated
    params["Jet_MTags"] += [awkward.fromiter([[], [4], [3]])]
    expected_distances += [[[phi_distance, 0], [abs(tag_pt-jet_pt), 0],
                            [abs(jet_rapidity-tag_rapidity), 0]]]
    expected_sig += [[0, 1]]
    expected_bg += [[mass_jet3/total_bg, 0.]]
    expected_percent += [1.]
    # in an event where a jet has tags from two diferent shower clusters it should
    # be assocated witht he one where it has greates inheritance
    params["Jet_MTags"] += [awkward.fromiter([[], [4, 3], []])]
    # the other one will simply be unfound
    expected_distances += [[[np.nan, 0], [np.nan, 0],
                            [np.nan, 0]]]
    expected_sig += [[0, 1]]
    expected_bg += [[0., 0.]]
    expected_percent += [.5]
    params = {key: awkward.fromiter(val) for key,val in params.items()}
    with TempTestDir("tst") as dir_name:
        # this will raise a value error if given an empty eventWise
        save_name = "test.awkd"
        ew = Components.EventWise(dir_name, save_name)
        ew.append(**params)
        jet_idxs = awkward.fromiter([[], [], [0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]])
        CompareClusters.get_detectable_comparisons(ew, "Jet", jet_idxs, True)
        for event_n in range(len(expected_percent)):
            ew.selected_index = event_n
            err_msg = f"Event {event_n}; kinematics differ from expected"
            found = np.array([ew.Jet_DistancePhi.tolist(), ew.Jet_DistancePT.tolist(),
                               ew.Jet_DistanceRapidity.tolist()])
            #print()
            #print((found, expected_distances[event_n]))
            tst.assert_allclose(found, expected_distances[event_n], err_msg=err_msg)
            err_msg = f"Event {event_n}; signal ratios differs from expected"
            #print([ew.Jet_SignalMassRatio, expected_sig[event_n]])
            tst.assert_allclose(ew.Jet_SignalMassRatio, expected_sig[event_n], err_msg=err_msg)
            err_msg = f"Event {event_n}; background ratios differs from expected"
            #print((ew.Jet_BGMassRatio, expected_bg[event_n]))
            tst.assert_allclose(ew.Jet_BGMassRatio, expected_bg[event_n], err_msg=err_msg)
            err_msg = f"Event {event_n}; percentage found differs from expected"
            #print([ew.Jet_PercentFound, expected_percent[event_n]])
            tst.assert_allclose(ew.Jet_PercentFound, expected_percent[event_n], err_msg=err_msg)
            

def fake_quality_width_fraction(eventWise, name, mass):
    return np.nan, len(name)/mass


def fake_detectable_comparisons(eventWise, name, jet_idx, append):
    scores = {}
    scores[name + "_Bork"] = awkward.fromiter([np.inf, np.inf, np.inf])
    scores[name + "_Quack"] = awkward.fromiter([[0.5, np.nan], [0.5, -np.inf], []])
    scores[name + "_SeperateMask"] = awkward.fromiter([False, True, True])
    return scores

def fake_empty(*args, **kwargs):
    pass


def test_append_scores():
    # set up as for filter jets
    params = {}
    # event 0
    params['Jet_InputIdx'] = [awkward.fromiter([])]
    params['Jet_Parent'] = [awkward.fromiter([])]
    params['Jet_Child1'] = [awkward.fromiter([])]
    params['Jet_PT'] =     [awkward.fromiter([])]
    params['Energy'] = [awkward.fromiter([])]
    params['Px'] = [awkward.fromiter([])]
    params['Py'] = [awkward.fromiter([])]
    params['Pz'] = [awkward.fromiter([])]
    params['Children'] = [awkward.fromiter([])]
    params['Parents'] = [awkward.fromiter([])]
    params['MCPID'] = [awkward.fromiter([])]
    params['JetInputs_SourceIdx'] = [awkward.fromiter([])]
    params['BQuarkIdx'] = [awkward.fromiter([])]
    # event 1
    params['Jet_InputIdx'] = [awkward.fromiter([[0], [1, 2, 3, 4, 5]])]
    params['Jet_Parent'] += [awkward.fromiter([[-1], [-1, 1, 1, 2, 2]])]
    params['Jet_Child1'] += [awkward.fromiter([[-1], [1, 2, -1, -1, -1]])]
    params['Jet_PT'] +=     [awkward.fromiter([[50.,], [0.2, 0.1, 0., 0., .1]])]
    params['JetInputs_SourceIdx'] += [awkward.fromiter(np.arange(6))]
    params['Energy'] += [awkward.fromiter([30., 10., 20., 70., 20., 10., 45., 56., 40., 25.])]
    params['Px'] += [awkward.fromiter([3., 0., 2., 1., 2., -1., 0., 3., -1., 0.])]
    params['Py'] += [awkward.fromiter([3., 0., 2., 2., 2., 1., -1., -3., 0., -1.])]
    params['Pz'] += [awkward.fromiter([3., 0., 2., 0., 2., 2., -5., -2., 1., 0.])]
    params['Children'] += [awkward.fromiter([[], [3], [],  [5], [], [],  [2, 7, 8, 9], [], [], [], []])]
    params['Parents'] +=  [awkward.fromiter([[], [],  [6], [1], [], [3], [],           [6],[6],[6],[]])]
    params['MCPID'] +=    [awkward.fromiter([4, -5,   5,   3,   2,  1,   -5,          -1,  7,  11, 12])]
    params['BQuarkIdx'] += [awkward.fromiter([1, 6])]
    # event 2
    params['Jet_InputIdx'] = [awkward.fromiter([[0], [1, 2, 3]])]
    params['Jet_Parent'] += [awkward.fromiter([[-1], [-1, 1, 1]])]
    params['Jet_Child1'] += [awkward.fromiter([[-1], [1, -1, -1]])]
    params['Jet_PT'] +=     [awkward.fromiter([[50.,], [21., 0.1, 0.]])]
    params['JetInputs_SourceIdx'] += [awkward.fromiter(np.arange(6))]
    params['Energy'] += [awkward.fromiter([30., 10., 20., 70., 20., 10., 45., 56., 40., 25.])]
    params['Px'] += [awkward.fromiter([3., 0., 2., 1., 2., -1., 0., 3., -1., 0.])]
    params['Py'] += [awkward.fromiter([3., 0., 2., 2., 2., 1., -1., -3., 0., -1.])]
    params['Pz'] += [awkward.fromiter([3., 0., 2., 0., 2., 2., -5., -2., 1., 0.])]
    params['Children'] += [awkward.fromiter([[], [3], [],  [5], [], [],  [2, 7, 8, 9], [], [], [], []])]
    params['Parents'] +=  [awkward.fromiter([[], [],  [6], [1], [], [3], [],           [6],[6],[6],[]])]
    params['MCPID'] +=    [awkward.fromiter([4, -5,   5,   3,   2,  1,   -5,          -1,  7,  11, 12])]
    params['BQuarkIdx'] += [awkward.fromiter([1, 6])]
    params = {key: awkward.fromiter(val) for key,val in params.items()}
    with TempTestDir("tst") as dir_name:
        # this will raise a value error if given an empty eventWise
        save_name = "test.awkd"
        ew = Components.EventWise(dir_name, save_name)
        ew.append(**params)
        # mock JetQuality.quality_width_fracton and get_detectable_comparisons
        with unittest.mock.patch('tree_tagger.JetQuality.quality_width_fracton',
                                 new=fake_quality_width_fraction):
            with unittest.mock.patch('tree_tagger.CompareClusters.get_detectable_comparisons',
                                     new=fake_detectable_comparisons):
                with unittest.mock.patch('tree_tagger.TrueTag.add_detectable_fourvector',
                                         new=fake_empty):
                    CompareClusters.append_scores(ew)
                    tst.assert_allclose(ew.Jet_Bork, [np.inf, np.inf, np.inf])
                    tst.assert_allclose(ew.Jet_Quack[0], [0.5, np.nan])
                    tst.assert_allclose(ew.Jet_Quack[1], [0.5, -np.inf])
                    tst.assert_allclose(ew.Jet_Quack[2], [])
                    assert np.isnan(ew.Jet_AveBork)
                    tst.assert_allclose(ew.Jet_AveQuack, 0.5)
                    assert np.isnan(ew.Jet_QualityWidth)
                    tst.assert_allclose(ew.Jet_QualityFraction, 3/Constants.dijet_mass)


def test_tabulate_scores():
    params1 = {}
    params2 = {}
    params1["DogJet_QualityWidth"] = 4
    params1["DogJet_DeltaR"] = .5
    params2["CatJet_QualityWidth"] = np.inf
    with TempTestDir("tst") as dir_name:
        # this will raise a value error if given an empty eventWise
        save_name1 = "test1.awkd"
        save_name2 = "test2.awkd"
        ew1 = Components.EventWise(dir_name, save_name1)
        ew1.append_hyperparameters(**params1)
        ew1.append(DogJet_InputIdx = awkward.fromiter([[]]))
        path1 = os.path.join(dir_name, save_name1)
        ew2 = Components.EventWise(dir_name, save_name2)
        ew2.append_hyperparameters(**params2)
        ew2.append(CatJet_InputIdx = awkward.fromiter([[]]))
        path2 = os.path.join(dir_name, save_name2)
        all_cols, variable_cols, score_cols, table = CompareClusters.tabulate_scores([path1, path2])
        assert len(all_cols) == len(table[0])
        assert len(all_cols) == len(variable_cols) + len(score_cols) + 3
        assert len(table) == 2
        dog_row = next(i for i, name in enumerate(table[:, all_cols.index("jet_name")])
                       if name == "DogJet")
        cat_row = next(i for i, name in enumerate(table[:, all_cols.index("jet_name")])
                       if name == "CatJet")
        use_cols = [all_cols.index("QualityWidth"), all_cols.index("DeltaR"), all_cols.index("AveBGMassRatio")]
        tst.assert_allclose(table[dog_row, use_cols], [4, .5, np.nan])
        tst.assert_allclose(table[cat_row, use_cols], [np.inf, np.nan, np.nan])


def test_make_scale():
    # this has two modes, ordinal and float
    # either way it should return somehting sensible
    # given an empty list it should not choke
    empty = np.array([])
    positions, scale_positions, scale_labels = CompareClusters.make_scale(empty)
    assert len(positions) == 0
    assert len(scale_positions) == 0
    assert len(scale_labels) == 0
    # a list with just one number should return it
    one = np.array([1])
    positions, scale_positions, scale_labels = CompareClusters.make_scale(one)
    assert len(positions) == 1
    tst.assert_allclose(positions, one)
    assert len(scale_positions) == 1
    tst.assert_allclose(scale_positions, one)
    assert len(scale_labels) == 1
    assert scale_labels[0] == '1'
    # ditto for one string
    one = np.array(['frog'])
    positions, scale_positions, scale_labels = CompareClusters.make_scale(one)
    assert len(positions) == 1
    assert len(scale_positions) == 1
    assert len(scale_labels) == 1
    assert scale_labels[0] == one[0]
    # one nan or inf should return that, but it may also add 0 as a scale pos
    one = np.array([np.nan])
    positions, scale_positions, scale_labels = CompareClusters.make_scale(one)
    assert len(positions) == 1
    scale_loc = np.argmin(np.abs(scale_positions - positions[0]))
    assert 'nan' in scale_labels[scale_loc] or 'None' in scale_labels[scale_loc]
    one = np.array([-np.inf])
    positions, scale_positions, scale_labels = CompareClusters.make_scale(one)
    assert len(positions) == 1
    scale_loc = np.argmin(np.abs(scale_positions - positions[0]))
    assert 'inf' in scale_labels[scale_loc] and '-' in scale_labels[scale_loc]
    one = np.array([np.inf])
    positions, scale_positions, scale_labels = CompareClusters.make_scale(one)
    assert len(positions) == 1
    scale_loc = np.argmin(np.abs(scale_positions - positions[0]))
    assert 'inf' in scale_labels[scale_loc] and '-' not in scale_labels[scale_loc]
    # now try some more complex arrangements
    inputs = [[4, 3.2], [4, 3.2, np.nan], [4, 3.2, 0., np.nan], [-4, 3.2, np.nan],
              [4, 3.2, np.inf], [4, 3.2, -np.inf], [-4, 3.2, 0, np.nan, np.inf],
              [None, 'dog', 'cat'],
              [np.inf, 1, 3, 5, 3.4, 10, np.nan, 2.2, 7, 3, 5, 9, 2, 2, -np.inf]]
    closest_contains = [['4', '3'], ['4', '3', 'nan'], ['4', '3', '0', 'nan'],
                        ['-4', '3', 'nan'], ['4', '3', 'inf'], ['4', '3', '-'], 
                        ['-4', '3', '0', 'nan', 'inf'], ['none', 'dog', 'cat'],
                        ['inf', '1', '3', '5', '3', '10', 'nan', '2', '7', '3', '5', '9',
                         '2', '2', '-']]
    for inp, close in zip(inputs, closest_contains):
        # make the test scale
        inp = np.array(inp)
        positions, scale_positions, scale_labels = CompareClusters.make_scale(inp)
        # check that for each of the values in the test scale
        for i, pos in enumerate(positions):
            s_position = np.argmin(np.abs(scale_positions - pos))
            # the closest label contains the expected string
            assert close[i] in scale_labels[s_position].lower(), f"inputs={inp}, expected={close[i]}, found={scale_labels[s_position]}"
    # finally check the behavior with tuples
    inp = [None, ('knn', 5), ('knn', 4), ('distance', 0), ('distance', 2)]
    # make the test scale
    inp = awkward.fromiter(inp)
    positions, scale_positions, scale_labels = CompareClusters.make_scale(inp)
    assert positions[1] > positions[2]
    assert positions[4] > positions[3]
    label0 = scale_labels[np.argmin(np.abs(scale_positions - positions[0]))].lower()
    assert 'none' in label0
    label1 = scale_labels[np.argmin(np.abs(scale_positions - positions[1]))].lower()
    assert 'knn' in label1 and '5' in label1
    label2 = scale_labels[np.argmin(np.abs(scale_positions - positions[2]))].lower()
    assert 'knn' in label2 and '4' in label2
    label3 = scale_labels[np.argmin(np.abs(scale_positions - positions[3]))].lower()
    assert 'distance' in label3 and '0' in label3
    label4 = scale_labels[np.argmin(np.abs(scale_positions - positions[4]))].lower()
    assert 'distance' in label4 and '2' in label4






