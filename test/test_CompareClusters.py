""" tests for the CompareClusters module """
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
    # Jet_Parent, Jet_Inheritance, Jet_InputIdx
    # Jet_Energy, Jet_Px, Jet_Py, Jet_Pz
    # Jet_ITags
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
    params["Jet_Inheritance"] = [awkward.fromiter([])]
    params["Jet_InputIdx"] = [awkward.fromiter([])]
    params["Jet_ITags"] = [awkward.fromiter([])]
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
    params["Jet_Inheritance"] += [awkward.fromiter([])]
    params["Jet_InputIdx"] += [awkward.fromiter([])]
    params["Jet_ITags"] += [awkward.fromiter([])]
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
    params["Jet_Inheritance"] += [awkward.fromiter([[[1., 1., 1.], [0.], [0., 0., 0.]],
                                                    [[0., 0., 0.], [1.], [0., 0., 0.]]])] * repeat
    params["Jet_InputIdx"] += [awkward.fromiter([[1, 2, 10], [3], [40, 4, 5]])] * repeat
    # a perfect event should return a perfect score
    params["Jet_ITags"] += [awkward.fromiter([[3], [4], []])]
    expected_distances += [[[0, 0], [0, 0], [0, 0]]]
    expected_sig += [[1, 1]]
    expected_bg += [[0., 0.]]
    expected_percent += [1.]
    # a perfectly imperfect event should return the worst score
    params["Jet_ITags"] += [awkward.fromiter([[], [], [3]])]
    
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
    params["Jet_ITags"] += [awkward.fromiter([[], [4], [3]])]
    expected_distances += [[[phi_distance, 0], [abs(tag_pt-jet_pt), 0],
                            [abs(jet_rapidity-tag_rapidity), 0]]]
    expected_sig += [[0, 1]]
    expected_bg += [[mass_jet3/total_bg, 0.]]
    expected_percent += [1.]
    # in an event where a jet has tags from two diferent shower clusters it should
    # be assocated witht he one where it has greates inheritance
    params["Jet_ITags"] += [awkward.fromiter([[], [4, 3], []])]
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
            

def test_filter_jets():
    # will need 
    # Jet_Parent, Jet_Child1, Jet_PT
    min_pt = 0.1
    min_ntracks = 2
    params = {}
    # an empty event should return nothing
    params['Jet_Parent'] = [awkward.fromiter([])]
    params['Jet_Child1'] = [awkward.fromiter([])]
    params['Jet_PT'] =     [awkward.fromiter([])]
    # an event with nothing that passes cuts
    params['Jet_Parent'] += [awkward.fromiter([[-1], [-1, 1, 1, 2, 2]])]
    params['Jet_Child1'] += [awkward.fromiter([[-1], [1, 2, -1, -1, -1]])]
    params['Jet_PT'] +=     [awkward.fromiter([[50.,], [0.2, 0.1, 0., 0., .1]])]
    # an event with somthing that passes cuts
    params['Jet_Parent'] += [awkward.fromiter([[-1], [-1, 1, 1]])]
    params['Jet_Child1'] += [awkward.fromiter([[-1], [1, -1, -1]])]
    params['Jet_PT'] +=     [awkward.fromiter([[50.,], [21., 0.1, 0.]])]
    params = {key: awkward.fromiter(val) for key,val in params.items()}
    with TempTestDir("tst") as dir_name:
        # this will raise a value error if given an empty eventWise
        save_name = "test.awkd"
        ew = Components.EventWise(dir_name, save_name)
        ew.append(**params)
        # using defaults
        jet_idxs = CompareClusters.filter_jets(ew, "Jet")
        assert len(jet_idxs[0]) == 0
        assert len(jet_idxs[1]) == 0
        assert len(jet_idxs[2]) == 1
        assert 1 in jet_idxs[2]
        # using selected values
        jet_idxs = CompareClusters.filter_jets(ew, "Jet", min_pt, min_ntracks)
        assert len(jet_idxs[0]) == 0
        assert len(jet_idxs[1]) == 1
        assert 1 in jet_idxs[1]
        assert len(jet_idxs[2]) == 1
        assert 1 in jet_idxs[2]


def fake_quality_width_fraction(eventWise, name, mass):
    return np.nan, len(name)/mass


def fake_detectable_comparisons(eventWise, name, jet_idx, append):
    scores = {}
    scores[name + "_Bork"] = awkward.fromiter([np.inf, np.inf, np.inf])
    scores[name + "_Quack"] = awkward.fromiter([[0.5, np.nan], [0.5, -np.inf], []])
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
    # event 1
    params['Jet_InputIdx'] = [awkward.fromiter([[0], [1, 2, 3, 4, 5]])]
    params['Jet_Parent'] += [awkward.fromiter([[-1], [-1, 1, 1, 2, 2]])]
    params['Jet_Child1'] += [awkward.fromiter([[-1], [1, 2, -1, -1, -1]])]
    params['Jet_PT'] +=     [awkward.fromiter([[50.,], [0.2, 0.1, 0., 0., .1]])]
    # event 2
    params['Jet_InputIdx'] = [awkward.fromiter([[0], [1, 2, 3]])]
    params['Jet_Parent'] += [awkward.fromiter([[-1], [-1, 1, 1]])]
    params['Jet_Child1'] += [awkward.fromiter([[-1], [1, -1, -1]])]
    params['Jet_PT'] +=     [awkward.fromiter([[50.,], [21., 0.1, 0.]])]
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



