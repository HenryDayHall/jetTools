""" methods for testing JetQuality.py """
import os
import pytest
from ipdb import set_trace as st
import unittest.mock  # for mocker
from jet_tools.test.tools import TempTestDir
import numpy as np
import numpy.testing as tst
import awkward
from jet_tools.src import JetQuality, Components

def test_sorted_masses():
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
    # only intrested in 0 1
    expected = []
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
    params['Jet_Energy'] += [[[8, 10, 10], [12, 12, 12], [5, 5, 5], [10, 10, 10]]]
    params['Jet_Phi'] += [[[0, 0, 0], [0, 0, 0], [3, 3, 3], [1, 3, 3]]]
    params['Jet_PT'] += [[[2, 5, 4], [10, 10, 10], [1, 1, 1], [7, 7, 7]]]
    params['Jet_Rapidity'] += [[[1, 1, 1], [1, 1, 1], [-1, -1, -1], [10, 10, 10]]]
    params['Jet_Tags'] += [[[1], [], [4], []]]
    params['Jet_Child1'] += [[[1, -1, -1], [4, -1, -1], [-1, 8, -1], [11, -1, -1]]]
    params['Jet_Parent'] += [[[-1, 0, 0], [-1, 3, 3], [7, -1, 7], [-1, 9, 9]]]
    expected.append(13)
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
    params['Jet_Parent'] += [[[-1, 0, 0], [-1, 3, 3], [7, -1, 7], [-1, 10, 10], [13, -1, 13]]]
    # PT order is 2, 0, 1, 4
    expected.append(15)
    prep_params = {key: awkward.fromiter([awkward.fromiter(e) for e in v]) for key, v in params.items()}
    # check this results in the predicted masses
    with TempTestDir("tst") as dir_name:
        eventWise = Components.EventWise(os.path.join(dir_name, "tmp.awkd"))
        eventWise.append(**prep_params)
        masses = JetQuality.sorted_masses(eventWise, jet_name,
                                          mass_function='highest pt pair',
                                          jet_pt_cut=0)
        tst.assert_allclose(masses, sorted(expected))
        masses = JetQuality.sorted_masses(eventWise, jet_name,
                                          mass_function='highest pt pair',
                                          jet_pt_cut=2.1)
        tst.assert_allclose(masses, sorted(expected[1:]))


def test_quality_width_fraction():
    def fake_sorted_masses(eventWise, *args, **kwargs):
        return eventWise.Sorted_masses
    jet_name = "Jet"
    with unittest.mock.patch('jet_tools.src.JetQuality.sorted_masses', new=fake_sorted_masses):
        eventWise = lambda: None  # hack to get an object that can have attributes assigned
        # an empty dataset should raise a runtime error
        eventWise.selected_index = None
        eventWise.Jet_InputIdx = []
        eventWise.Sorted_masses = []
        with pytest.raises(RuntimeError):
            found = JetQuality.quality_width(eventWise, jet_name)
        with pytest.raises(RuntimeError):
            found = JetQuality.quality_fraction(eventWise, jet_name, 25.)
        with pytest.raises(RuntimeError):
            found = JetQuality.quality_width_fracton(eventWise, jet_name, 1.)
        # with 2 masses we should have a width
        eventWise.Jet_InputIdx = list(range(2))
        eventWise.Sorted_masses = np.array([1., 2.])
        found = JetQuality.quality_width(eventWise, jet_name, fraction=0.51)
        tst.assert_allclose(found, 1/2)
        # if the window gets too small we should get an error
        with pytest.raises(RuntimeError):
            found = JetQuality.quality_width(eventWise, jet_name, fraction=0.4)
        # a window smaller than 1 should only capture 1
        found = JetQuality.quality_fraction(eventWise, jet_name, .00001)
        tst.assert_allclose(found, 1/0.5)
        f_fraction_width = JetQuality.quality_width_fracton(eventWise, jet_name, 1., 0.51,
                                                            multiplier=0.5)
        tst.assert_allclose(f_fraction_width, [1/2, 1./0.5])
        # a larger window should capture both
        found = JetQuality.quality_fraction(eventWise, jet_name, 1.)
        tst.assert_allclose(found, 1.)
        # the combined function will not thro any errors here
        f_fraction_width = JetQuality.quality_width_fracton(eventWise, jet_name, 1., 0.4)
        tst.assert_allclose(f_fraction_width, [1/2, 1.])
        # with 3 masses the width should eb the smaller of the two distances
        eventWise.Jet_InputIdx = list(range(3))
        eventWise.Sorted_masses = np.array([1., 2., 5.])
        found = JetQuality.quality_width(eventWise, jet_name, fraction=0.4)
        tst.assert_allclose(found, 1./3.)
        # the width of the window needed to capture 2 but not 3
        found = JetQuality.quality_fraction(eventWise, jet_name, 1., multiplier=3.)
        tst.assert_allclose(found, 3./2.)
        found = JetQuality.quality_fraction(eventWise, jet_name, 1., multiplier=.5)
        tst.assert_allclose(found, 3.)
        # if the width demands more masses than are avalible it should raise an error
        eventWise.Jet_InputIdx = list(range(10))
        eventWise.Sorted_masses = np.array([1., 2., 5.])
        with pytest.raises(RuntimeError):
            found = JetQuality.quality_width(eventWise, jet_name, fraction=0.4)
        # this will not raise errors
        f_fraction_width = JetQuality.quality_width_fracton(eventWise, jet_name, 1., 0.4,
                                                            multiplier=1.5)
        tst.assert_allclose(f_fraction_width, [4./10., 10./2.])



