""" A module to test the Parameter investigations module"""
import os
from ipdb import set_trace as st
import awkward
from jet_tools.test.tools import TempTestDir
from jet_tools import Components, ParameterInvestigation
import numpy.testing as tst
import numpy as np

def test_create_eigenvectors():
    params = {}
    # event 0  # no particles, should not have issues
    params['X'] = [awkward.fromiter([])]
    params['JetInputs_SourceIdx'] = [awkward.fromiter([])]
    params['JetInputs_Energy'] = [awkward.fromiter([])]
    params['JetInputs_Px'] = [awkward.fromiter([])]
    params['JetInputs_Py'] = [awkward.fromiter([])]
    params['JetInputs_Pz'] = [awkward.fromiter([])]
    # event 1  # just one particle should still work
    params['X'] += [awkward.fromiter(np.arange(1))]
    params['JetInputs_SourceIdx'] += [awkward.fromiter(np.arange(1))]
    params['JetInputs_Energy'] += [awkward.fromiter([30.])]
    params['JetInputs_Px'] += [awkward.fromiter([3.])]
    params['JetInputs_Py'] += [awkward.fromiter([3.])]
    params['JetInputs_Pz'] += [awkward.fromiter([3.])]
    # event 2  # two totally seperated particles shoudl produce totaly sperated results
    params['X'] += [awkward.fromiter(np.arange(1))]
    params['JetInputs_SourceIdx'] += [awkward.fromiter(np.arange(1))]
    params['JetInputs_Energy'] += [awkward.fromiter([30., 40.])]
    params['JetInputs_Px'] += [awkward.fromiter([3., 3.])]
    params['JetInputs_Py'] += [awkward.fromiter([3., -3.])]
    params['JetInputs_Pz'] += [awkward.fromiter([3., -3.])]
    # event 3  # standard event
    params['X'] += [awkward.fromiter(np.arange(6))]
    params['JetInputs_SourceIdx'] += [awkward.fromiter(np.arange(6))]
    params['JetInputs_Energy'] += [awkward.fromiter([30., 10., 20., 70., 20., 10.])]
    params['JetInputs_Px'] += [awkward.fromiter([3., 0., 2., 1., 2., -1.])]
    params['JetInputs_Py'] += [awkward.fromiter([3., 0., 2., 2., 2., 1.])]
    params['JetInputs_Pz'] += [awkward.fromiter([3., 0., 2., 0., 2., 2.])]
    # invarient_mass                      873  100  388   4859, 388, 94
    # shifted energy                       30   10   20   70 sqrt(393) sqrt(103)
    with TempTestDir("tst") as dir_name:
        eventWise = Components.EventWise(os.path.join(dir_name, "tmp.awkd"))
        eventWise.append(**params)
        Components.add_phi(eventWise, "JetInputs")
        Components.add_PT(eventWise, "JetInputs")
        Components.add_rapidity(eventWise, "JetInputs")
        jet_params = dict(CutoffDistance=1.)
        values, vectors = ParameterInvestigation.create_eigenvectors(eventWise, jet_params)
        # first event was empty
        assert len(values[0]) == 0
        assert len(vectors[0]) == 0
        # second event only had one object
        assert len(values[1]) == 1
        assert len(vectors[1]) == 1
        # third event should be totaly seperated
        assert len(values[2]) == 2
        assert len(vectors[2]) == 2
        tst.assert_allclose(vectors[2][0, 1], 0)
        tst.assert_allclose(vectors[2][1, 0], 0)
        # forth event is regular
        assert len(values[3]) == 5
        assert len(vectors[3]) == 6
        assert len(vectors[3][0]) == 5


def test_label_parings():
    # will need BQuarkIdx, Children, Parents, X, JetInputs_SourceIdx
    params = {}
    # event 0  # no particles, should not have issues
    params['X'] = [awkward.fromiter([])]
    params['JetInputs_SourceIdx'] = [awkward.fromiter([])]
    params['BQuarkIdx'] = [awkward.fromiter([])]
    params['Children'] = [awkward.fromiter([])]
    params['Parents'] = [awkward.fromiter([])]
    # event 1  # no jet inputs
    params['X'] += [awkward.fromiter([3,4,5])]
    params['JetInputs_SourceIdx'] += [awkward.fromiter([])]
    params['BQuarkIdx'] += [awkward.fromiter([0])]
    params['Children'] += [awkward.fromiter([[1], [2], []])]
    params['Parents'] += [awkward.fromiter([[], [0], [1]])]
    # event 2 all b decendents, but only one jet input
    params['X'] += [awkward.fromiter([3,4,5])]
    params['JetInputs_SourceIdx'] += [awkward.fromiter([2])]
    params['BQuarkIdx'] += [awkward.fromiter([0])]
    params['Children'] += [awkward.fromiter([[1], [2], []])]
    params['Parents'] += [awkward.fromiter([[], [0], [1]])]
    # event 3 no b decendents
    params['X'] += [awkward.fromiter([3,4,5])]
    params['JetInputs_SourceIdx'] += [awkward.fromiter([2])]
    params['BQuarkIdx'] += [awkward.fromiter([])]
    params['Children'] += [awkward.fromiter([[1], [2], []])]
    params['Parents'] += [awkward.fromiter([[], [0], [1]])]
    # event 3 some b decendents
    params['X'] += [awkward.fromiter([3,4,5,1,1,1])]
    params['JetInputs_SourceIdx'] += [awkward.fromiter([2, 4, 5])]
    params['BQuarkIdx'] += [awkward.fromiter([3])]
    params['Children'] += [awkward.fromiter([[1], [2], [], [4, 5], [], []])]
    params['Parents'] += [awkward.fromiter([[], [0], [1], [], [3], [3]])]
    with TempTestDir("tst") as dir_name:
        eventWise = Components.EventWise(os.path.join(dir_name, "tmp.awkd"))
        eventWise.append(**params)
        labels = ParameterInvestigation.label_parings(eventWise)
        # first 2 should contain nothing
        assert len(labels[0]) == 0
        assert len(labels[1]) == 0
        # the 2nd has only one particle, it is in a b-shower
        assert len(labels[2]) == 1
        assert labels[2][0,0]  # should be true
        # event 3 has only one non b
        assert len(labels[3]) == 1
        assert not labels[3][0,0]
        # event 4 has a mix
        assert len(labels[4]) == 3
        expected = np.array([[False, False, False],
                             [False, True, True],
                             [False, True, True]])
        assert np.all(expected == labels[4])


def test_closest_relative():
    # need X, JetInputs_SourceIdx, Children
    params = {}
    # event 0  # no particles, should not have issues
    params['X'] = [awkward.fromiter([])]
    params['JetInputs_SourceIdx'] = [awkward.fromiter([])]
    params['Children'] = [awkward.fromiter([])]
    params['Parents'] = [awkward.fromiter([])]
    # event 1  # no jet inputs
    params['X'] += [awkward.fromiter([3,4,5])]
    params['JetInputs_SourceIdx'] += [awkward.fromiter([])]
    params['Children'] += [awkward.fromiter([[1], [2], []])]
    params['Parents'] += [awkward.fromiter([[], [0], [1]])]
    # event 2 # only one jet input
    params['X'] += [awkward.fromiter([3,4,5])]
    params['JetInputs_SourceIdx'] += [awkward.fromiter([2])]
    params['Children'] += [awkward.fromiter([[1], [2], []])]
    params['Parents'] += [awkward.fromiter([[], [0], [1]])]
    # event 3 no b decendents
    params['X'] += [awkward.fromiter([3,4,5])]
    params['JetInputs_SourceIdx'] += [awkward.fromiter([1, 2])]
    params['Children'] += [awkward.fromiter([[1,2], [], []])]
    params['Parents'] += [awkward.fromiter([[], [0], [0]])]
    # event 3 some b decendents
    params['X'] += [awkward.fromiter([3,4,5,1,1,1])]
    params['JetInputs_SourceIdx'] += [awkward.fromiter([2, 4, 5])]
    params['Children'] += [awkward.fromiter([[1, 5], [2, 3], [], [4, 5], [], []])]
    params['Parents'] += [awkward.fromiter([[], [0], [1], [1], [3], [3]])]
    with TempTestDir("tst") as dir_name:
        eventWise = Components.EventWise(os.path.join(dir_name, "tmp.awkd"))
        eventWise.append(**params)
        relatives = ParameterInvestigation.closest_relative(eventWise)
        # first 2 should contain nothing
        assert len(relatives[0]) == 0
        assert len(relatives[1]) == 0
        # the 2nd has only one particle
        assert len(relatives[2]) == 1
        assert relatives[2][0,0] == 0
        # event 3 has two particles
        assert len(relatives[3]) == 2
        expected = np.array([[0, 2],
                             [2, 0]])
        assert np.all(relatives[3] == expected)
        # event 4 has more
        assert len(relatives[4]) == 3
        expected = np.array([[0, 3, 3],
                             [3, 0, 2],
                             [3, 2, 0]])
        assert np.all(expected == relatives[4]), relatives[4]


def test_get_seperations():
    # start with an empty event
    eigenvalues = [np.empty(0)]
    eigenvectors = [np.empty((0, 0))]
    # try somethign with one particle and 5 vectors
    eigenvalues += [np.ones(5)*2]
    eigenvectors += [np.ones((1, 5))]
    eigenvalues += [np.zeros(5)]
    eigenvectors += [np.zeros((1, 5))]
    # try two particles and one vector
    eigenvalues += [np.ones(1)*2]
    eigenvectors += [np.array([[0.], [1.]])]
    eigenvalues += [np.ones(1)]
    eigenvectors += [np.array([[0.], [0.]])]
    # run the function
    seperations = ParameterInvestigation.get_seperations(eigenvectors, eigenvalues)
    num_metrics = len(ParameterInvestigation.eig_metric_names)
    half_metrics = int(num_metrics/2)
    # first one should be empty
    assert seperations[0].shape == (num_metrics, 0, 0)
    # next one should be just one 0
    assert seperations[1].shape == (num_metrics, 1, 1)
    assert seperations[2].shape == (num_metrics, 1, 1)
    tst.assert_allclose(seperations[1].flatten(), 0.)
    tst.assert_allclose(seperations[2][0, 0], 0.)
    # next is two particles, the first should be seperated
    assert seperations[3].shape == (num_metrics, 2, 2)
    # make the nan values real so they don't intere
    mask = ~np.isnan(seperations[3][:, 0, 1])
    assert np.all(seperations[3][:, 0, 1][mask] > 0.)
    mask = ~np.isnan(seperations[3][:, 1, 0])
    assert np.all(seperations[3][:, 1, 0][mask] > 0.)
    tst.assert_allclose(seperations[3][:, 0, 0], 0)
    tst.assert_allclose(seperations[3][:, 1, 1], 0)
    # maek the nans 0, becuase they don't matter
    seperations[3][np.isnan(seperations[3])] = 0
    # also, the latter half of the seperations ( where the eigenspace is normed)
    # should be less seperated than the first half
    unnormed = seperations[3][:half_metrics]
    normed = seperations[3][half_metrics:]
    assert np.any(unnormed > normed)  # at least euclidien should be diferent
    assert np.all(unnormed >= normed)  # nothing should be getting futher appart
    assert seperations[4].shape == (num_metrics, 2, 2)
    # maek the nans 0, becuase they don't matter
    seperations[4][np.isnan(seperations[4])] = 0
    tst.assert_allclose(seperations[4][:, 1, 0], 0)
    tst.assert_allclose(seperations[4][:, 0, 1], 0)
    tst.assert_allclose(seperations[4][:, 0, 0], 0)
    tst.assert_allclose(seperations[4][:, 1, 1], 0)


def test_label_crossings():
    # start by making labels
    # will need BQuarkIdx, Children, Parents, X, JetInputs_SourceIdx
    params = {}
    # event 0  # no particles, should not have issues
    params['X'] = [awkward.fromiter([])]
    params['JetInputs_SourceIdx'] = [awkward.fromiter([])]
    params['BQuarkIdx'] = [awkward.fromiter([])]
    params['Children'] = [awkward.fromiter([])]
    params['Parents'] = [awkward.fromiter([])]
    # event 1  # no jet inputs
    params['X'] += [awkward.fromiter([3,4,5])]
    params['JetInputs_SourceIdx'] += [awkward.fromiter([])]
    params['BQuarkIdx'] += [awkward.fromiter([0])]
    params['Children'] += [awkward.fromiter([[1], [2], []])]
    params['Parents'] += [awkward.fromiter([[], [0], [1]])]
    # event 2 all b decendents, but only one jet input
    params['X'] += [awkward.fromiter([3,4,5])]
    params['JetInputs_SourceIdx'] += [awkward.fromiter([2])]
    params['BQuarkIdx'] += [awkward.fromiter([0])]
    params['Children'] += [awkward.fromiter([[1], [2], []])]
    params['Parents'] += [awkward.fromiter([[], [0], [1]])]
    # event 3 no b decendents
    params['X'] += [awkward.fromiter([3,4,5])]
    params['JetInputs_SourceIdx'] += [awkward.fromiter([2])]
    params['BQuarkIdx'] += [awkward.fromiter([])]
    params['Children'] += [awkward.fromiter([[1], [2], []])]
    params['Parents'] += [awkward.fromiter([[], [0], [1]])]
    # event 3 some b decendents
    params['X'] += [awkward.fromiter([3,4,5,1,1,1])]
    params['JetInputs_SourceIdx'] += [awkward.fromiter([2, 4, 5])]
    params['BQuarkIdx'] += [awkward.fromiter([3])]
    params['Children'] += [awkward.fromiter([[1], [2], [], [4, 5], [], []])]
    params['Parents'] += [awkward.fromiter([[], [0], [1], [], [3], [3]])]
    with TempTestDir("tst") as dir_name:
        eventWise = Components.EventWise(os.path.join(dir_name, "tmp.awkd"))
        eventWise.append(**params)
        labels = ParameterInvestigation.label_parings(eventWise)
        crossings = ParameterInvestigation.label_crossings(labels)
        # first 2 should contain nothing
        assert len(crossings[0]) == 0
        assert len(crossings[1]) == 0
        # the 2nd has only one particle, it is not crossing
        assert len(crossings[2]) == 1
        assert not crossings[2][0,0]  # should be true
        # event 3 has only one non b
        assert len(crossings[3]) == 1
        assert not crossings[3][0,0]
        # event 4 has a mix
        assert len(crossings[4]) == 3
        expected = np.array([[False, True, True],
                             [True, False, False],
                             [True, False, False]])
        assert np.all(expected == crossings[4])


def test_physical_distances():
    # start with an empty event
    phis = [np.empty(0)]
    rapidities = [np.empty(0)]
    pts = [np.empty(0)]
    # try somethign with one particle
    phis += [np.ones(1)]
    rapidities += [np.ones(1)]
    pts += [np.ones(1)]
    exp_multipler = 0.
    phis += [np.zeros(1)]
    rapidities += [np.zeros(1)]
    pts += [np.ones(1)]
    # try two particles and one vector
    phis += [np.array([0., 1.])]
    rapidities += [np.array([0., 1.])]
    pts += [np.array([4., 4.])]
    # run the function
    exp_multipler = 0.
    distances0 = ParameterInvestigation.physical_distances(phis, rapidities,
                                                           pts, exp_multipler)
    exp_multipler = 1.
    distances1 = ParameterInvestigation.physical_distances(phis, rapidities,
                                                           pts, exp_multipler)
    exp_multipler = -1.
    distancesm1 = ParameterInvestigation.physical_distances(phis, rapidities,
                                                            pts, exp_multipler)
    num_metrics = len(ParameterInvestigation.phys_metric_names)
    half_metrics = int(num_metrics/2)
    # first one should be empty
    assert distances0[0].shape == (num_metrics, 0, 0)
    assert distances1[0].shape == (num_metrics, 0, 0)
    assert distancesm1[0].shape == (num_metrics, 0, 0)
    # next one should be just one 0
    assert distances0[1].shape == (num_metrics, 1, 1)
    assert distances0[2].shape == (num_metrics, 1, 1)
    tst.assert_allclose(distances0[1].flatten(), 0.)
    tst.assert_allclose(distances0[2][0, 0], 0.)
    assert distances1[1].shape == (num_metrics, 1, 1)
    assert distances1[2].shape == (num_metrics, 1, 1)
    tst.assert_allclose(distances1[1].flatten(), 0.)
    tst.assert_allclose(distances1[2][0, 0], 0.)
    assert distancesm1[1].shape == (num_metrics, 1, 1)
    assert distancesm1[2].shape == (num_metrics, 1, 1)
    tst.assert_allclose(distancesm1[1].flatten(), 0.)
    tst.assert_allclose(distancesm1[2][0, 0], 0.)
    # next is two particles, the first should be seperated
    assert distances0[3].shape == (num_metrics, 2, 2)
    assert distances1[3].shape == (num_metrics, 2, 2)
    assert distancesm1[3].shape == (num_metrics, 2, 2)
    # ignore the nan
    mask = ~np.isnan(distances0[3][:, 0, 1])
    assert np.all(distances0[3][:, 0, 1][mask] > 0.)
    mask = ~np.isnan(distances0[3][:, 1, 0])
    assert np.all(distances0[3][:, 1, 0][mask] > 0.)
    mask = ~np.isnan(distances1[3][:, 0, 1])
    assert np.all(distances1[3][:, 0, 1][mask] > 0.)
    mask = ~np.isnan(distances1[3][:, 1, 0])
    assert np.all(distances1[3][:, 1, 0][mask] > 0.)
    mask = ~np.isnan(distancesm1[3][:, 0, 1])
    assert np.all(distancesm1[3][:, 0, 1][mask] > 0.)
    mask = ~np.isnan(distancesm1[3][:, 1, 0])
    assert np.all(distancesm1[3][:, 1, 0][mask] > 0.)
    # all diagnoals should be 0
    tst.assert_allclose(distances0[3][:, 0, 0], 0)
    tst.assert_allclose(distances0[3][:, 1, 1], 0)
    tst.assert_allclose(distances1[3][:, 0, 0], 0)
    tst.assert_allclose(distances1[3][:, 0, 0], 0)
    tst.assert_allclose(distancesm1[3][:, 1, 1], 0)
    tst.assert_allclose(distancesm1[3][:, 1, 1], 0)
    # maek the nans 0, becuase they don't matter
    distances0[3][np.isnan(distances0[3])] = 0
    distances1[3][np.isnan(distances1[3])] = 0
    distancesm1[3][np.isnan(distancesm1[3])] = 0
    # the PT is always gt 1, so a positive exponent multiplier should
    # make the distances larger
    assert np.any(distances1[3] > distances0[3])
    assert np.all(distances1[3] >= distances0[3])
    assert np.any(distances0[3] > distancesm1[3])
    assert np.all(distances0[3] >= distancesm1[3])


def test_get_linked():
    params = {}
    # event 0  # no particles, should not have issues
    params['X'] = [awkward.fromiter([])]
    params['JetInputs_SourceIdx'] = [awkward.fromiter([])]
    params['JetInputs_Energy'] = [awkward.fromiter([])]
    params['JetInputs_Px'] = [awkward.fromiter([])]
    params['JetInputs_Py'] = [awkward.fromiter([])]
    params['JetInputs_Pz'] = [awkward.fromiter([])]
    # event 1  # just one particle should still work
    params['X'] += [awkward.fromiter([0.])]
    params['JetInputs_SourceIdx'] += [awkward.fromiter(np.arange(1))]
    params['JetInputs_Energy'] += [awkward.fromiter([30.])]
    params['JetInputs_Px'] += [awkward.fromiter([3.])]
    params['JetInputs_Py'] += [awkward.fromiter([3.])]
    params['JetInputs_Pz'] += [awkward.fromiter([3.])]
    # event 2  # two totally identical particles
    params['X'] += [awkward.fromiter([0., 0.])]
    params['JetInputs_SourceIdx'] += [awkward.fromiter(np.arange(2))]
    params['JetInputs_Energy'] += [awkward.fromiter([30., 30.])]
    params['JetInputs_Px'] += [awkward.fromiter([3., 3.])]
    params['JetInputs_Py'] += [awkward.fromiter([3., 3.])]
    params['JetInputs_Pz'] += [awkward.fromiter([3., 3.])]
    # event 3  # two totally seperated particles should get cut off by distance
    params['X'] += [awkward.fromiter([0., 0.])]
    params['JetInputs_SourceIdx'] += [awkward.fromiter(np.arange(2))]
    params['JetInputs_Energy'] += [awkward.fromiter([30., 40.])]
    params['JetInputs_Px'] += [awkward.fromiter([3., 3.])]
    params['JetInputs_Py'] += [awkward.fromiter([3., -3.])]
    params['JetInputs_Pz'] += [awkward.fromiter([3., -3.])]
    # event 4 particles progressivly futher apart
    params['X'] += [awkward.fromiter(np.arange(6).astype(float))]
    params['JetInputs_SourceIdx'] += [awkward.fromiter(np.arange(6))]
    params['JetInputs_Energy'] += [awkward.fromiter(np.arange(6)*50. + 1.)]
    params['JetInputs_Px'] += [awkward.fromiter(np.arange(6.))]
    params['JetInputs_Py'] += [awkward.fromiter(np.arange(6., 0., -1.))]
    params['JetInputs_Pz'] += [awkward.fromiter(np.arange(6)*5.)]
    with TempTestDir("tst") as dir_name:
        eventWise = Components.EventWise(os.path.join(dir_name, "tmp.awkd"))
        eventWise.append(**params)
        Components.add_phi(eventWise, "JetInputs")
        Components.add_PT(eventWise, "JetInputs")
        Components.add_rapidity(eventWise, "JetInputs")
        # first try no cutoff
        eventWise.selected_index = None
        jet_params = dict(ExpofPTPosition='input', ExpofPTMultiplier=0.,
                          PhyDistance='angular',
                          CutoffDistance=None, CutoffKNN=None)
        is_linked, percent_sparcity = ParameterInvestigation.get_linked(eventWise, jet_params)
        expected_sparsity = [np.nan, 0., 0., 0., 0.]
        tst.assert_allclose(percent_sparcity, expected_sparsity)
        for linked in is_linked:
            assert np.all(linked)
        # now try a v small distance cutoff
        eventWise.selected_index = None
        jet_params = dict(ExpofPTPosition='input', ExpofPTMultiplier=0.,
                          PhyDistance='angular',
                          CutoffDistance=0.1, CutoffKNN=None)
        is_linked, percent_sparcity = ParameterInvestigation.get_linked(eventWise, jet_params)
        expected_sparsity = [np.nan, 0., 0., .5, 5/6]
        tst.assert_allclose(percent_sparcity, expected_sparsity)
        assert len(is_linked[0]) == 0.
        assert np.all(is_linked[1].flatten() == np.array([True]))
        assert np.all(is_linked[2].flatten() == np.array([True]*4))
        assert np.all(is_linked[3] == np.array([[True, False], [False, True]]))
        # now two nearest neighbours
        eventWise.selected_index = None
        jet_params = dict(ExpofPTPosition='input', ExpofPTMultiplier=0.,
                          PhyDistance='angular',
                          CutoffKNN=2, CutoffDistance=None)
        is_linked, percent_sparcity = ParameterInvestigation.get_linked(eventWise, jet_params)
        expected4 = np.array([[True, True, True, False, False, False],
                              [True, True, True, False, False, False],
                              [True, True, True, True, False, False],
                              [False, False, True, True, True, True],
                              [False, False, False, True, True, True],
                              [False, False, False, True, True, True]])
        expected_sparsity = [np.nan, 0., 0., 0., np.sum(~expected4)/(6*6)]
        tst.assert_allclose(percent_sparcity, expected_sparsity)
        assert len(is_linked[0]) == 0.
        assert np.all(is_linked[1].flatten() == np.array([True]))
        assert np.all(is_linked[2].flatten() == np.array([True]*4))
        assert np.all(is_linked[3].flatten() == np.array([True]*4))
        assert np.all(is_linked[4] == expected4)


def test_get_isolated():
    is_linked = []
    labels = []
    expected_isolated = []
    expected_percent = []
    # 0 should not choke on a n empty set
    empty = np.array([], dtype=bool).reshape((0, 0))
    is_linked.append(empty)
    labels.append(empty)
    expected_isolated.append(empty.flatten())
    expected_percent.append(np.nan)
    # 1 one particle
    is_linked.append(np.array([[True]]))
    labels.append(np.array([[True]]))
    expected_isolated.append(np.array([False]))
    expected_percent.append(0.)
    # 2 if it's not labeled it shouldn't matter if it's linked
    is_linked.append(np.array([[False]]))
    labels.append(np.array([[False]]))
    expected_isolated.append(np.array([False]))
    expected_percent.append(np.nan)
    # 3 if it's labeled and not linked to itself then it is isolated
    is_linked.append(np.array([[False]]))
    labels.append(np.array([[True]]))
    expected_isolated.append(np.array([True]))
    expected_percent.append(1.)
    # 4 two particles
    is_linked.append(np.array([[True, True],
                               [True, True]]))
    labels.append(np.array([[True, True],
                            [True, True]]))
    expected_isolated.append(np.array([False, False]))
    expected_percent.append(0.)
    # 5 not linked
    is_linked.append(np.array([[True, False],
                               [False, True]]))
    labels.append(np.array([[True, True],
                            [True, True]]))
    expected_isolated.append(np.array([True, True]))
    expected_percent.append(1.)
    # 6 not linked, only one in b jet
    is_linked.append(np.array([[True, False],
                               [False, True]]))
    labels.append(np.array([[False, False],
                            [False, True]]))
    expected_isolated.append(np.array([False, False]))
    expected_percent.append(0.)
    # 7 not linked, not in b jet
    is_linked.append(np.array([[True, False],
                               [False, True]]))
    labels.append(np.array([[False, False],
                            [False, False]]))
    expected_isolated.append(np.array([False, False]))
    expected_percent.append(np.nan)
    # 8 not labeled
    is_linked.append(np.array([[True, True],
                               [True, True]]))
    labels.append(np.array([[False, False],
                            [False, False]]))
    expected_isolated.append(np.array([False, False]))
    expected_percent.append(np.nan)
    # 9 complex event
    is_linked.append(np.array([[True, True, False],
                               [True, True, False],
                               [False, False, True]]))
    labels.append(np.array([[False, False, False],
                            [False, True, True],
                            [False, True, True]]))
    expected_isolated.append(np.array([False, True, True]))
    expected_percent.append(1.)
    # 10 complex event 2
    is_linked.append(np.array([[True, True, False],
                               [True, True, False],
                               [False, False, True]]))
    labels.append(np.array([[True, True, False],
                            [True, True, True],
                            [False, True, True]]))
    expected_isolated.append(np.array([False, False, True]))
    expected_percent.append(1/3)
    found = ParameterInvestigation.get_isolated(is_linked, labels)
    for i, (found_isolated, found_percent) in enumerate(zip(*found)):
        tst.assert_allclose(found_isolated, expected_isolated[i], err_msg=f"isolated {i} not as exspected")
        tst.assert_allclose(found_percent, expected_percent[i], err_msg=f"percent {i} not as exspected")

    


    
