""" A module to test the Parameter investigations module"""
from ipdb import set_trace as st
import awkward
from tools import TempTestDir
from tree_tagger import Components, ParameterInvestigation
import numpy.testing as tst
import numpy as np

# Physical distance ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Eigenspace distance ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
        eventWise = Components.EventWise(dir_name, "tmp.awkd")
        eventWise.append(**params)
        Components.add_phi(eventWise, "JetInputs")
        Components.add_PT(eventWise, "JetInputs")
        Components.add_rapidity(eventWise, "JetInputs")
        jet_params = dict(AffinityCutoff=('distance',1.))
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
        eventWise = Components.EventWise(dir_name, "tmp.awkd")
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
        eventWise = Components.EventWise(dir_name, "tmp.awkd")
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
    num_metrics = len(ParameterInvestigation.metric_names)
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

