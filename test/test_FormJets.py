from numpy import testing as tst
import pytest
import warnings
import os
from ipdb import set_trace as st
import numpy as np
from tree_tagger import Components, FormJets
from tools import TempTestDir
from test_Components import AwkdArrays
import awkward
import itertools
import scipy.spatial


def test_knn():
    # zero points should not create an error
    points = []
    expected = np.array([]).reshape((0, 0))
    distances = np.array([]).reshape((0, 0))
    tst.assert_allclose(FormJets.knn(distances, 1),  expected)
    # one point is always it's own nearest neigbour
    points = [1]
    expected = np.array([[True]])
    distances = scipy.spatial.distance.pdist(np.array(points).reshape((-1, 1)))
    distances = scipy.spatial.distance.squareform(distances)
    tst.assert_allclose(FormJets.knn(distances, 1),  expected)
    # even if no neighbours are requested
    distances = scipy.spatial.distance.pdist(np.array(points).reshape((-1, 1)))
    distances = scipy.spatial.distance.squareform(distances)
    tst.assert_allclose(FormJets.knn(distances, 0),  expected)
    # two points
    points = [1, 2]
    expected = np.full((2,2), True)
    distances = scipy.spatial.distance.pdist(np.array(points).reshape((-1, 1)))
    distances = scipy.spatial.distance.squareform(distances)
    tst.assert_allclose(FormJets.knn(distances, 1),  expected)
    # without neighbours
    expected = np.full((2,2), False)
    np.fill_diagonal(expected, True)
    distances = scipy.spatial.distance.pdist(np.array(points).reshape((-1, 1)))
    distances = scipy.spatial.distance.squareform(distances)
    tst.assert_allclose(FormJets.knn(distances, 0),  expected)
    # three points
    points = [3, 1, 2.5]
    distances = scipy.spatial.distance.pdist(np.array(points).reshape((-1, 1)))
    distances = scipy.spatial.distance.squareform(distances)
    expected = np.full((3,3), False)
    np.fill_diagonal(expected, True)
    tst.assert_allclose(FormJets.knn(distances, 0),  expected)
    expected = np.array([[True, False, True],
                         [False, True, True],
                         [True, True, True]])
    tst.assert_allclose(FormJets.knn(distances, 1),  expected)
    expected = np.full((3,3), True)
    tst.assert_allclose(FormJets.knn(distances, 2),  expected)


# Consider adding tests for jets created with pseudorapidity instead of rapidity.
# if you ever use that functionality again....

class SimpleClusterSamples:
    config_1 = {'DeltaR': 1., 'ExpofPTMultiplier': 0}
    config_2 = {'DeltaR': 1., 'ExpofPTMultiplier': 1}
    config_3 = {'DeltaR': 1., 'ExpofPTMultiplier': -1}
    config_4 = {'DeltaR': .4, 'ExpofPTMultiplier': 0}
    config_5 = {'DeltaR': .4, 'ExpofPTMultiplier': 1}
    config_6 = {'DeltaR': .4, 'ExpofPTMultiplier': -1}
    # there is no garantee on left right child order, or global order of pseudojets
    unitless = True   # do we work with a unitless version of distance
    empty_inp = {'ints': np.array([]).reshape((-1, 5)), 'floats': np.array([]).reshape((-1, 8))}
    one_inp = {'ints': np.array([[0, -1, -1, -1, -1]]),
               'floats': np.array([[1., 0., 0., 1., 1., 0., 0., 0.]])}
    two_degenerate = {'ints': np.array([[0, -1, -1, -1, -1],
                                        [1, -1, -1, -1, -1]]),
                      'floats': np.array([[1., 0., 0., 1., 1., 0., 0., 0.],
                                          [1., 0., 0., 1., 1., 0., 0., 0.]])}
    degenerate_join = {'ints': np.array([[0, 2, -1, -1, -1],
                                         [1, 2, -1, -1, -1],
                                         [2, -1, 0, 1, 0]]),
                       'floats': np.array([[1., 0., 0., 1., 1., 0., 0., 0.],
                                           [1., 0., 0., 1., 1., 0., 0., 0.],
                                           [2., 0., 0., 2., 2., 0., 0., 0.]])}
    two_close = {'ints': np.array([[0, -1, -1, -1, -1],
                                   [1, -1, -1, -1, -1]]),
                 'floats': np.array([[1., 0., 0., 1., 1., 0., 0., 0.],
                                     [1., 0., 0.1, 1., np.cos(0.1), np.sin(0.1), 0., 0.]])}
    close_join = {'ints': np.array([[0, 2, -1, -1, -1],
                                    [1, 2, -1, -1, -1],
                                    [2, -1, 0, 1, 0]]),
                  'floats': np.array([[1., 0., 0., 1., 1., 0., 0., 0.],
                                      [1., 0., 0.1, 1., np.cos(0.1), np.sin(0.1), 0., 0.],
                                      [2.*np.cos(0.05), 0., 0.05, 2., 1. + np.cos(0.1), np.sin(0.1), 0., 0.1]])}
    two_oposite = {'ints': np.array([[0, -1, -1, -1, -1],
                                     [1, -1, -1, -1, -1]]),
                   'floats': np.array([[1., 0., 0., 1., 1., 0., 0., 0.],
                                       [1., 0., np.pi, 1., -1., 0., 0., 0.]])}
    @classmethod
    def get_invarient_mass(cls, inputs):
        invarient_mass = 1.
        if cls.unitless:
                                                           # energy = col3
            invarient_mass = np.sqrt(np.sum(inputs['floats'][:, 3])**2
                                                           # pt = col0,  pz = col6
                                     - np.sum(np.sum(inputs['floats'][:, [0, 6]], axis=0)**2))
        if invarient_mass == 0:
            invarient_mass = 1.
        return invarient_mass

    @classmethod
    def pair_production(cls, config):
        randoms = np.random.rand(6)
        max_pi = 100.
        two = {'ints': np.array([[0, -1, -1, -1, -1],
                                 [1, -1, -1, -1, -1]]),
               'floats': np.array([[0., 0., 0., 0., randoms[0]*max_pi, randoms[1]*max_pi, randoms[2]*max_pi, 0.],
                                   [0., 0., 0., 0., randoms[3]*max_pi, randoms[4]*max_pi, randoms[5]*max_pi, 0.]])}
        cls.fill_angular(two['floats'][0])
        cls.fill_angular(two['floats'][1])
        return cls.outcome(config, two)

    @classmethod
    def phi_split(cls, config, phi2):
        rapidity = 0.
        pt = 1.
        energy = 1.1
        phi1 = 0.
        two = {'ints': np.array([[0, -1, -1, -1, -1],
                                 [1, -1, -1, -1, -1]]),
               'floats': np.array([[pt, rapidity, phi1, energy, 0., 0., 0., 0.],
                                   [pt, rapidity, phi2, energy, 0., 0., 0., 0.]])}
        cls.fill_linear(two['floats'][0])
        cls.fill_linear(two['floats'][1])
        return cls.outcome(config, two)


    @classmethod
    def outcome(cls, config, two):
        # get the angular displacement by converting to unit vectors and back
        phi_1 = two['floats'][0, 2]
        phi_2 = two['floats'][1, 2]
        angle = abs(phi_1 - phi_2)%(2*np.pi)
        #vec_1 = np.array([np.cos(phi_1), np.sin(phi_1)])
        #vec_2 = np.array([np.cos(phi_2), np.sin(phi_2)])
        #angle = np.arccos(np.clip(np.dot(vec_1, vec_2), -1., 1.))
        invar_mass = cls.get_invarient_mass(two)
        exponent = 2*config['ExpofPTMultiplier']
        distance2 = np.min(np.power(two['floats'][:, 0], exponent)) * \
                   ((two['floats'][0, 1] - two['floats'][1, 1])**2 + angle**2) *\
                   invar_mass**-exponent
        beam_distance2 = (config['DeltaR']**2)*np.min(np.power(two['floats'][:, 0], exponent)) *\
                        invar_mass**-exponent
        if distance2 > beam_distance2:
            #print(f"angle={angle}, exponent={exponent}, invar_mass={invar_mass}, distance2={distance2}, beam_distance2={beam_distance2}")
            # don't join
            return two, two
        else:
            # form the joined row
            joined_floats = two['floats'][0] + two['floats'][1]
            cls.fill_angular(joined_floats, False)
            joined_two = {'ints': np.array([[0, 2, -1, -1, -1],
                                            [1, 2, -1, -1, -1],
                                            [2, -1, 0, 1, 0]]),
                          'floats': np.vstack((two['floats'], joined_floats))}
            joined_two['floats'][-1, -1] = np.sqrt(distance2)
            return two, joined_two

    @classmethod
    def fill_angular(cls, floats, change_energy=True):
        """ given floast that contain px py pz calculate the other values"""
        px, py, pz = floats[4:7]
        if change_energy is True:
            energy = (1 + np.random.rand())*np.linalg.norm([px, py, pz], 2)
        else:
            energy = floats[3]
        floats[0] = np.linalg.norm([px, py], 2)
        if energy == pz and px == 0 and py == 0:
            floats[1] = np.Inf
        else:
            floats[1] = 0.5*np.log((energy + pz)/(energy - pz))
        floats[2] = np.arctan2(py, px)
        floats[3] = energy

    @classmethod
    def fill_linear(cls, floats):
        """ given floast that contain pt rapidity phi calculate the other values"""
        pt, rapidity, phi, e = floats[0:4]
        # rapidity
        # rapidity = 0.5 log((e+pz)/(e-pz))
        # or on the light cone
        # |rapidity| = 0.5 log((pt^2 + m^2)/(e - |pz|)^2)
        # exp(2 |rapidity|) = (e + pz)/(e-|pz|)
        # e ( exp(2 |rapidity|) - 1) = |pz| (1+exp(2 |rapidity|))
        # |pz| = e (exp(2 |rapidity|) - 1)/(1 + exp( 2| rapidity|)
        floats[4] = pt*np.cos(phi)
        floats[5] = pt*np.sin(phi)
        exp_2rap = np.exp(2* np.abs(rapidity))
        mag_pz = e * (exp_2rap - 1)/(exp_2rap + 1)
        floats[6] = np.sign(rapidity) * np.abs(mag_pz)

    @classmethod
    def match_ints(cls, ints1, ints2):
        """ find out if one array of ints in infact a reshuffle of the other
        get shuffle order. Is independent of the order of child 1 and 2"""
        ints1 = np.array(ints1)
        ints2 = np.array(ints2)
        if ints1.shape != ints2.shape:
            assert False, f"Ints dont have the same shape; {ints1.shape}, {ints2.shape}"
        child_columns = slice(2, 4)
        ints1.T[child_columns] = np.sort(ints1.T[child_columns], axis=0)
        ints2.T[child_columns] = np.sort(ints2.T[child_columns], axis=0)
        avalible = list(range(ints1.shape[0]))  # make sure we match one to one
        order = []
        for row in ints2:
            matches = np.where(np.all(ints1[avalible] == row, axis=1))[0]
            if len(matches) == 0:
                assert False, f"Ints don't match. \n{ints1}\n~~~\n{ints2}"
            picked = avalible.pop(matches[0])
            order.append(picked)
        return order

    @classmethod
    def match_ints_floats(cls, ints1, floats1, ints2, floats2, compare_distance=True, distance_modifier=1.):
        ints_order = cls.match_ints(ints1, ints2)
        floats1 = np.array(floats1)[ints_order]
        floats2 = np.array(floats2)
        if not compare_distance:
            floats1 = floats1[:, :-1]
            floats2 = floats2[:, :-1]
        else:
            floats2[:, -1] *= distance_modifier
        tst.assert_allclose(floats1, floats2, atol=0.0005, err_msg="Floats don't match")


def set_JetInputs(eventWise, floats):
    columns = [name.replace("Pseudojet", "JetInputs") for name in FormJets.PseudoJet.float_columns
               if "Distance" not in name]
    if len(floats):
        contents = {name: awkward.fromiter([floats[:, i]]) for i, name in enumerate(columns)}
    else:
        contents = {name: awkward.fromiter([[]]) for i, name in enumerate(columns)}
    columns.append("JetInputs_SourceIdx")
    contents["JetInputs_SourceIdx"] = awkward.fromiter([np.arange(len(floats))])
    eventWise.append(**contents)


def clustering_algorithm(empty_ew, make_pseudojets, compare_distance=False):
    # for the randomly places components, accept some error for floating point
    n_random_tries = 20
    n_acceptable_fails = 1
    config_list = [getattr(SimpleClusterSamples, f"config_{i}") for i in range(1, 7)]
    for config in config_list:
        # an empty set of ints should safely return an empty pseudojet
        empty_pseudojet = make_pseudojets(empty_ew, config['DeltaR'], config['ExpofPTMultiplier'],
                                             ints = SimpleClusterSamples.empty_inp['ints'],
                                             floats = SimpleClusterSamples.empty_inp['floats'])
        assert len(empty_pseudojet._ints) == 0
        assert len(empty_pseudojet._floats) == 0
        jets = empty_pseudojet.split()
        assert len(jets) == 0
        # one track should return one pseudojet
        one_pseudojet = make_pseudojets(empty_ew, config['DeltaR'], config['ExpofPTMultiplier'],
                                           ints = SimpleClusterSamples.one_inp['ints'],
                                           floats = SimpleClusterSamples.one_inp['floats'])
        SimpleClusterSamples.match_ints_floats(one_pseudojet._ints, one_pseudojet._floats,
                                               SimpleClusterSamples.one_inp['ints'],
                                               SimpleClusterSamples.one_inp['floats'],
                                               compare_distance=compare_distance)
        jets = one_pseudojet.split()
        assert len(jets) == 1
        SimpleClusterSamples.match_ints_floats(jets[0]._ints, jets[0]._floats,
                                               SimpleClusterSamples.one_inp['ints'],
                                               SimpleClusterSamples.one_inp['floats'],
                                               compare_distance=compare_distance)
        # two tracks degenerate should join
        two_pseudojet = make_pseudojets(empty_ew, config['DeltaR'], config['ExpofPTMultiplier'],
                                        ints=SimpleClusterSamples.two_degenerate['ints'],
                                        floats=SimpleClusterSamples.two_degenerate['floats'])
        SimpleClusterSamples.match_ints_floats(two_pseudojet._ints, two_pseudojet._floats,
                                               SimpleClusterSamples.degenerate_join['ints'],
                                               SimpleClusterSamples.degenerate_join['floats'],
                                               compare_distance=compare_distance)
        jets = two_pseudojet.split()
        assert len(jets) == 1
        SimpleClusterSamples.match_ints_floats(jets[0]._ints, jets[0]._floats,
                                               SimpleClusterSamples.degenerate_join['ints'],
                                               SimpleClusterSamples.degenerate_join['floats'],
                                               compare_distance=compare_distance)
        # two tracks close together should join
        modifier = SimpleClusterSamples.get_invarient_mass(SimpleClusterSamples.two_close)**-(2*config['ExpofPTMultiplier'])
        two_pseudojet = make_pseudojets(empty_ew, config['DeltaR'], config['ExpofPTMultiplier'],
                                           ints=SimpleClusterSamples.two_close['ints'],
                                           floats=SimpleClusterSamples.two_close['floats'])
        SimpleClusterSamples.match_ints_floats(two_pseudojet._ints, two_pseudojet._floats,
                                               SimpleClusterSamples.close_join['ints'],
                                               SimpleClusterSamples.close_join['floats'],
                                               compare_distance=compare_distance,
                                               distance_modifier=modifier)
        jets = two_pseudojet.split()
        assert len(jets) == 1
        SimpleClusterSamples.match_ints_floats(jets[0]._ints, jets[0]._floats,
                                               SimpleClusterSamples.close_join['ints'],
                                               SimpleClusterSamples.close_join['floats'],
                                               compare_distance=compare_distance)
        # two tracks far apart should not join
        two_pseudojet = make_pseudojets(empty_ew, config['DeltaR'], config['ExpofPTMultiplier'],
                                           ints=SimpleClusterSamples.two_oposite['ints'],
                                           floats=SimpleClusterSamples.two_oposite['floats'])
        SimpleClusterSamples.match_ints_floats(two_pseudojet._ints, two_pseudojet._floats,
                                               SimpleClusterSamples.two_oposite['ints'],
                                               SimpleClusterSamples.two_oposite['floats'],
                                               compare_distance=compare_distance)
        jets = two_pseudojet.split()
        assert len(jets) == 2
        regroup_ints = np.vstack([jets[0]._ints, jets[1]._ints])
        regroup_floats = np.vstack([jets[0]._floats, jets[1]._floats])
        SimpleClusterSamples.match_ints_floats(regroup_ints, regroup_floats,
                                               SimpleClusterSamples.two_oposite['ints'],
                                               SimpleClusterSamples.two_oposite['floats'],
                                               compare_distance=compare_distance)
        # split two tracks in phi
        phis = np.linspace(-np.pi, np.pi, 11)
        for phi in phis:
            start, expected_end = SimpleClusterSamples.phi_split(config, phi)
            pseudojets = make_pseudojets(empty_ew, config['DeltaR'], config['ExpofPTMultiplier'],
                                            ints = start['ints'], floats = start['floats'])
            jets = pseudojets.split()
            end_ints = np.vstack([j._ints for j in jets])
            end_floats = np.vstack([j._floats for j in jets])
            if len(end_ints) == len(expected_end['ints']):
                SimpleClusterSamples.match_ints_floats(pseudojets._ints, pseudojets._floats,
                                                       end_ints, end_floats,
                                                       compare_distance=compare_distance)
            else:
                assert False, f"phi={phi} to 0. didn't behave as expected"
            


        # now create some random pairs and check the clustering algroithm returns what is expected
        fails = 0
        for i in range(n_random_tries):
            start, expected_end = SimpleClusterSamples.pair_production(config)
            modifier = SimpleClusterSamples.get_invarient_mass(start)**-(2*config['ExpofPTMultiplier'])
            pseudojets = make_pseudojets(empty_ew, config['DeltaR'], config['ExpofPTMultiplier'],
                                            ints=start['ints'], floats=start['floats'])
            jets = pseudojets.split()
            end_ints = np.vstack([j._ints for j in jets])
            end_floats = np.vstack([j._floats for j in jets])
            # check the split dosn't chagne things
            SimpleClusterSamples.match_ints_floats(pseudojets._ints, pseudojets._floats,
                                                   end_ints, end_floats, distance_modifier=1)
            if len(end_ints) == len(expected_end['ints']):
                SimpleClusterSamples.match_ints_floats(end_ints, end_floats,
                                                       expected_end['ints'], expected_end['floats'],
                                                       compare_distance=False)
            else:
                st()
                fails += 1
                print("Failed to join/incorrectly joined")
                if fails > n_acceptable_fails:
                    assert False, f"{fails} out of {i} incorrect clusters from homejet"


# running the whole cluster process from start to finish to see if we get what is expected
def test_Traditional():
    with TempTestDir("pseudojet") as dir_name:
        # there are two ways to construct a pseudojet
        # 1. telling it ints and floats as constructor arguments
        # 2. letting it read ints and floats from an eventwise object
        # We start with the first method and do checks on the behavior of pseudojet
        empty_name = "empty.awkd"
        empty_path = os.path.join(dir_name, empty_name)
        empty_ew = Components.EventWise(dir_name, empty_name)
        empty_ew.selected_index = 0
        # method 1
        def make_jets1(eventWise, DeltaR, ExpofPTMultiplier, ints, floats):
            pseudojets = FormJets.Traditional(eventWise, DeltaR=DeltaR, ExpofPTMultiplier=ExpofPTMultiplier, ints_floats=(ints, floats))
            pseudojets.assign_parents()
            return pseudojets
        clustering_algorithm(empty_ew, make_jets1)
        # method 2
        def make_jets2(eventWise, DeltaR, ExpofPTMultiplier, ints, floats):
            set_JetInputs(eventWise, floats)
            eventWise.selected_index = 0
            pseudojets = FormJets.Traditional(eventWise, DeltaR=DeltaR, ExpofPTMultiplier=ExpofPTMultiplier)
            pseudojets.assign_parents()
            return pseudojets
        clustering_algorithm(empty_ew, make_jets2)
        # test the save method
        test_inp = SimpleClusterSamples.one_inp
        name = "TestA"
        test_jet = FormJets.Traditional(empty_ew, DeltaR=1., ExpofPTMultiplier=-1.,
                                        ints_floats=(test_inp['ints'], test_inp['floats']), jet_name=name)
        FormJets.Traditional.write_event([test_jet], name, event_index=2)
        test_jet.check_params(empty_ew)  # required to put in hyperparameters
        # read out again
        jets = FormJets.Traditional.multi_from_file(empty_path, event_idx=2, jet_name=name)
        # expect only one jet
        assert len(jets) == 1
        SimpleClusterSamples.match_ints_floats(test_inp['ints'], test_inp['floats'], jets[0]._ints, jets[0]._floats)
        # test collective properties
        test_inp = SimpleClusterSamples.empty_inp
        empty_ew.selected_index = 0
        test_jet = FormJets.Traditional(empty_ew, DeltaR=1., ExpofPTMultiplier=-1., ints_floats=(test_inp['ints'], test_inp['floats']))
        expected_summaries = np.full(10, np.nan)
        found_summaries = np.array([test_jet.PT, test_jet.Rapidity, test_jet.Phi, test_jet.Energy,
                                    test_jet.Px, test_jet.Py, test_jet.Pz, test_jet.JoinDistance,
                                    test_jet.Pseudorapidity, test_jet.Theta])
        tst.assert_allclose(expected_summaries, found_summaries)
        test_inp = SimpleClusterSamples.one_inp
        test_jet = FormJets.Traditional(empty_ew, DeltaR=1., ExpofPTMultiplier=-1., ints_floats=(test_inp['ints'], test_inp['floats']))
        test_jet.assign_parents()
        test_jets = test_jet.split()
        for expected, jet in zip(test_inp['floats'], test_jets):
            theta = np.arctan2(expected[0], expected[6])
            pseudorapidity = -np.log(np.tan(theta/2))
            expected_summaries = list(expected) + [pseudorapidity, theta]
            found_summaries = np.array([jet.PT, jet.Rapidity, jet.Phi, jet.Energy,
                                        jet.Px, jet.Py, jet.Pz, jet.JoinDistance,
                                        jet.Pseudorapidity, jet.Theta])
            tst.assert_allclose(expected_summaries, found_summaries)


# setup and test indervidual methods inside the jet classes
def make_simple_jets(floats, jet_params={}, jet_class=FormJets.PseudoJet, assign=False, **kwargs):
    with TempTestDir("tst") as dir_name:
        ew = Components.EventWise(dir_name, "tmp.awkd")
        set_JetInputs(ew, floats)
        ew.selected_index = 0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        jets = jet_class(ew, dict_jet_params=jet_params, assign=assign, **kwargs)
    return jets


def apply_internal(jet_class, *internal_tests, additional_jet_params=None):
    if additional_jet_params is None:
        additional_jet_params = {}
    physical_distance_measures = ["angular", "invarient", "normed", "Luclus"]
    exp_of_pt_pos = ["input", "eigenspace"]
    exponents = [-1, -0.5, 0, 0.5, 1]
    deltaR = [0, 0.5, 1]
    #          pt  rap phi e  linear.... join_dist
    floats1 = [1., 0., 0., 2., np.nan, np.nan, np.nan, -1]
    SimpleClusterSamples.fill_linear(floats1)
    floats2 = [1., 0., 1., 2., np.nan, np.nan, np.nan, -1]
    SimpleClusterSamples.fill_linear(floats2)
    floats3 = [0.1, 1., 0., 2., np.nan, np.nan, np.nan, -1]
    SimpleClusterSamples.fill_linear(floats3)
    all_floats = [floats1, floats2, floats3]
    # some tests should still work with one or zero float lines
    one_tests = [test for test in internal_tests if test.valid_one]
    zero_tests = [test for test in internal_tests if test.valid_zero]
    for measure in physical_distance_measures:
        for pos in exp_of_pt_pos:
            for exp in exponents:
                for dr in deltaR:
                    jet_params = {"PhyDistance": measure, "ExpofPTPosition": pos,
                                  "ExpofPTMultiplier": exp, "DeltaR": dr,
                                  **additional_jet_params}
                    # run tests with no inputs
                    try:
                        jets = make_simple_jets(np.array([]).reshape((0, 0)),
                                                jet_params, jet_class=jet_class)
                    except AssertionError:
                        # maybe truying to make an invalid parameter combination
                        continue
                    else:
                        for internal_test in zero_tests:
                            internal_test(jets, jet_params)
                    for floatsA in all_floats:
                        # run tests with one input
                        jets = make_simple_jets(np.array([floatsA]), jet_params, jet_class=jet_class)
                        for internal_test in one_tests:
                            internal_test(jets, jet_params)
                        for floatsB in all_floats:
                            # run tests with two inputs
                            floats = np.array([floatsA, floatsB])
                            jets = make_simple_jets(floats, jet_params, jet_class=jet_class)
                            # run the tests
                            for internal_test in internal_tests:
                                internal_test(jets, jet_params)


# PseudoJet ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def internal_physical_distance(jets, param_dict):
    # extract quantities
    float_array = np.array(jets._floats)
    energies = float_array[:, jets._Energy_col]
    momentums = float_array[:, [jets._Px_col, jets._Py_col, jets._Pz_col]]
    pts = float_array[:, jets._PT_col]
    phis = float_array[:, jets._Phi_col]
    rapidities = float_array[:, jets._Rapidity_col]
    # prepare calculations
    invarient_mass2 = np.sum(energies)**2 - np.sum(np.sum(momentums, axis=0)**2)
    exponent = 2*param_dict['ExpofPTMultiplier']
    # get the pt factor
    if param_dict['ExpofPTPosition'] == 'input':
        if param_dict['PhyDistance'] == 'Luclus':
            pt_factor = np.product(pts)**exponent*invarient_mass2**(-0.5*exponent)/np.sum(pts)**exponent
        else:
            pt_factor = np.min(pts**exponent) * invarient_mass2**(-0.5*exponent)
    else:
        pt_factor = 1.
    # get the distanc ebetween the particels
    if param_dict['PhyDistance'] in ['invarient', 'normed']:
        invar = energies[0]*energies[1] - np.sum(momentums[0]*momentums[1])
        if param_dict['PhyDistance'] == 'invarient':
            distance2 = pt_factor*invar/invarient_mass2
        else:
            distance2 = pt_factor*invar/np.product(energies)
    else:
        angular_change2 = Components.angular_distance(*phis)**2 + (rapidities[0] - rapidities[1])**2
        distance2 = pt_factor*angular_change2
    # get the beam distance
    if param_dict['PhyDistance'] == 'Luclus' or param_dict['ExpofPTPosition'] != 'input':
        pt_factor = 1.
    else:
        pt_factor = pts**exponent * invarient_mass2**(-0.5*exponent)
    beam_distance2 = param_dict['DeltaR']**2 * pt_factor
    found_distance2 = jets.physical_distance2(*jets._floats)
    try:
        tst.assert_allclose(distance2, found_distance2, err_msg=f"distance between particles with floats \n{float_array}\nwhen joined with parameters\n{param_dict}\n does not match expected")
    except Exception as e:
        st()
    found_beam2 = [jets.beam_distance2(jets._floats[0]), jets.beam_distance2(jets._floats[1])]
    try:
        tst.assert_allclose(beam_distance2, found_beam2, err_msg=f"distance to beam with floats \n{float_array}\nwhen joined with parameters\n{param_dict}\n does not match expected")
    except Exception as e:
        st()
internal_physical_distance.valid_one = False
internal_physical_distance.valid_zero = False


def test_physical_distance():
    # check that an invalid event mass raises and error
    floats = np.ones((4, 8))
    with pytest.raises(ValueError):
        jets = make_simple_jets(floats, {}, FormJets.Spectral)


def internal_currently_avalible(jets, param_dict):
    avalible = np.sum(jets.Parent == -1)
    jets._calculate_currently_avalible()
    assert avalible == jets.currently_avalible
internal_currently_avalible.valid_one = True
internal_currently_avalible.valid_zero = True


def internal_getattr(jets, param_dict):
    # check an int column
    int_array = np.array(jets._ints)
    try:
        parents = int_array[:, jets._Parent_col]
    except IndexError:
        parents = []
    tst.assert_allclose(parents, jets.Parent)
    # check the float columns
    roots = jets.Parent == -1
    float_array = np.array(jets._floats)[roots]
    try:
        energy = float_array[:, jets._Energy_col]
        momentum = float_array[:, [jets._Px_col, jets._Py_col, jets._Pz_col]]
    except IndexError:
        energy = momentum = []
    with pytest.raises(AttributeError):
        jets.Bloggle  # check nonsenes attributes return attribute errors
    found = np.array([jets.PT, jets.Rapidity, jets.Phi, jets.Energy,
                      jets.Px, jets.Py, jets.Pz]).T.reshape((-1, 7))
    if len(energy):
        # calculate teh angular components
        expected = []
        for e, p in zip(energy, momentum):
            row = [None, None, None, e, *p]
            SimpleClusterSamples.fill_angular(row, change_energy=False)
            expected.append(row)
        tst.assert_allclose(found, expected)
        # also test P and Theta
        birr = np.sqrt(np.sum(momentum**2, axis=1))
        tst.assert_allclose(jets.P, birr)
        expected = np.array(expected)
        theta = Components.ptpz_to_theta(expected[:, 0], momentum[:, 2])
        tst.assert_allclose(jets.Theta, theta)
    else:
        assert np.all(np.isnan(found))
        assert np.all(np.isnan([jets.Theta, jets.P]))
internal_getattr.valid_one = True
internal_getattr.valid_zero = True


def internal_combine(jets, param_dict):
    i0, i1 = len(jets)-2, len(jets) -1
    int_array = np.array(jets._ints)[[i0, i1]]
    float_array = np.array(jets._floats)[[i0, i1]]
    combined_floats = np.sum(float_array, axis=0)
    SimpleClusterSamples.fill_angular(combined_floats, change_energy=False)
    distance2 = 1
    combined_floats[-1] = distance2
    combined_ints = [max(jets.InputIdx) + 1, -1, i0, i1, max(jets.Rank) + 1]
    found_ints, found_floats = jets._combine(i0, i1, distance2)
    SimpleClusterSamples.match_ints_floats([combined_ints], [combined_floats],
                                           [found_ints], [found_floats])
internal_combine.valid_one = False
internal_combine.valid_zero = False


def test_Pseudojet_internal():
    # testing Pseudojet functions, but creating Spectral jets
    # as Pseudojet should not be directly created and Traditional lack support for all options
    apply_internal(FormJets.Spectral,
                   internal_physical_distance,
                   internal_currently_avalible,
                   internal_getattr,
                   internal_combine)
    

def test_calculate_roots():
    ints = np.zeros((4, 5), dtype=int)
    floats = np.zeros((4, 8))
    jets = make_simple_jets(floats, {}, FormJets.Spectral)
    ints[:, jets._InputIdx_col] = list(range(4))
    ints[:, jets._Parent_col] = (1, -1, 5, 1)
    jets._ints = ints.tolist()
    jets.currently_avalible = 0  # need to set to 0 in order to calculate roots
    jets._calculate_roots()
    tst.assert_allclose(jets.root_jetInputIdxs, [1, 2])


def test_idx_from_inpIdx():
    n_rows = 4
    ints = np.zeros((n_rows, 5), dtype=int) -1
    floats = np.zeros((n_rows, 8))
    jets = make_simple_jets(floats, {}, FormJets.Spectral)
    input_idx = [0, 5, 3, 1]
    ints[:, jets._InputIdx_col] = input_idx
    jets._ints = ints.tolist()
    for i, inpidx in enumerate(input_idx):
        assert i == jets.idx_from_inpIdx(inpidx)
    with pytest.raises(ValueError):
        jets.idx_from_inpIdx(7)


def test_get_decendants():
    n_rows = 4
    ints = np.zeros((n_rows, 5), dtype=int) -1
    floats = np.zeros((n_rows, 8))
    jets = make_simple_jets(floats, {}, FormJets.Spectral)
    ints[:, jets._InputIdx_col] = [3,2,1,0]
    ints[:, jets._Parent_col] = (1, -1, -1, 1)
    ints[1, jets._Child1_col] = 0
    ints[1, jets._Child2_col] = 3
    jets._ints = ints.tolist()
    tst.assert_allclose(sorted(jets.get_decendants(lastOnly=False, jetInputIdx=2)),
                        [0, 2, 3])
    tst.assert_allclose(sorted(jets.get_decendants(lastOnly=True, jetInputIdx=2)),
                        [0, 3])
    tst.assert_allclose(sorted(jets.get_decendants(lastOnly=True, jetInputIdx=1)),
                        [1])
    tst.assert_allclose(sorted(jets.get_decendants(lastOnly=False, jetInputIdx=0)),
                        [0])
    # calling it without any index should raise a TypeError
    with pytest.raises(TypeError):
        jets.get_decendants()
    # should be equally possible to call it with local indices
    tst.assert_allclose(sorted(jets.get_decendants(lastOnly=False, pseudojet_idx=1)),
                        [0, 2, 3])
    tst.assert_allclose(sorted(jets.get_decendants(lastOnly=True, pseudojet_idx=1)),
                        [0, 3])
    tst.assert_allclose(sorted(jets.get_decendants(lastOnly=True, pseudojet_idx=2)),
                        [1])
    tst.assert_allclose(sorted(jets.get_decendants(lastOnly=False, pseudojet_idx=3)),
                        [0])


def test_local_obs_idx():
    n_rows = 4
    ints = np.zeros((n_rows, 5), dtype=int) -1
    floats = np.zeros((n_rows, 8))
    jets = make_simple_jets(floats, {}, FormJets.Spectral)
    ints[:, jets._InputIdx_col] = [3,2,1,0]
    ints[:, jets._Parent_col] = (1, -1, -1, 1)
    ints[1, jets._Child1_col] = 0
    ints[1, jets._Child2_col] = 3
    jets._ints = ints.tolist()
    tst.assert_allclose(sorted(jets.local_obs_idx()), [0, 2, 3])

# Traditional ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def internal_recalculate_one(jets, param_dict):
    int_array = np.array(jets._ints)
    float_array = np.array(jets._floats)
    # duplicate the floats and ints
    int_array[:, jets._InputIdx_col] += len(jets)
    jets._ints += int_array.tolist()
    jets._floats += float_array.tolist()
    jets._calculate_currently_avalible()
    # calculate the distances
    jets._calculate_distances()
    start_distances = np.copy(jets._distances2)
    # now mess with one float rows
    change = 1
    float_array[change, jets._Energy_col] += 3.
    float_array[change, [jets._Px_col, jets._Py_col]] += 1.
    SimpleClusterSamples.fill_angular(float_array[change])
    jets._floats[change] = float_array[change].tolist()
    # recalculate another distance
    # expectation is that floats in the replace row have changed
    # but this can be used to check if the right distances are being altered
    replace, remove = 2, 3  # it is important that the remove axis is the last
    jets.currently_avalible -= 1
    jets._recalculate_one(remove, replace)
    start_distances = np.delete(start_distances, remove, axis=0)
    start_distances = np.delete(start_distances, remove, axis=1)
    # now all start_distances that don't intersect with change should eb the same
    should_change = np.full_like(start_distances, False, dtype=bool)
    should_change[:, change] = True
    should_change[change, :] = True
    upper_triangle = np.triu_indices(len(should_change), -1)
    # we don't need the upper triangle to change per say, but it can to o detriment change
    should_change[upper_triangle] = True
    tst.assert_allclose(start_distances[~should_change], jets._distances2[~should_change])
    with pytest.raises(AssertionError):
        tst.assert_allclose(start_distances[should_change], jets._distances2[should_change])
internal_recalculate_one.valid_one = False
internal_recalculate_one.valid_zero = False


def test_Traditional_internal():
    # testing Pseudojet functions, but creating Spectral jets
    # as Pseudojet should not be directly created and Traditional lack support for all options
    apply_internal(FormJets.Traditional, internal_recalculate_one)
    
# Spectral ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# wait until spectral to test this becuase the cutoff is the most complex parameter
def test_write_event():
    with TempTestDir("tst") as dir_name:
        jet_params = {'DeltaR': 0.4, 'ExpofPTPosition': 'input', 'AffinityCutoff': ('knn', 3)}
        ints = SimpleClusterSamples.close_join['ints']
        floats = SimpleClusterSamples.close_join['floats']
        ew = Components.EventWise(dir_name, "tmp.awkd")
        set_JetInputs(ew, floats)
        ew.selected_index = 0
        jets = FormJets.Spectral(ew, dict_jet_params=jet_params, assign=False)
        jets._ints = ints.tolist()
        jets._floats = floats.tolist()
        FormJets.Spectral.write_event([jets])
        # check what got written is correct
        jet_name = jets.jet_name
        jet_columns = [name.split('_', 1)[1] for name in ew.columns
                       if name.startswith(jet_name)]
        for col in jet_columns:
            found = getattr(ew, jet_name + '_' + col).flatten().flatten()
            if col == 'RootInputIdx':
                # this won't be in the jets
                assert len(found) == 0
                continue
            is_int = any(full_col.split('_', 1)[1] == col for full_col in jets.int_columns)
            col_num = getattr(jets, '_'+col+'_col')
            expected = ints[:, col_num] if is_int else floats[:, col_num]
            expected = expected.flatten()
            tst.assert_allclose(found, expected, err_msg=f'Missmatch in {col}, is_int={is_int}')
        for key in jet_params:
            found = getattr(ew, jet_name+'_'+key)
            assert found == jet_params[key], \
                    f"{key}; Found {found}, expected {jet_params[key]}"
        # then check params should return true as well
        assert jets.check_params(ew)
        # try changing one of the parameters in the jet
        jet_params['DeltaR'] += 1
        jets = make_simple_jets(floats, jet_params, FormJets.Spectral)
        assert not jets.check_params(ew)
        jet_params['DeltaR'] -= 1
        jet_params['AffinityCutoff'] = None
        jets = make_simple_jets(floats, jet_params, FormJets.Spectral)
        assert not jets.check_params(ew)
        # the chekc should return false if the eventWise has diferent parameters listed
        jet_params['AffinityCutoff'] = ('knn', 3)
        ew.remove(jet_name + '_AffinityCutoff')
        jets = make_simple_jets(floats, jet_params, FormJets.Spectral)
        assert not jets.check_params(ew)


# TODO add test for removing correct eigenvector when there are seperated graph components
def test_calculate_eigenspace_distances():
    n_rows = 4
    dr = 0.8
    # test that two seperated clusters results in an eigenvectors that seperates them
    floats = np.array([[1., 0., 0., 1., 1., 0., 0., 0.],
                       [1., 0., np.pi-0.1, 1., -1., 0., 0., 0.],
                       [1., 0., np.pi, 1., -1., 0., 0., 0.]])
    for row in floats:
        SimpleClusterSamples.fill_angular(row)
    jets = make_simple_jets(floats, {'Laplacien': 'symmetric',
                                     'AffinityCutoff': ('distance', 0.2),
                                     'NumEigenvectors': 1,
                                     'DeltaR': dr}, FormJets.Spectral)
    # check that there is an eigenvector with the pattern [x, y, y]
    assert jets._eigenspace.shape[1] == 1
    eigenvector = jets._eigenspace[:, 0]
    assert not np.isclose(eigenvector[0], eigenvector[1])
    assert np.isclose(eigenvector[1], eigenvector[2])
    # try again with 2 in ech group
    floats = np.array([[1., 0., 0., 1., 1., 0., 0., 0.],
                       [1., 0., 0.1, 1., 1.+0.1, 0., 0., 0.],
                       [1., 0., np.pi-0.1, 1., -1.+0.1, 0., 0., 0.],
                       [1., 0., np.pi, 1., -1., 0., 0., 0.]])
    for row in floats:
        SimpleClusterSamples.fill_angular(row)
    jets = make_simple_jets(floats, {'Laplacien': 'symmetric',
                                     'AffinityCutoff': ('distance', 0.2),
                                     'NumEigenvectors': 1,
                                     'DeltaR': dr}, FormJets.Spectral)
    # check that there is an eigenvector with the pattern [x, x, y, y]
    assert jets._eigenspace.shape[1] == 1
    eigenvector = jets._eigenspace[:, 0]
    assert not np.isclose(eigenvector[0], eigenvector[2])
    assert np.isclose(eigenvector[0], eigenvector[1])
    assert np.isclose(eigenvector[2], eigenvector[3])
    # test some random combinations
    for i in range(5):
        floats = np.random.random((n_rows, 8))
        # set distance to 0
        floats[:, -1] = 0.
        for row in floats:
            SimpleClusterSamples.fill_angular(row)
        for ltype in ('unnormalised', 'symmetric'):
            for exp_mul, exp_pos in [(-1, 'input'), (-1, 'eigenspace'), (0, 'eigenspace'), (.5, 'eigenspace')]:
                jets = make_simple_jets(floats, {'Laplacien': ltype,
                                                 'ExpofPTPosition': exp_pos,
                                                 'ExpofPTMultiplier': exp_mul,
                                                 'DeltaR': dr}, FormJets.Spectral)
                # check that distance obtained from this is correct
                distances2 = np.array([[np.sum((row-col)**2) for row in jets._eigenspace]
                                      for col in jets._eigenspace])
                if exp_pos == 'eigenspace':
                    pts = np.fromiter((row[jets._PT_col] for row in jets._floats), dtype=float)
                    factors = np.array([[min(pt1**(2*exp_mul), pt2**(2*exp_mul)) for pt1 in pts]
                                        for pt2 in pts])
                    distances2 *= factors
                np.fill_diagonal(distances2, dr**2)
                tst.assert_allclose(jets._distances2, distances2)
            # check that the eigenvectors and the eigenvalues have the correct relationship
            # with the laplacien
            # this only needs to be done once for each laplacien type,
            # hence we do it outside the loop.
            affinities = jets._affinity
            diagonal = np.diag(np.sum(affinities, axis=0))
            laplacien = diagonal - affinities
            if ltype == 'symmetric':
                alt_diag = np.diag(np.diag(diagonal)**(-0.5))
                laplacien = np.matmul(alt_diag, np.matmul(laplacien, alt_diag))
            for eigenvalue, eigenvector in zip(jets.eigenvalues[0], jets._eigenspace.T):
                lhs = np.matmul(laplacien, eigenvector.reshape((-1, 1))).flatten()
                rhs = eigenvalue*eigenvector
                tst.assert_allclose(lhs, rhs)


def test_merge_remove_pseudojets():
    n_rows = 4
    ints = np.zeros((n_rows, 5), dtype=int) -1
    floats = np.random.random((n_rows, 8))
    # set distance to 0
    floats[:, -1] = 0.
    for row in floats:
        SimpleClusterSamples.fill_angular(row)
    jets = make_simple_jets(floats, {}, FormJets.Spectral)
    ints[:, jets._InputIdx_col] = list(range(4))
    ints[:, jets._Parent_col] = (1, -1, -1, 1)
    ints[1, jets._Child1_col] = 0
    ints[1, jets._Child2_col] = 3
    jets._ints = ints.tolist()
    # save teh eigen space and distances befor the merge
    old_eigenspace = np.copy(jets._eigenspace)
    old_distance = np.copy(jets._distances2)
    #  start by merging rows  1 and 2
    i0, i1 = 1, 2
    distance2 = 1
    new_ints, new_floats = jets._combine(i0, i1, distance2)
    ints[[i0,i1], jets._Parent_col] = new_ints[jets._InputIdx_col]
    jets._merge_pseudojets(i0, i1, distance2)
    # it is expected that the lower index with contain the new jet
    SimpleClusterSamples.match_ints_floats([jets._ints[i0]], [jets._floats[i0]],
                                           [new_ints], [new_floats])
    # the old ones should now be at the end
    SimpleClusterSamples.match_ints_floats(jets._ints[-2:], jets._floats[-2:],
                                           ints[[i0,i1]], floats[[i0, i1]])
    assert len(jets) == len(ints) + 1
    # the recalculated eigenspace position is out of scope for this test
    # that will be tested in _combine
    # but the others should be shifted by the removed index
    # check the top left
    tst.assert_allclose(old_distance[0, 0], jets._distances2[0, 0])
    # check the bottom right
    tst.assert_allclose(old_distance[3:, 3:], jets._distances2[2:, 2:])
    # check the top right
    tst.assert_allclose(old_distance[0, 3:], jets._distances2[0, 2:])
    # check the bottom left
    tst.assert_allclose(old_distance[3:, 0], jets._distances2[2:, 0])
    # the first row of the eigenspac should eb unchanged
    tst.assert_allclose(old_eigenspace[0], jets._eigenspace[0])
    # and all after the second
    tst.assert_allclose(old_eigenspace[3:], jets._eigenspace[2:])


def internal_calculate_affinity(jets, param_dict):
    # not neded right now as no ties
    # this function does not deal with tie breaking
    # which is reansomable, since ties are artificial
    # so "shake" all the point in the jet
    #small_distance = 0.01
    #floats = np.array(jets._floats)
    #floats += small_distance*np.random.random(floats.shape)
    #[SimpleClusterSamples.fill_angular(row, False) for row in floats]
    #jets._floats = floats.tolist()
    ## then recalculate the distances
    #jets.currently_avalible = len(floats)
    #jets._calculate_distances()
    # and procede with a comparison calculation
    distances2 = np.array([[jets.physical_distance2(row, col) for col in jets._floats]
                           for row in jets._floats])
    if len(distances2) < 1:
        tst.assert_allclose(jets._affinity, np.array([[]]))
        return
    if param_dict['StoppingCondition'] == 'beamparticle':
        # if there is a beam particles the last row should eb beam distances
        # but only if something exists
        beam_row = np.fromiter((jets.beam_distance2(row) for row in jets._floats), dtype=float).reshape((1, -1))
        distances2 = np.concatenate((distances2, beam_row))
        beam_row = np.concatenate((beam_row.T, [[0]]))
        distances2 = np.concatenate((distances2, beam_row), axis=1)
    if len(distances2) < 2:
        tst.assert_allclose(jets._affinity, np.array([[]]))
        return
    np.fill_diagonal(distances2, 0)
    distances = np.sqrt(distances2)  # now distances will be physical distances
    expected = np.zeros_like(distances)
    if param_dict['AffinityType'] == 'linear':
        expected = np.max(distances) - distances
    elif param_dict['AffinityType'] == 'inverse':
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            expected = 1/distances
    elif param_dict['AffinityType'] == 'exponent':
        expected = np.exp(-distances)
    elif param_dict['AffinityType'] == 'exponent2':
        expected = np.exp(-distances2)
    else:
        raise KeyError
    # apply the cut off
    if param_dict['AffinityCutoff'] is not None:
        to_keep = param_dict['AffinityCutoff'][1]
        if param_dict['AffinityCutoff'][0] == 'knn':
            np.fill_diagonal(expected, 0)
            order = np.argsort(distances, axis=1)
            # plus on for the diagonal
            expected[order > to_keep + 1] = 0
        elif param_dict['AffinityCutoff'][0] == 'distance':
            expected[distances > to_keep] = 0
        else:
            raise KeyError
    if len(expected) == 0:
        expected = expected.reshape((1, 0))
    np.fill_diagonal(expected, 0)
    if np.inf in expected:  # fix internal approximations
        jets._affinity[jets._affinity > 10^100] = np.inf
    try:
        tst.assert_allclose(jets._affinity, expected, atol=0.0001, err_msg=f"Unexpected affinity for jets;\n{param_dict}\n Found distances\n {distances}")
    except:
        st()
        jets2 = make_simple_jets(np.array(jets._floats), param_dict, type(jets))
    expected
internal_calculate_affinity.valid_one = True
internal_calculate_affinity.valid_zero = True


def test_step_assign_parents():
    # check results have correct form
    n_rows = 8
    for _ in range(10):
        floats = np.random.random((n_rows, 8))
        # set distance to 0
        floats[:, -1] = 0.
        for row in floats:
            SimpleClusterSamples.fill_angular(row)
        jets = make_simple_jets(floats, {}, FormJets.Spectral)
        assert jets.currently_avalible == n_rows
        assert len(jets._ints) == n_rows
        assert len(jets._floats) == n_rows
        jets._step_assign_parents()
        assert jets.currently_avalible == n_rows - 1
        # two possibilities, a jet was removed or a jet was merged
        if len(jets.root_jetInputIdxs) == 0:
            # merging
            assert len(jets._ints)  == n_rows + 1
            assert len(jets._floats) == n_rows + 1
            parents = jets.Parent
            assert len(set(parents)) == 2
            new_jet_idx = jets.idx_from_inpIdx(max(parents))
            assert parents[jets.idx_from_inpIdx(jets.Child1[new_jet_idx])] == max(parents)
            assert parents[jets.idx_from_inpIdx(jets.Child2[new_jet_idx])] == max(parents)
        else:
            # removed
            assert len(jets.root_jetInputIdxs) == 1
            assert jets.root_jetInputIdxs[0] in jets.InputIdx
            assert len(jets._ints)  == n_rows
            assert len(jets._floats) == n_rows


def test_Spectral_internal_linear():
    # testing Pseudojet functions, but creating Spectral jets
    # as Pseudojet should not be directly created and Traditional lack support for all options
    affinity = 'linear'
    for affinity_cutoff in [None, ('knn', 3), ('distance', 0.5), ('knn', 1), ('distance', 0)]:
        additional_jet_params = dict(StoppingCondition='standard',
                                     AffinityType=affinity,
                                     AffinityCutoff=affinity_cutoff)
        apply_internal(FormJets.Spectral, internal_calculate_affinity,
                       additional_jet_params=additional_jet_params)
        additional_jet_params['StoppingCondition'] = 'beamparticle'
        apply_internal(FormJets.Spectral, internal_calculate_affinity,
                       additional_jet_params=additional_jet_params)
    
def test_Spectral_internal_exponent():
    # testing Pseudojet functions, but creating Spectral jets
    # as Pseudojet should not be directly created and Traditional lack support for all options
    affinity  = 'exponent'
    for affinity_cutoff in [None, ('knn', 3), ('distance', 0.5), ('knn', 1), ('distance', 0)]:
        additional_jet_params = dict(StoppingCondition='standard',
                                     AffinityType=affinity,
                                     AffinityCutoff=affinity_cutoff)
        apply_internal(FormJets.Spectral, internal_calculate_affinity,
                       additional_jet_params=additional_jet_params)
        additional_jet_params['StoppingCondition'] = 'beamparticle'
        apply_internal(FormJets.Spectral, internal_calculate_affinity,
                       additional_jet_params=additional_jet_params)
    
def test_Spectral_internal_exponent2():
    # testing Pseudojet functions, but creating Spectral jets
    # as Pseudojet should not be directly created and Traditional lack support for all options
    affinity = 'exponent2'
    for affinity_cutoff in [None, ('knn', 3), ('distance', 0.5), ('knn', 1), ('distance', 0)]:
        additional_jet_params = dict(StoppingCondition='standard',
                                     AffinityType=affinity,
                                     AffinityCutoff=affinity_cutoff)
        apply_internal(FormJets.Spectral, internal_calculate_affinity,
                       additional_jet_params=additional_jet_params)
        additional_jet_params['StoppingCondition'] = 'beamparticle'
        apply_internal(FormJets.Spectral, internal_calculate_affinity,
                       additional_jet_params=additional_jet_params)
    
def test_Spectral_internal_inverse():
    # testing Pseudojet functions, but creating Spectral jets
    # as Pseudojet should not be directly created and Traditional lack support for all options
    affinity = 'inverse'
    for affinity_cutoff in [None, ('knn', 3), ('distance', 0.5), ('knn', 1), ('distance', 0)]:
        additional_jet_params = dict(StoppingCondition='standard',
                                     AffinityType=affinity,
                                     AffinityCutoff=affinity_cutoff)
        apply_internal(FormJets.Spectral, internal_calculate_affinity,
                       additional_jet_params=additional_jet_params)
        additional_jet_params['StoppingCondition'] = 'beamparticle'
        apply_internal(FormJets.Spectral, internal_calculate_affinity,
                       additional_jet_params=additional_jet_params)
    

def test_SP_recalculate_one():
    # short of repeating the calculation
    # can only really check the form
    # and that points move
    n_rows = 10
    floats = np.random.random((n_rows, 8))
    for row in floats:
        SimpleClusterSamples.fill_angular(row)
    deltaR=0.4
    jets = make_simple_jets(floats, {'DeltaR':0.4}, FormJets.Spectral)
    replace, remove = 0, 1
    old_eigenspace = np.delete(jets._eigenspace, remove, axis=0)
    jets._merge_pseudojets(remove, replace, 0)
    # check form
    assert jets._eigenspace.shape == (n_rows-1, n_rows-1)
    assert jets._distances2.shape == (n_rows-1, n_rows-1)
    # check they moved
    assert not np.allclose(jets._eigenspace, old_eigenspace)
    # again, with a symmetric laplacien
    jets = make_simple_jets(floats, {'deltar':0.4,
                                     'laplacien':'symmetric'}, FormJets.Spectral)
    replace, remove = 0, 1
    old_eigenspace = np.delete(jets._eigenspace, remove, axis=0)
    jets._merge_pseudojets(remove, replace, 0)
    # check form
    assert jets._eigenspace.shape == (n_rows-1, n_rows-1)
    assert jets._distances2.shape == (n_rows-1, n_rows-1)
    # check they moved
    assert not np.allclose(jets._eigenspace, old_eigenspace)
    # again with a beam particle
    jets = make_simple_jets(floats, {'DeltaR':0.4,
                                     'StoppingCondition': "beamparticle"}, FormJets.Spectral)
    replace, remove = 0, 1
    old_eigenspace = np.delete(jets._eigenspace, remove, axis=0)
    jets._merge_pseudojets(remove, replace, 0)
    # check form
    assert jets._eigenspace.shape == (n_rows, n_rows)
    assert jets._distances2.shape == (n_rows, n_rows)
    # check they moved
    assert not np.allclose(jets._eigenspace, old_eigenspace)
    
# Splitting ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# TODO when this has stabalised make distance and eigenspace tests
#def internal_Splitting_calculate_distances():
#    pass  # TODO


#def internal_Splitting_calculate_eigenspace():
#    pass  # TODO


#def internal_Splitting_step_assign_parents():
#    pass  # TODO


def test_merge_complete_jet():
    n_rows = 8
    floats = np.random.random((n_rows, 8))
    for row in floats:
        SimpleClusterSamples.fill_angular(row)
    jets = make_simple_jets(floats, {}, FormJets.Splitting)
    ints = np.array(jets._ints)
    assert jets.currently_avalible == n_rows
    assert len(jets._ints) == n_rows
    assert len(jets._floats) == n_rows
    merge_idxs = [4,2,1]
    unmerged = [i for i in range(n_rows) if i not in merge_idxs]
    merge_input_idxs = [jets.idx_from_inpIdx(i) for i in merge_idxs]
    jets._merge_complete_jet(merge_input_idxs)
    # check that things that shouldn't move didnt
    new_ints = np.array(jets._ints[:jets.currently_avalible])
    new_floats = np.array(jets._floats[:jets.currently_avalible])
    SimpleClusterSamples.match_ints_floats(ints[unmerged], floats[unmerged], new_ints, new_floats,
                                           compare_distance=False)
    # check that the cluster was formed
    assert jets.currently_avalible == n_rows - len(merge_idxs)
    assert len(jets.root_jetInputIdxs) == 1
    # the children of the root should be only and all the merged items
    children = jets.get_decendants(True, jetInputIdx=jets.root_jetInputIdxs[0])
    assert set(children) == set(merge_input_idxs)


def test_merge_complete_jets():
    n_rows = 10
    floats = np.random.random((n_rows, 8))
    for row in floats:
        SimpleClusterSamples.fill_angular(row)
    jets = make_simple_jets(floats, {}, FormJets.Splitting)
    ints = np.array(jets._ints)
    assert jets.currently_avalible == n_rows
    assert len(jets._ints) == n_rows
    assert len(jets._floats) == n_rows
    merge_idxs = np.array([[4,2,1], [3,6,7]])
    merge_sets = [set(jet) for jet in merge_idxs]
    jets._jets = merge_idxs
    jets._merge_complete_jets()
    # check that the cluster was formed
    assert jets.currently_avalible == n_rows - len(merge_idxs.flatten())
    assert len(jets.root_jetInputIdxs) == len(merge_idxs)
    for root in jets.root_jetInputIdxs:
        # the children of the root should be only and all the merged items
        children = jets.get_decendants(True, jetInputIdx=root)
        assert set(children) in merge_sets

#def test_Splitting_internal():
#    apply_internal(FormJets.Splitting,
#                   internal_Splitting_calculate_distances,
#                   internal_Splitting_calculate_eigenspace,
#                   internal_Splitting_step_assign_parents)

# SpectralMean ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def test_SM_recalculate_one():
    n_columns = 8
    float_variations = []
    # try with all zeros
    float_variations.append(np.zeros((2, n_columns)))
    # try some set pieces
    floats = SimpleClusterSamples.two_degenerate['floats']
    float_variations.append(floats)
    floats = SimpleClusterSamples.two_close['floats']
    float_variations.append(floats)
    floats = SimpleClusterSamples.two_oposite['floats']
    float_variations.append(floats)
    for i in range(20):
        floats = np.random.random((10, n_columns))
        float_variations.append(floats)

    for floats in float_variations:
        for row in floats:
            SimpleClusterSamples.fill_angular(row, True)
        deltaR=0.4
        jets = make_simple_jets(floats, {'DeltaR':0.4}, FormJets.SpectralMean)
        err_msg = "Problem when recalculating with floats;\n"
        err_msg += "E, Px, Py, Pz\n"
        for row in floats:
            err_msg += f"{row[jets._Energy_col]}, {row[jets._Px_col]}, {row[jets._Py_col]}"
            err_msg += f", {row[jets._Pz_col]}\n"
        replace, remove = 0, 1
        expected_position = (jets._eigenspace[remove] + jets._eigenspace[replace])/2
        jets._merge_pseudojets(remove, replace, 0)
        tst.assert_allclose(expected_position, jets._eigenspace[replace], err_msg=err_msg+"Eigenspace wrong")
        # now caculate all distances
        distances2 = np.array([[np.sum((row-col)**2) for col in jets._eigenspace]
                               for row in jets._eigenspace])
        np.fill_diagonal(distances2, deltaR**2)
        tst.assert_allclose(jets._distances2, distances2, err_msg=err_msg+"Distance2 wrong")
        # try with a pt in the eigenspace
        exp_mul = 0.5
        jets = make_simple_jets(floats, {'DeltaR':0.4,
                                         'ExpofPTPosition':'eigenspace',
                                         'ExpofPTMultiplier': exp_mul}, FormJets.SpectralMean)
        # grap the pt before the combination
        pt_list = floats[1:, jets._PT_col]**(2*exp_mul)
        combined_PT2 = (floats[remove, jets._Px_col] + floats[replace, jets._Px_col])**2 +\
                       (floats[remove, jets._Py_col] + floats[replace, jets._Py_col])**2
        pt_list[0] = combined_PT2**exp_mul
        expected_position = (jets._eigenspace[remove] + jets._eigenspace[replace])/2
        jets._merge_pseudojets(remove, replace, 0)
        tst.assert_allclose(expected_position, jets._eigenspace[replace], err_msg=err_msg+"Eigenspace wrong")
        # now caculate all distances
        distances2 = np.array([[min(pta, ptb)*np.sum((row-col)**2) for col, pta in zip(jets._eigenspace, pt_list)]
                               for row, ptb in zip(jets._eigenspace, pt_list)])
        np.fill_diagonal(distances2, deltaR**2)
        tst.assert_allclose(jets._distances2, distances2, err_msg=err_msg+"Distance2 wrong")
        # try with a beam particle
        exp_mul = 0.5
        jets = make_simple_jets(floats, {'DeltaR':0.4,
                                         'StoppingCondition':'beamparticle'}, FormJets.SpectralMean)
        expected_position = (jets._eigenspace[remove] + jets._eigenspace[replace])/2
        jets._merge_pseudojets(remove, replace, 0)
        tst.assert_allclose(expected_position, jets._eigenspace[replace], err_msg=err_msg+"Eigenspace wrong")
    

# SpectralFull ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def test_SF_recalculate_one():
    n_rows = 10
    floats = np.random.random((n_rows, 8))
    for row in floats:
        SimpleClusterSamples.fill_angular(row)
    deltaR=0.4
    jets = make_simple_jets(floats, {'DeltaR':0.4}, FormJets.SpectralFull)
    replace, remove = 0, 1
    assert len(jets._distances2) == n_rows
    assert len(jets._eigenspace) == n_rows
    jets._merge_pseudojets(remove, replace, 0)
    # now check that the eigenspace and the distances have shrunk,
    # as the eigenspace itself is already tested that should be fine
    assert len(jets._distances2) == n_rows-1
    assert len(jets._eigenspace) == n_rows-1

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def test_cluster_edge_cases():
    # go through a subset of clustering combinations checking that two particles
    # that are placed on top of eachother are combined always
    spectral_classes = (FormJets.Spectral, FormJets.SpectralMean, FormJets.SpectralFull)
    deltaR = (0.1, 1.)
    num_eig = (1, 3, np.inf)
    exp_pos = ('input', 'eigenspace')
    exp_mul = (-1, 0, 1)
    affinity_type = ('linear', 'exponent')  # skip exponent2 and inverse
    # also skip affinity cutoffs
    laplacien = ('unnormalised', 'symmetric')
    dist = ('angular', 'invarient', 'Luclus')
    stopping = ('beamparticle', 'standard')
    param_order = ['DeltaR', 'NumEigenvectors', 'ExpofPTPosition', 'ExpofPTMultiplier',
                  'AffinityType', 'Laplacien', 'PhyDistance', 'StoppingCondition']
    to_combine = (spectral_classes, deltaR, num_eig, exp_pos, exp_mul, affinity_type,
                  laplacien, dist, stopping)
    for combination in itertools.product(*to_combine):
        jet_class = combination[0]
        jet_params = {name: value for name, value in zip(param_order, combination[1:])}
        # the degenrate pair
        jet = make_simple_jets(SimpleClusterSamples.two_degenerate['floats'],
                               jet_params, jet_class=jet_class, assign=True)


def test_filter_obs():
    with TempTestDir("filter_obs") as dir_name:
        name = "test.awkd"
        ew = Components.EventWise(dir_name, name, columns=["Particle_Tower", "Particle_Track"],
                contents={"Particle_Track": AwkdArrays.event_ints, "Particle_Tower": AwkdArrays.event_ints})
        ew.selected_index = 0
        input_outputs = [
                ([], []),
                ([0], [0]),
                ([0, 1], [0, 1])]
        for inp, expected in input_outputs:
            out = FormJets.filter_obs(ew, np.array(inp))
            tst.assert_allclose(out, expected)
        all_unseen = awkward.fromiter([[-1, -1], [-1]])
        ew = Components.EventWise(dir_name, name, columns=["Particle_Tower", "Particle_Track"],
                contents={"Particle_Track": AwkdArrays.event_ints, "Particle_Tower": all_unseen})
        ew.selected_index = 0
        input_outputs = [
                ([], []),
                ([0], [0]),
                ([0, 1], [0, 1])]
        for inp, expected in input_outputs:
            out = FormJets.filter_obs(ew, np.array(inp))
            tst.assert_allclose(out, expected)
        ew = Components.EventWise(dir_name, name, columns=["Particle_Tower", "Particle_Track"],
                contents={"Particle_Track": all_unseen, "Particle_Tower": AwkdArrays.event_ints})
        ew.selected_index = 0
        input_outputs = [
                ([], []),
                ([0], [0]),
                ([0, 1], [0, 1])]
        for inp, expected in input_outputs:
            out = FormJets.filter_obs(ew, np.array(inp))
            tst.assert_allclose(out, expected)
        ew = Components.EventWise(dir_name, name, columns=["Particle_Tower", "Particle_Track"],
                contents={"Particle_Track": all_unseen, "Particle_Tower": all_unseen})
        ew.selected_index = 0
        input_outputs = [
                ([], []),
                ([0], []),
                ([0, 1], [])]
        for inp, expected in input_outputs:
            out = FormJets.filter_obs(ew, np.array(inp))
            tst.assert_allclose(out, expected)
        one_unseen = awkward.fromiter([[-1, 1], [2]])
        ew = Components.EventWise(dir_name, name, columns=["Particle_Tower", "Particle_Track"],
                contents={"Particle_Track": one_unseen, "Particle_Tower": all_unseen})
        ew.selected_index = 0
        input_outputs = [
                ([], []),
                ([0], []),
                ([0, 1], [1])]
        for inp, expected in input_outputs:
            out = FormJets.filter_obs(ew, np.array(inp))
            tst.assert_allclose(out, expected)


def test_filter_ends():
    with TempTestDir("filter_ends") as dir_name:
        name = "test.awkd"
        ew = Components.EventWise(dir_name, name, columns=["Children"],
                                  contents={"Children": AwkdArrays.empty_jets})
        ew.selected_index = 0
        input_outputs = [
                ([], []),
                ([0], [0]),
                ([0, 1], [0, 1])]
        for inp, expected in input_outputs:
            out = FormJets.filter_ends(ew, np.array(inp))
            tst.assert_allclose(out, expected)
        ew = Components.EventWise(dir_name, name, columns=["Children"],
                                  contents={"Children": AwkdArrays.empty_jet})
        ew.selected_index = 0
        input_outputs = [
                ([], []),
                ([0], [0]),
                ([0, 1], [0])]
        for inp, expected in input_outputs:
            out = FormJets.filter_ends(ew, np.array(inp))
            tst.assert_allclose(out, expected)


def test_filter_pt_eta():
    with TempTestDir("filter_pt_eta") as dir_name:
        name = "test.awkd"
        ew = Components.EventWise(dir_name, name, columns=["PT", "Pseudorapidity"],
                contents={"PT": AwkdArrays.event_floats, "Pseudorapidity": AwkdArrays.event_floats})
        ew.selected_index = 0
        input_outputs = [
                (0.05, 1., [], []),
                (0.05, 1., [0], [0]),
                (0.05, 1., [0, 1], [0, 1]),
                (0.15, 1., [], []),
                (0.15, 1., [0], []),
                (0.15, 1., [0, 1], [ 1])]
        for pt, eta, inp, expected in input_outputs:
            out = FormJets.filter_pt_eta(ew, np.array(inp), pt, eta)
            tst.assert_allclose(out, expected)
        pt = awkward.fromiter([[0.1, 0.1, 0.1, 0.1, 0.1]])
        pz = awkward.fromiter([[0., 0.1, 100., -0.1, -100.]])
        ew = Components.EventWise(dir_name, name, columns=["PT", "Pz"],
                contents={"PT": pt, "Pz": pz})
        ew.selected_index = 0
        input_outputs = [
                (0.05, 1., [], []),
                (0.05, 1., [0, 1, 2, 3, 4], [0, 1, 3]),
                (0.05, .1, [], []),
                (0.05, .1, [0, 1, 2, 3, 4], [0])]
        for pt, eta, inp, expected in input_outputs:
            out = FormJets.filter_pt_eta(ew, np.array(inp), pt, eta)
            tst.assert_allclose(out, expected)


def test_create_JetInputs():
    with TempTestDir("create_JetInputs") as dir_name:
        name = "test.awkd"
        columns = [name[len("Pseudojet_"):] for name in FormJets.PseudoJet.float_columns
                   if "Distance" not in name]
        columns_unchanged = [c for c in columns]  # because the ew will change the columns list
        floats = SimpleClusterSamples.two_oposite['floats']
        contents = {name: awkward.fromiter([x]) for name, x in zip(columns, floats.T)}
        def return_all(_, current_idx):
            return current_idx
        def return_second(_, current_idx):
            return current_idx[1:2]
        ew = Components.EventWise(dir_name, name, columns=columns, contents=contents)
        FormJets.create_jetInputs(ew, filter_functions=[return_all])
        for name in columns_unchanged:
            ji_name = "JetInputs_" + name
            idx = columns.index(name)
            assert hasattr(ew, ji_name)
            tst.assert_allclose(getattr(ew, ji_name).tolist(), [floats.T[idx]],
                                err_msg=f"In two_oposite {name} not matching")
        ew.remove_prefix("JetInputs")
        FormJets.create_jetInputs(ew, filter_functions=[return_second])
        for name in columns_unchanged:
            ji_name = "JetInputs_" + name
            idx = columns.index(name)
            assert hasattr(ew, ji_name)
            tst.assert_allclose(getattr(ew, ji_name).tolist(), [floats.T[idx][1:2]])
        # test batching
        columns = [name[len("Pseudojet_"):] for name in FormJets.PseudoJet.float_columns
                   if "Distance" not in name]
        contents = {name: awkward.fromiter([x, x, x]) for name, x in zip(columns, floats.T)}
        ew = Components.EventWise(dir_name, name, columns=columns, contents=contents)
        FormJets.create_jetInputs(ew, filter_functions=[return_all], batch_length=0)
        for name in columns_unchanged:
            ji_name = "JetInputs_" + name
            assert len(getattr(ew, ji_name)) == 0
        FormJets.create_jetInputs(ew, filter_functions=[return_all], batch_length=1)
        for name in columns_unchanged:
            ji_name = "JetInputs_" + name
            assert hasattr(ew, ji_name)
            idx = columns.index(name)
            tst.assert_allclose(getattr(ew, ji_name).tolist(), [floats.T[idx]])
        FormJets.create_jetInputs(ew, filter_functions=[return_all], batch_length=1)
        for name in columns_unchanged:
            ji_name = "JetInputs_" + name
            assert hasattr(ew, ji_name)
            idx = columns.index(name)
            tst.assert_allclose(getattr(ew, ji_name).tolist(), [floats.T[idx], floats.T[idx]])


def test_produce_summary():
    with TempTestDir("produce_summary") as dir_name:
        name = "test.awkd"
        n_jet_inputs = len(FormJets.PseudoJet.float_columns)-1
        path = os.path.join(dir_name, "summary_observables.csv")
        # try an empty event
        empty_ew = Components.EventWise(dir_name, name)
        jet_inputs = np.array([]).reshape((-1, n_jet_inputs))
        set_JetInputs(empty_ew, jet_inputs)
        empty_ew.selected_index = 0
        FormJets.produce_summary(empty_ew)
        with open(path, 'r') as summary_file:
            text = summary_file.readlines()
        assert text[0][0] == '#'
        assert '0' in text[0]
        assert len(text) == 1
        # an event filled with 0
        num_tracks = 20
        jet_inputs = np.zeros((num_tracks, n_jet_inputs))
        set_JetInputs(empty_ew, jet_inputs)
        empty_ew.selected_index = 0
        FormJets.produce_summary(empty_ew)
        with open(path, 'r') as summary_file:
            text = summary_file.readlines()
        assert text[0][0] == '#'
        assert '0' in text[0]
        assert len(text) == num_tracks + 1
        content = np.genfromtxt(path)
        tst.assert_allclose(content[:, 0], np.arange(num_tracks))
        tst.assert_allclose(content[:, 1:], np.zeros((num_tracks, 4)))
        # an event filled with random numbers
        num_tracks = 20
        jet_inputs = np.random.rand(num_tracks, n_jet_inputs)
        set_JetInputs(empty_ew, jet_inputs)
        empty_ew.selected_index = 0
        FormJets.produce_summary(empty_ew)
        with open(path, 'r') as summary_file:
            text = summary_file.readlines()
        assert text[0][0] == '#'
        assert '0' in text[0]
        assert len(text) == num_tracks + 1
        content = np.genfromtxt(path)
        input_idx = [4, 5, 6, 3]
        tst.assert_allclose(content[:, 0], np.arange(num_tracks))
        tst.assert_allclose(content[:, 1:], jet_inputs[:, input_idx])

# FOr some reason this dosn't work in the testing framework
#def test_run_FastJet():
#    # ignoring warnings here
#    with warnings.catch_warnings():
#        warnings.simplefilter('ignore')
#        with TempTestDir("fastjet") as dir_name:
#            empty_name = "empty.awkd"
#            empty_path = os.path.join(dir_name, empty_name)
#            empty_ew = Components.EventWise(dir_name, empty_name)
#            # can run fast jets via summary files or the pipe
#            def make_jets3(eventWise, DeltaR, ExpofPTMultiplier, ints, floats):
#                set_JetInputs(eventWise, floats)
#                eventWise.selected_index = 0
#                return FormJets.run_FastJet(eventWise, DeltaR, ExpofPTMultiplier, use_pipe=False)
#            clustering_algorithm(empty_ew, make_jets3, compare_distance=False)
#            def make_jets4(eventWise, DeltaR, ExpofPTMultiplier, ints, floats):
#                set_JetInputs(eventWise, floats)
#                eventWise.selected_index = 0
#                return FormJets.run_FastJet(eventWise, DeltaR, ExpofPTMultiplier, use_pipe=True)
#            clustering_algorithm(empty_ew, make_jets4, compare_distance=False)


def test_cluster_multiapply():
    # event 1
    jet_class = FormJets.SpectralMean
    jet_params = {}
    for i in range(5):
        n_rows = 4
        floats1 = np.random.random((n_rows, 8))
        # set distance to 0
        floats1[:, -1] = 0.
        for row in floats1:
            SimpleClusterSamples.fill_angular(row)
        # event 2
        n_rows = 5
        floats2 = np.random.random((n_rows, 8))
        # set distance to 0
        floats2[:, -1] = 0.
        for row in floats2:
            SimpleClusterSamples.fill_angular(row)
        # try with the first only
        with TempTestDir("tst") as dir_name:
            eventWise = Components.EventWise(dir_name, "tmp.awkd")
            set_JetInputs(eventWise, floats1)
            eventWise.selected_index = 0
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            jets = jet_class(eventWise, dict_jet_params=jet_params, assign=True)
            end_int1 = np.array(jets._ints)
            end_float1 = np.array(jets._floats)
        # try with the second only
        with TempTestDir("tst") as dir_name:
            eventWise = Components.EventWise(dir_name, "tmp.awkd")
            set_JetInputs(eventWise, floats2)
            eventWise.selected_index = 0
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            jets = jet_class(eventWise, dict_jet_params=jet_params, assign=True)
            end_int2 = np.array(jets._ints)
            end_float2 = np.array(jets._floats)
        # do both together with multiapply
        columns = [name.replace("Pseudojet", "JetInputs") for name in FormJets.PseudoJet.float_columns
                   if "Distance" not in name]
        contents = {name: awkward.fromiter([floats1[:, i], floats2[:, i]])
                    for i, name in enumerate(columns)}
        columns.append("JetInputs_SourceIdx")
        contents["JetInputs_SourceIdx"] = awkward.fromiter([np.arange(len(floats1)),
                                                            np.arange(len(floats2))])
        with TempTestDir("tst") as dir_name:
            eventWise = Components.EventWise(dir_name, "tmp.awkd")
            eventWise.append(**contents)
            jet_name = "TestJet"
            finished = FormJets.cluster_multiapply(eventWise, jet_class, jet_params,
                                                   jet_name, silent=True)
            assert finished, "Multiapply did not finish the two clusters"
            file_name = os.path.join(dir_name, "tmp.awkd")
            jets1 = jet_class.multi_from_file(file_name, 0, jet_name)
            multi_ints1 = np.vstack([jet._ints for jet in jets1])
            multi_floats1 = np.vstack([jet._floats for jet in jets1])
            SimpleClusterSamples.match_ints_floats(end_int1, end_float1, multi_ints1, multi_floats1)
            jets2 = jet_class.multi_from_file(file_name, 1, jet_name)
            multi_ints2 = np.vstack([jet._ints for jet in jets2])
            multi_floats2 = np.vstack([jet._floats for jet in jets2])
            SimpleClusterSamples.match_ints_floats(end_int2, end_float2, multi_ints2, multi_floats2)


def test_check_hyperparameters():
    params1 = {'DeltaR': .2, 'NumEigenvectors': np.inf,
               'ExpofPTPosition': 'input', 'ExpofPTMultiplier': 0,
               'AffinityType': 'exponent', 'AffinityCutoff': None,
               'Laplacien': 'unnormalised',
               'PhyDistance': 'angular', 'StoppingCondition': 'standard'}
    FormJets.check_hyperparameters(FormJets.Spectral, params1)
    params1['DeltaR'] = -1
    with pytest.raises(ValueError):
        FormJets.check_hyperparameters(FormJets.Spectral, params1)
    # Traditional should not have all these params
    params1['DeltaR'] = 1
    with pytest.raises(ValueError):
        FormJets.check_hyperparameters(FormJets.Traditional, params1)
    params1['NumEigenvectors'] = 3
    params1['AffinityCutoff'] = ('knn', 3)
    FormJets.check_hyperparameters(FormJets.Spectral, params1)
    del params1['NumEigenvectors']
    FormJets.check_hyperparameters(FormJets.Spectral, params1)



def test_get_jet_names_params():
    jet_class = FormJets.SpectralMean
    jet_paramsA = {'DeltaR': 0.4, 'AffinityCutoff': ('distance', 2.3), 
                   'ExpofPTPosition': 'input'}
    floats = np.random.random((3, 8))
    # set distance to 0
    floats[:, -1] = 0.
    for row in floats:
        SimpleClusterSamples.fill_angular(row)
    # need to keep the eventwise file around
    with TempTestDir("tst") as dir_name:
        eventWise = Components.EventWise(dir_name, "tmp.awkd")
        set_JetInputs(eventWise, floats)
        eventWise.selected_index = 0
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            jets = jet_class(eventWise, dict_jet_params=jet_paramsA, assign=True)
            jets = jets.split()
            jets[0].write_event(jets, jet_name="AAJet", event_index=0, eventWise=eventWise)
            jet_paramsB = {k: v for k, v in jet_paramsA.items()}
            jet_paramsB['AffinityCutoff'] = None
            eventWise.selected_index = 0
            jets = jet_class(eventWise, dict_jet_params=jet_paramsB, assign=True)
            jets = jets.split()
            jets[0].write_event(jets, jet_name="BBJet", event_index=0)
        jet_names = FormJets.get_jet_names(eventWise)
        assert "AAJet" in jet_names
        assert "BBJet" in jet_names
        assert len(jet_names) == 2
        found_paramsA = FormJets.get_jet_params(eventWise, "AAJet", add_defaults=True)
        for name in jet_paramsA:
            assert found_paramsA[name] == jet_paramsA[name]
        found_paramsB = FormJets.get_jet_params(eventWise, "BBJet", add_defaults=False)
        for name in jet_paramsB:
            assert found_paramsB[name] == jet_paramsB[name]


#def test_check_plots_run():
#    pass
