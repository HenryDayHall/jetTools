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
        energy = 1.
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
    n_random_tries = 100
    n_acceptable_fails = 5
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
def make_simple_jets(floats, jet_params={}, jet_class=FormJets.PseudoJet, **kwargs):
    with TempTestDir("tst") as dir_name:
        ew = Components.EventWise(dir_name, "tmp.awkd")
        set_JetInputs(ew, floats)
        ew.selected_index = 0
    jets = jet_class(ew, dict_jet_params=jet_params, assign=False, **kwargs)
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
        FormJets.Spectral.write_event([jets])
        # check what got written is correct
        jet_name = jets.jet_name
        jet_columns = [name.split('_', 1)[1] for name in ew.columns
                       if name.startswith(jet_name)]
        for col in jet_columns:
            found = getattr(ew, jet_name + '_' + col)
            is_int = isinstance(next(found), int)
            col_num = getattr(jets, '_'+col+'_col')
            expected = ints[:, col_num] if is_int else floats[:, col_num]
            tst.assert_allclose(found, expected, err_msg=f'Missmatch in {col}, is_int={is_int}')
        for key in jet_params:
            found = getattr(ew, jet_name+'_'+key)
            assert found == jet_params[key], \
                    f"{key}; Found {found}, expected {jet_params[key]}"


def test_calculate_eigenspace():
    n_rows = 4
    ints = np.zeros((n_rows, 5), dtype=int) -1
    for i in range(5):
        floats = np.random.random((n_rows, 8))
        # set distance to 0
        floats[:, -1] = 0.
        for row in floats:
            SimpleClusterSamples.fill_angular(row)
        for ltype in ('unnormalised', 'symmetric'):
            jets = make_simple_jets(floats, {'Laplacien': ltype}, FormJets.Spectral)
            # check that the eigenvectors and the eigenvalues have the correct relationship
            # with the laplacien
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
    mask = distances != 0
    if param_dict['AffinityType'] == 'linear':
        expected = max(distances) - distances
    elif param_dict['AffinityType'] == 'inverse':
        expected[mask] = 1/distances[mask]
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
            # want to keep the largest affinities and the smallest ones have the lowest numbers
            to_keep = len(expected) - to_keep
            order = np.argsort(expected, axis=1)
            expected[order < to_keep] = 0
        elif param_dict['AffinityCutoff'][0] == 'distance':
            expected[expected < to_keep] == 0
        else:
            raise KeyError
    if len(expected) == 0:
        expected = expected.reshape((1, 0))
    np.fill_diagonal(expected, 0)
    tst.assert_allclose(jets._affinity, expected, err_msg=f"Unexpected affinity for jets;\n{param_dict}\n Found distances\n {distances}")
internal_calculate_affinity.valid_one = True
internal_calculate_affinity.valid_zero = True


def test_calculate_distances():
    # check results have correct form
    pass


def test_step_assign_parents():
    # check results have correct form
    pass

def test_Spectral_internal():
    # testing Pseudojet functions, but creating Spectral jets
    # as Pseudojet should not be directly created and Traditional lack support for all options
    apply_internal(FormJets.Spectral, internal_calculate_affinity, additional_jet_params={'StoppingCondition': 'standard'})
    apply_internal(FormJets.Spectral, internal_calculate_affinity, additional_jet_params={'StoppingCondition': 'beamparticle'})
    
# Splitting ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SpectralMean ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SpectralFull ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
#

