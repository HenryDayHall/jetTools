from numpy import testing as tst
import warnings
import os
from ipdb import set_trace as st
import numpy as np
from tree_tagger import Components, FormJets
from tools import TempTestDir
from test_Components import AwkdArrays
import awkward


class SimpleClusterSamples:
    config_1 = {'DeltaR': 1., 'ExponentMultiplier': 0}
    config_2 = {'DeltaR': 1., 'ExponentMultiplier': 1}
    config_3 = {'DeltaR': 1., 'ExponentMultiplier': -1}
    config_4 = {'DeltaR': .4, 'ExponentMultiplier': 0}
    config_5 = {'DeltaR': .4, 'ExponentMultiplier': 1}
    config_6 = {'DeltaR': .4, 'ExponentMultiplier': -1}
    # there is no garantee on left right child order, or global order of pseudojets
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
                                      [2.*np.cos(0.05), 0., 0.05, 2., 1. + np.cos(0.1), np.sin(0.1), 0., 0.1**2]])}
    two_oposite = {'ints': np.array([[0, -1, -1, -1, -1],
                                     [1, -1, -1, -1, -1]]),
                   'floats': np.array([[1., 0., 0., 1., 1., 0., 0., 0.],
                                       [1., 0., np.pi, 1., -1., 0., 0., 0.]])}

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
        vec_1 = np.array([np.cos(phi_1), np.sin(phi_1)])
        vec_2 = np.array([np.cos(phi_2), np.sin(phi_2)])
        angle = np.arccos(np.clip(np.dot(vec_1, vec_2), -1., 1.))
        distance = np.min(np.power(two['floats'][:, 0], 2*config['ExponentMultiplier'])) * \
                   ((two['floats'][0, 1] - two['floats'][1, 1])**2 + angle**2)
        beam_distance = (config['DeltaR']**2)*np.min(np.power(two['floats'][:, 0], 2*config['ExponentMultiplier']))
        if distance > beam_distance:
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
    def match_ints_floats(cls, ints1, floats1, ints2, floats2, compare_distance=True):
        ints_order = cls.match_ints(ints1, ints2)
        floats1 = np.array(floats1)[ints_order]
        floats2 = np.array(floats2)
        if not compare_distance:
            floats1 = floats1[:, :-1]
            floats2 = floats2[:, :-1]
        tst.assert_allclose(floats1, floats2, atol=0.0001, err_msg="Floats don't match")


def set_JetInputs(eventWise, floats):
    columns = [name.replace("Pseudojet", "JetInputs") for name in FormJets.PseudoJet.float_columns
               if "Distance" not in name]
    contents = {name: awkward.fromiter([floats[:, i]]) for i, name in enumerate(columns)}
    columns.append("JetInputs_SourceIdx")
    contents["JetInputs_SourceIdx"] = awkward.fromiter([np.arange(len(floats))])
    eventWise.append(contents)


def clustering_algorithm(empty_ew, make_pseudojets, compare_distance=True):
    # for the randomly places components, accept some error for floating point
    n_random_tries = 100
    n_acceptable_fails = 5
    config_list = [getattr(SimpleClusterSamples, f"config_{i}") for i in range(1, 7)]
    for config in config_list:
        # an empty set of ints should safely return an empty pseudojet
        empty_pseudojet = make_pseudojets(empty_ew, config['DeltaR'], config['ExponentMultiplier'],
                                             ints = SimpleClusterSamples.empty_inp['ints'],
                                             floats = SimpleClusterSamples.empty_inp['floats'])
        assert len(empty_pseudojet._ints) == 0
        assert len(empty_pseudojet._floats) == 0
        jets = empty_pseudojet.split()
        assert len(jets) == 0
        # one track should return one pseudojet
        one_pseudojet = make_pseudojets(empty_ew, config['DeltaR'], config['ExponentMultiplier'],
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
        two_pseudojet = make_pseudojets(empty_ew, config['DeltaR'], config['ExponentMultiplier'],
                                        ints = SimpleClusterSamples.two_degenerate['ints'],
                                        floats = SimpleClusterSamples.two_degenerate['floats'])
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
        two_pseudojet = make_pseudojets(empty_ew, config['DeltaR'], config['ExponentMultiplier'],
                                           ints = SimpleClusterSamples.two_close['ints'],
                                           floats = SimpleClusterSamples.two_close['floats'])
        SimpleClusterSamples.match_ints_floats(two_pseudojet._ints, two_pseudojet._floats,
                                               SimpleClusterSamples.close_join['ints'],
                                               SimpleClusterSamples.close_join['floats'],
                                               compare_distance=compare_distance)
        jets = two_pseudojet.split()
        assert len(jets) == 1
        SimpleClusterSamples.match_ints_floats(jets[0]._ints, jets[0]._floats,
                                               SimpleClusterSamples.close_join['ints'],
                                               SimpleClusterSamples.close_join['floats'],
                                               compare_distance=compare_distance)
        # two tracks far apart should not join
        two_pseudojet = make_pseudojets(empty_ew, config['DeltaR'], config['ExponentMultiplier'],
                                           ints = SimpleClusterSamples.two_oposite['ints'],
                                           floats = SimpleClusterSamples.two_oposite['floats'])
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
            pseudojets = make_pseudojets(empty_ew, config['DeltaR'], config['ExponentMultiplier'],
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
            pseudojets = make_pseudojets(empty_ew, config['DeltaR'], config['ExponentMultiplier'],
                                            ints=start['ints'], floats=start['floats'])
            jets = pseudojets.split()
            end_ints = np.vstack([j._ints for j in jets])
            end_floats = np.vstack([j._floats for j in jets])
            if len(end_ints) == len(expected_end['ints']):
                SimpleClusterSamples.match_ints_floats(pseudojets._ints, pseudojets._floats,
                                                       end_ints, end_floats)
            else:
                fails += 1
                print("Failed to join/incorrectly joined")
                if fails > n_acceptable_fails:
                    assert False, f"{fails} out of {i} incorrect clusters from homejet"


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
        def make_jets1(eventWise, DeltaR, ExponentMultiplier, ints, floats):
            pseudojets = FormJets.Traditional(eventWise, DeltaR=DeltaR, ExponentMultiplier=ExponentMultiplier, ints_floats=(ints, floats))
            pseudojets.assign_parents()
            return pseudojets
        clustering_algorithm(empty_ew, make_jets1)
        # method 2
        def make_jets2(eventWise, DeltaR, ExponentMultiplier, ints, floats):
            set_JetInputs(eventWise, floats)
            eventWise.selected_index = 0
            pseudojets = FormJets.Traditional(eventWise, DeltaR=DeltaR, ExponentMultiplier=ExponentMultiplier)
            pseudojets.assign_parents()
            return pseudojets
        clustering_algorithm(empty_ew, make_jets2)
        # test the save method
        test_inp = SimpleClusterSamples.one_inp
        name = "TestA"
        test_jet = FormJets.Traditional(empty_ew, DeltaR=1., ExponentMultiplier=-1.,
                                        ints_floats=(test_inp['ints'], test_inp['floats']), jet_name=name)
        FormJets.Traditional.write_event([test_jet], name, event_index=2)
        # read out again
        jets = FormJets.Traditional.multi_from_file(empty_path, event_idx=2, jet_name=name)
        # expect only one jet
        assert len(jets) == 1
        SimpleClusterSamples.match_ints_floats(test_inp['ints'], test_inp['floats'], jets[0]._ints, jets[0]._floats)
        # test collective properties
        test_inp = SimpleClusterSamples.empty_inp
        empty_ew.selected_index = 0
        test_jet = FormJets.Traditional(empty_ew, DeltaR=1., ExponentMultiplier=-1., ints_floats=(test_inp['ints'], test_inp['floats']))
        expected_summaries = np.full(10, np.nan)
        found_summaries = np.array([test_jet.PT, test_jet.Rapidity, test_jet.Phi, test_jet.Energy,
                                    test_jet.Px, test_jet.Py, test_jet.Pz, test_jet.JoinDistance,
                                    test_jet.Pseudorapidity, test_jet.Theta])
        tst.assert_allclose(expected_summaries, found_summaries)
        test_inp = SimpleClusterSamples.one_inp
        test_jet = FormJets.Traditional(empty_ew, DeltaR=1., ExponentMultiplier=-1., ints_floats=(test_inp['ints'], test_inp['floats']))
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
            tst.assert_allclose(getattr(ew, ji_name).tolist(), [floats.T[idx]])
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
            assert not hasattr(ew, ji_name)
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
#            def make_jets3(eventWise, DeltaR, ExponentMultiplier, ints, floats):
#                set_JetInputs(eventWise, floats)
#                eventWise.selected_index = 0
#                return FormJets.run_FastJet(eventWise, DeltaR, ExponentMultiplier, use_pipe=False)
#            clustering_algorithm(empty_ew, make_jets3, compare_distance=False)
#            def make_jets4(eventWise, DeltaR, ExponentMultiplier, ints, floats):
#                set_JetInputs(eventWise, floats)
#                eventWise.selected_index = 0
#                return FormJets.run_FastJet(eventWise, DeltaR, ExponentMultiplier, use_pipe=True)
#            clustering_algorithm(empty_ew, make_jets4, compare_distance=False)
#

