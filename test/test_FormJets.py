from numpy import testing as tst
from ipdb import set_trace as st
import numpy as np
from tree_tagger import Components, FormJets
from tools import TempTestDir


class SimpleClusterSamples:
    config_1 = {'deltaR': 1., 'exponent_multiplyer': 0}
    config_2 = {'deltaR': 1., 'exponent_multiplyer': 1}
    config_3 = {'deltaR': 1., 'exponent_multiplyer': -1}
    config_4 = {'deltaR': .4, 'exponent_multiplyer': 0}
    config_5 = {'deltaR': .4, 'exponent_multiplyer': 1}
    config_6 = {'deltaR': .4, 'exponent_multiplyer': -1}
    empty_inp = {'ints': np.array([]).reshape((-1, 5)), 'floats': np.array([]).reshape((-1, 8))}
    one_inp = {'ints': np.array([[0, -1, -1, -1, -1]]),
               'floats': np.array([[1., 0., 0., 0., 0., 0., 0., 0.]])}
    two_degenerate = {'ints': np.array([[0, -1, -1, -1, -1],
                                        [1, -1, -1, -1, -1]]),
                      'floats': np.array([[1., 0., 0., 1., 1., 0., 0., 0.],
                                          [1., 0., 0., 1., 1., 0., 0., 0.]])}
    degenerate_join = {'ints': np.array([[0, -1, -1, -1, -1],
                                         [1, -1, -1, -1, -1],
                                         [2, -1, 0, 1, 0]]),
                       'floats': np.array([[1., 0., 0., 1., 1., 0., 0., 0.],
                                           [1., 0., 0., 1., 1., 0., 0., 0.],
                                           [2., 0., 0., 2., 2., 0., 0., 0.]])}
    two_close = {'ints': np.array([[0, -1, -1, -1, -1],
                                   [1, -1, -1, -1, -1]]),
                 'floats': np.array([[1., 0., 0., 1., 1., 0., 0., 0.],
                                     [1., 0., 0.1, 1., np.cos(0.1), np.sin(0.1), 0., 0.]])}
    close_join = {'ints': np.array([[0, -1, -1, -1, -1],
                                    [1, -1, -1, -1, -1],
                                    [2, -1, 0, 1, 0]]),
                  'floats': np.array([[1., 0., 0., 1., 1., 0., 0., 0.],
                                      [1., 0., 0.1, 1., np.cos(0.1), np.sin(0.1), 0., 0.],
                                      [2.*np.cos(0.05), 0., 0.05, 2., 1. + np.cos(0.1), np.sin(0.1), 0., 0.1]])}
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
        cls.fill_floats(two['floats'][0])
        cls.fill_floats(two['floats'][1])
        distance = np.min(np.power(two['floats'][:, 0], 2*config['exponent_multiplyer'])) * \
                   np.linalg.norm(two['floats'][0, [1,2]] - two['floats'][1, [1,2]])
        beam_distance = config['deltaR']*np.min(np.power(two['floats'][:, 0], 2*config['exponent_multiplyer']))
        if distance > beam_distance:
            # don't join
            return two, two
        else:
            # form the joined row
            joined_floats = two['floats'][0] + two['floats'][1]
            cls.fill_floats(joined_floats, False)
            joined_two = {'ints': np.array([[0, -1, -1, -1, -1],
                                            [1, -1, -1, -1, -1],
                                            [2, -1, 0, 1, 0]]),
                          'floats': np.vstack((two['floats'], joined_floats))}
            return two, joined_two

    @classmethod
    def fill_floats(cls, floats, change_energy=True):
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



def test_PseudoJet():
    with TempTestDir("pseudojet") as dir_name:
        config_list = [getattr(SimpleClusterSamples, f"config_{i}") for i in range(1, 7)]
        # there are two ways to construct a pseudojet
        # 1. telling it ints and floats as constructor arguments
        # 2. letting it read ints and floats from an eventwise object
        # We start with the first method and do checks on the behavior of pseudojet
        empty_name = "empty.awkd"
        empty_ew = Components.EventWise(dir_name, empty_name)
        for config in config_list:
            # an empty set of ints should safely return an empty pseudojet
            empty_pseudojet = FormJets.PseudoJet(empty_ew, config['deltaR'], config['exponent_multiplyer'],
                                                 ints = SimpleClusterSamples.empty_inp['ints'],
                                                 floats = SimpleClusterSamples.empty_inp['floats'])
            empty_pseudojet.assign_parents()
            assert len(empty_pseudojet._ints) == 0
            assert len(empty_pseudojet._floats) == 0
            jets = empty_pseudojet.split()
            assert len(jets) == 1
            assert len(jets[0]._ints) == 0
            assert len(jets[0]._floats) == 0
            # one track should return one pseudojet
            one_pseudojet = FormJets.PseudoJet(empty_ew, config['deltaR'], config['exponent_multiplyer'],
                                               ints = SimpleClusterSamples.one_inp['ints'],
                                               floats = SimpleClusterSamples.one_inp['floats'])
            one_pseudojet.assign_parents()
            tst.assert_allclose(one_pseudojet._ints, SimpleClusterSamples.one_inp['ints'])
            tst.assert_allclose(one_pseudojet._floats, SimpleClusterSamples.one_inp['floats'])
            jets = one_pseudojet.split()
            assert len(jets) == 1
            tst.assert_allclose(jets[0]._ints, SimpleClusterSamples.one_inp['ints'])
            tst.assert_allclose(jets[0]._floats, SimpleClusterSamples.one_inp['floats'])


