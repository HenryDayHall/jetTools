""" Tests for the PDG names module """
import numpy.testing as tst
from tree_tagger import PDGNames


def test_Identities():
    ids = PDGNames.Identities()
    str(ids)  # check the string method runs
    known_particles = [{'id': 5, 'name': 'b', 'charge': -1/3, 'spin':1/2},
                       {'id': -5, 'name': 'bbar', 'charge': 1/3, 'spin':1/2},
                       {'id': 25, 'name': 'h0', 'charge': 0, 'spin': 0}]
    for particle in known_particles:
        found = ids[particle['id']]
        for key in particle:
            assert particle[key] == found[key], f"For test particle {particle} found {found}"
    assert 'cbar' == ids.antiNames[4]
    assert 'cbar' == ids.antiNames[-4]
    assert 2/3 == ids.charges[4]
    assert -2/3 == ids.charges[-4]


def test_IDConverter():
    ids = PDGNames.IDConverter()
    assert 'c' == ids[4]
    assert 'cbar' == ids[-4]
    assert 2/3 == ids.charges[4]
    assert -2/3 == ids.charges[-4]
    # try somethign that won't be found
    large_num = 10**100
    assert large_num == ids[large_num]

def test_match():
    # first try getting an exact match
    desired = 'b'
    pid_list = [3, 5, -5, 5122, 111]
    tst.assert_allclose(PDGNames.match(pid_list, desired, False),
                        [False, True, False, False, False])
    tst.assert_allclose(PDGNames.match(pid_list, desired, True),
                        [False, True, True, True, False])
    tst.assert_allclose(PDGNames.match([], desired, True), [])


