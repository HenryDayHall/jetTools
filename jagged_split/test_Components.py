import numpy as np
import os
from ipdb import set_trace as st
from numpy import testing as tst
import pytest
import Components
from tools import TempTestDir
import awkward

def test_split():
    with TempTestDir("tst") as dir_name:
        save_name = "test.awkd"
        # try with 10 events
        content_1 = awkward.fromiter(np.arange(10))
        content_2 = awkward.fromiter(np.random.rand(10))
        content_3 = awkward.fromiter([np.random.rand(np.random.randint(5)) for _ in range(10)])
        ew = Components.EventWise(dir_name, save_name, {'C1': content_1, 'C2': content_2, 'C3': content_3})
        # check nothing changes in the original
        tst.assert_allclose(ew.contents["C1"], content_1)
        tst.assert_allclose(ew.contents["C2"], content_2)
        tst.assert_allclose(ew.contents["C3"].flatten(), content_3.flatten())
        paths = ew.split([0, 5, 7, 7], [5, 7, 7, 10], "c1", "dog")
        tst.assert_allclose(ew.contents["C1"], content_1)
        tst.assert_allclose(ew.contents["C2"], content_2)
        tst.assert_allclose(ew.contents["C3"].flatten(), content_3.flatten())
        # check the segments contain what They should
        ew0, ew1, ew2, ew3 = paths
        ew0 = Components.EventWise.from_file(ew0)
        tst.assert_allclose(ew0.contents["C1"], content_1[:5])
        tst.assert_allclose(ew0.contents["C2"], content_2[:5])
        tst.assert_allclose(ew0.contents["C3"].flatten(), content_3[:5].flatten())
        ew1 = Components.EventWise.from_file(ew1)
        tst.assert_allclose(ew1.contents["C1"], content_1[5:7])
        tst.assert_allclose(ew1.contents["C2"], content_2[5:7])
        tst.assert_allclose(ew1.contents["C3"].flatten(), content_3[5:7].flatten())
        assert ew2 is None
        ew3 = Components.EventWise.from_file(ew3)
        tst.assert_allclose(ew3.contents["C1"], content_1[7:])
        tst.assert_allclose(ew3.contents["C2"], content_2[7:])
        tst.assert_allclose(ew3.contents["C3"].flatten(), content_3[7:].flatten())

if __name__ == '__main__':
    test_split()
