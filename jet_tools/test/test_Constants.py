""" module to test the Constants.py module """
from jet_tools.tree_tagger import Constants
import numpy as np
import pytest

numeric_classes = {'nn': 'natrual number', 'pdn': 'positive definite number', 'rn': 'real number'}
def test_is_numeric_class():
    # start with natrual numbers
    num_type = 'natrual number'
    passing = [1, 2, 100, int(np.nan_to_num(np.inf))]
    for p in passing:
        assert Constants.is_numeric_class(p, num_type)
    fail = [None, 0, -1, 0.5, np.inf]
    for f in fail:
        assert not Constants.is_numeric_class(f, num_type)
    num_type = 'positive definite number'
    passing = [1, 0.1, np.inf]
    for p in passing:
        assert Constants.is_numeric_class(p, num_type)
    fail = [None, 0, -1, np.nan]
    for f in fail:
        assert not Constants.is_numeric_class(f, num_type)
    num_type = 'real number'
    passing = [np.nan, -1, 0, 1, 0.1, np.inf]
    for p in passing:
        assert Constants.is_numeric_class(p, num_type)
    fail = [None]
    for f in fail:
        assert not Constants.is_numeric_class(f, num_type)
    with pytest.raises(ValueError):
        Constants.is_numeric_class(4, "labrador")

