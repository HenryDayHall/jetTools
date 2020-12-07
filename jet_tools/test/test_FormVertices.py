""" test the form vertices module """
from jet_tools.tree_tagger import FormVertices
import numpy as np
from numpy import testing as tst
from ipdb import set_trace as st

def test_closest_approch():
    a1 = [0, 0, 0]
    a2 = [0, 0, 1]
    a3 = [1, 1, 1]

    x_hat = [1, 0, 0]
    y_hat = [0, 1, 0]
    diag = [1, 1, 0]
    
    start_points = np.array([     a1,    a1,    a2,    a3,    a1])
    direction_vectors = np.array([x_hat, y_hat, x_hat, x_hat, diag])
    expected_out = np.array([[np.nan,     0., np.nan, np.nan, 0.],
                             [0.,     np.nan,     0.,    -1., 0.],
                             [np.nan,     0., np.nan, np.nan, 0.],
                             [np.nan,     1., np.nan, np.nan, 1.],
                             [0.,         0.,     0.,     0., np.nan]])
    receved_out = FormVertices.closest_approches(start_points, direction_vectors)
    #receved_out2 = FormVertices.closest_approches_np(start_points, direction_vectors)
    # don't check nan parts
    mask = ~np.isnan(expected_out)
    tst.assert_allclose(expected_out[mask], receved_out[mask])
    expected_distances = np.array([[    np.nan, 0., 1., np.sqrt(2), 0.],
                                   [        0., np.nan, 1., 1., 0.],
                                   [        1., 1., np.nan, 1., 1.],
                                   [np.sqrt(2), 1., 1., np.nan, 1.],
                                   [        0., 0., 1.,     1., np.nan]])
    expected_distances2 = expected_distances**2
    expected_midpoints = np.array([[[np.nan, np.nan, np.nan], [0., 0., 0.], [np.nan, 0., 0.5], [np.nan, 0.5, 0.5], [0., 0., 0.]],
                                   [[0., 0., 0.], [np.nan, np.nan, np.nan], [0., 0., 0.5], [0., 1., 0.5], [0., 0., 0.]],
                                   [[np.nan, 0., 0.5], [0., 0., 0.5], [np.nan, np.nan, np.nan], [np.nan, 0.5, 1.], [0., 0., 0.5]],
                                   [[np.nan, 0.5, 0.5], [0., 1., 0.5], [np.nan, 0.5, 1.], [np.nan, np.nan, np.nan], [1., 1., 0.5]],
                                   [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.5], [1., 1., 0.5], [np.nan, np.nan, np.nan]]])
    receved_distances2, receved_midpoints = FormVertices.distance2_midpoints(start_points, direction_vectors, receved_out)
    mask = ~np.isnan(expected_distances2)
    tst.assert_allclose(expected_distances2[mask], receved_distances2[mask])
    mask = ~np.isnan(expected_midpoints)
    tst.assert_allclose(expected_midpoints[mask], receved_midpoints[mask])




