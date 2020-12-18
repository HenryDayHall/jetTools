""" tests to evaluate the FormShower module """
import os
from ipdb import set_trace as st
from jet_tools.src import FormShower, DrawTrees, Components, PDGNames
import numpy.testing as tst
import numpy as np
from jet_tools.test.tools import TempTestDir
import pytest
import awkward


def test_Shower():
    # try making an empty shower
    empty = FormShower.Shower([], [], [], [])
    assert len(empty) == 0
    assert empty.n_particles == 0
    assert len(empty.root_idxs) == 0 
    empty.find_ranks()
    assert len(empty.ranks) == 0
    empty_graph = empty.graph()
    assert isinstance(empty_graph, DrawTrees.DotGraph)
    with pytest.raises(AttributeError):
        empty.outside_connections
    with pytest.raises(AttributeError):
        empty.roots
    assert len(empty.outside_connection_idxs) == 0
    assert len(empty.ends) == 0
    assert len(empty.flavour) == 0
    # now a simple shower where A -> B, C
    particle_idxs = [1, 2, 3]
    parents = [[], [1], [1]]
    children = [[2, 3], [], []]
    labels = ['A', 'B', 'C']
    amalgam = True
    simple = FormShower.Shower(particle_idxs, parents, children, labels, amalgam)
    assert len(simple) == 3
    assert simple.n_particles == 3
    tst.assert_allclose(simple.root_idxs, [1])
    simple.find_ranks()
    tst.assert_allclose(simple.ranks, [0, 1, 1])
    simple_graph = simple.graph()
    assert isinstance(simple_graph, DrawTrees.DotGraph)
    assert len(simple.outside_connection_idxs) == 0
    tst.assert_allclose(simple.ends, [2, 3])
    particle_idxs = [1, 2, 3, 4]
    parents = [[], [1], [1], [5]]
    children = [[2, 3], [6], [], []]
    labels = ['A', 'B', 'C', 'D']
    amalgam = True
    double_root = FormShower.Shower(particle_idxs, parents, children, labels, amalgam)
    tst.assert_allclose(sorted(double_root.root_idxs), [1, 4])
    tst.assert_allclose(sorted(double_root.outside_connection_idxs), [4])
    with pytest.raises(AssertionError):
        amalgam = False
        double_root = FormShower.Shower(particle_idxs, parents, children, labels, amalgam)

    particle_idxs = [2, 6, 7]
    parents = [[1], [2], [6]]
    children = [[6], [7], []]
    labels = ['B', 'E', 'F']
    amalgam = False
    addition = FormShower.Shower(particle_idxs, parents, children, labels, amalgam)
    double_root.amalgamate(addition)
    assert len(double_root) == 6
    tst.assert_allclose(sorted(double_root.root_idxs), [1, 4])
    tst.assert_allclose(sorted(double_root.outside_connection_idxs), [4])


def test_descendant_idxs():
    #           0   1       2    3       4   5   6          7   8   9   10
    children = [[], [2, 3], [5], [6, 5], [], [], [7, 8, 9], [], [], [], []]
    with TempTestDir("tst") as dir_name:
        eventWise = Components.EventWise(os.path.join(dir_name, "tmp.awkd"))
        eventWise.append(Children=[awkward.fromiter(children)])
        eventWise.selected_index = 0
        tst.assert_allclose(list(FormShower.descendant_idxs(eventWise, 0)), [0])
        tst.assert_allclose(list(FormShower.descendant_idxs(eventWise, 2)), [5])
        tst.assert_allclose(sorted(FormShower.descendant_idxs(eventWise, 1)),
                            [5, 7, 8, 9])


def test_append_b_idxs():
    #           0   1       2    3       4   5   6          7   8   9   10
    children = [[], [2, 3], [5], [6, 5], [], [], [7, 8, 9], [], [], [], []]
    mcpid =    [4,  5,      5,   3,      2,  1,  -5,        1,  7,  11, 12]
    expected = [2, 6]
    with TempTestDir("tst") as dir_name:
        eventWise = Components.EventWise(os.path.join(dir_name, "tmp.awkd"))
        eventWise.append(Children=[awkward.fromiter(children)],
                         MCPID=[awkward.fromiter(mcpid)])
        found = FormShower.append_b_idxs(eventWise, append=False)
        tst.assert_allclose(sorted(found['BQuarkIdx'][0]), expected)
        found = FormShower.append_b_idxs(eventWise, silent=False, append=True)
        eventWise.selected_index = 0
        tst.assert_allclose(sorted(eventWise.BQuarkIdx), expected)
        found = FormShower.append_b_idxs(eventWise, silent=False, append=True)
        assert found == True


def test_upper_layers():
    # will need an eventwise with Parents, Children, MCPID
    # layer     -1  0       1    1      -1   2   2          3   3   3   -1
    # idx       0   1       2    3       4   5       6          7   8   9   10
    children = [[], [2, 3], [5], [6, 5], [], [],     [7, 8, 9], [], [], [], []]
    parents =  [[], [],     [1], [1],    [], [2, 3], [3],       [6],[6],[6],[]]
    mcpid =    [4,  5,      5,   3,      2,  1,      -5,        -1, 7,  11, 12]
    expected = [2, 6]
    labeler = PDGNames.IDConverter()
    with TempTestDir("tst") as dir_name:
        eventWise = Components.EventWise(os.path.join(dir_name, "tmp.awkd"))
        eventWise.append(Children=[awkward.fromiter(children)],
                         Parents=[awkward.fromiter(parents)],
                         MCPID=[awkward.fromiter(mcpid)])
        eventWise.selected_index = 0
        expected_particle_idx = [0, 1, 2, 3, 4, 10]
        expected_children = awkward.fromiter([children[i] for i in expected_particle_idx]).flatten()
        expected_parents = awkward.fromiter([parents[i] for i in expected_particle_idx]).flatten()
        expected_labels = [labeler[mcpid[i]] for i in expected_particle_idx]
        shower = FormShower.upper_layers(eventWise, n_layers=2)
        order = np.argsort(shower.particle_idxs)
        tst.assert_allclose(shower.particle_idxs[order], expected_particle_idx)
        tst.assert_allclose(awkward.fromiter(shower.children[order]).flatten(),
                            expected_children)
        tst.assert_allclose(awkward.fromiter(shower.parents[order]).flatten(),
                            expected_parents)
        for a, b in zip(shower.labels[order], expected_labels):
            assert a == b
        # try with capture pids
        expected_particle_idx = [0, 1, 2, 3, 4, 5, 6, 10]
        expected_children = awkward.fromiter([children[i] for i in expected_particle_idx]).flatten()
        expected_parents = awkward.fromiter([parents[i] for i in expected_particle_idx]).flatten()
        expected_labels = [labeler[mcpid[i]] for i in expected_particle_idx]
        shower = FormShower.upper_layers(eventWise, n_layers=2, capture_pids=[1])
        order = np.argsort(shower.particle_idxs)
        tst.assert_allclose(shower.particle_idxs[order], expected_particle_idx)
        tst.assert_allclose(awkward.fromiter(shower.children[order]).flatten(),
                            expected_children)
        tst.assert_allclose(awkward.fromiter(shower.parents[order]).flatten(),
                            expected_parents)
        for a, b in zip(shower.labels[order], expected_labels):
            assert a == b


def test_get_roots():
    # test the empty list
    particle_ids = []; parents = []
    found = FormShower.get_roots(particle_ids, parents)
    assert len(found) == 0
    # test one particle
    particle_ids = [2]; parents = [[5]]
    found = FormShower.get_roots(particle_ids, parents)
    tst.assert_allclose(found, particle_ids)
    # test one root, one trailing
    particle_ids = [2, 3]
    parents = [[3], [5]]
    found = FormShower.get_roots(particle_ids, parents)
    tst.assert_allclose(found, [3])
    # test two roots
    particle_ids = [2, 3]
    parents = [[], [5]]
    found = FormShower.get_roots(particle_ids, parents)
    tst.assert_allclose(sorted(found), particle_ids)
    # test chain
    particle_ids = [2, 3, 5, 11]
    parents = [[], [2, 11], [3], [10]]
    found = FormShower.get_roots(particle_ids, parents)
    tst.assert_allclose(sorted(found), [2, 11])


def test_get_showers():
    def check(children, parents, mcpid, exclude_pids, expected):
        with TempTestDir("tst") as dir_name:
            eventWise = Components.EventWise(os.path.join(dir_name, "tmp.awkd"))
            eventWise.append(Children=[awkward.fromiter(children)],
                             Parents=[awkward.fromiter(parents)],
                             MCPID=[awkward.fromiter(mcpid)])
            eventWise.selected_index = 0
            showers = FormShower.get_showers(eventWise, exclude_pids)
            err_msg = f"Showers with parentage {parents}, expected {expected}, found {[set(s.particle_idxs) for s in showers]}"
            assert len(showers) == len(expected), err_msg
            for shower in showers:
                assert set(shower.particle_idxs) in expected, err_msg
    # check the empty case
    expected = children = mcpid = parents = exclude_pids = []
    check(children, parents, mcpid, exclude_pids, expected)
    exclude_pids = [2]
    check(children, parents, mcpid, exclude_pids, expected)
    # now check one particle
    children = parents = [[]]
    mcpid = [1]
    expected = [{0}]
    check(children, parents, mcpid, exclude_pids, expected)
    # check one excluded particle
    mcpid = exclude_pids = [2]
    expected = []
    check(children, parents, mcpid, exclude_pids, expected)
    # check a full chain
    # will need an eventwise with Parents, Children, MCPID
    # idx       0   1       2    3       4   5       6          7   8   9   10
    children = [[], [2, 3], [5], [6, 5], [], [],     [7, 8, 9], [], [], [], []]
    parents =  [[], [],     [1], [1],    [], [2, 3], [3],       [6],[6],[6],[]]
    mcpid = np.ones(len(children))
    exclude_pids = [2]
    expected = [{0}, {4}, {10}, {1, 2, 3, 5, 6, 7, 8, 9}]
    check(children, parents, mcpid, exclude_pids, expected)
    # now exclude the 3rd index
    mcpid[3] = 2
    expected = [{0}, {4}, {10}, {1, 2, 5}, {6, 7, 8, 9}]
    check(children, parents, mcpid, exclude_pids, expected)


def test_event_shared_ends():
    def check(children, parents, mcpid, all_roots, expected_roots, expected_shared):
        with TempTestDir("tst") as dir_name:
            is_leaf = [len(c) == 0 for c in children]
            eventWise = Components.EventWise(os.path.join(dir_name, "tmp.awkd"))
            eventWise.append(Children=[awkward.fromiter(children)],
                             Parents=[awkward.fromiter(parents)],
                             Is_leaf=[awkward.fromiter(is_leaf)],
                             MCPID=[awkward.fromiter(mcpid)])
            eventWise.selected_index = 0
            shared_counts = np.zeros((len(all_roots), len(all_roots))).tolist()
            all_roots, shared_counts = FormShower.event_shared_ends(eventWise, all_roots, shared_counts)
            if len(all_roots) == 0:  # don't really care about the dimensions for 0 length
                assert len(expected_roots) == 0
                assert len(np.array(shared_counts).flatten()) == 0
            else:
                root_err_msg = f"Showers with parentage {parents}, expected {expected_roots}, found {all_roots}"
                count_err_msg = f"Showers with parentage {parents}, expected {expected_shared}, found {shared_counts}"
                # need to impost accending order on them
                root_order = np.argsort(all_roots)
                shared_counts = np.array(shared_counts)[root_order]
                shared_counts = shared_counts[:, root_order]
                all_roots = np.array(all_roots)[root_order]
                tst.assert_allclose(all_roots, expected_roots, err_msg=root_err_msg)
                tst.assert_allclose(shared_counts, expected_shared, err_msg=count_err_msg)
    # check the empty case
    expected_roots = children = mcpid = parents = all_roots = []
    expected_shared = [[]]
    check(children, parents, mcpid, all_roots, expected_roots, expected_shared)
    all_roots = expected_roots = [1]
    expected_shared = [[0]]
    check(children, parents, mcpid, all_roots, expected_roots, expected_shared)
    # now check one particle
    children = parents = [[]]
    mcpid = [1]
    all_roots = [1]
    expected_roots = [1]
    expected_shared = [[0]]
    check(children, parents, mcpid, all_roots, expected_roots, expected_shared)
    all_roots = []
    check(children, parents, mcpid, all_roots, expected_roots, expected_shared)
    # check a full chain
    # idx       0   1       2    3       4   5       6          7   8   9   10
    children = [[], [2, 3], [5], [6, 5], [], [],     [7, 8, 9], [], [], [], []]
    parents =  [[], [],     [1], [1],    [], [2, 3], [3],       [6],[6],[6],[]]
    mcpid =    [5,  2,      1,   1,      3,  1,      1,         1,  1,  1,  4 ]
    all_roots = [2, 3, 4, 5]
    expected_roots = [2, 3, 4, 5]
    expected_shared = np.zeros((4, 4))
    check(children, parents, mcpid, all_roots, expected_roots, expected_shared)
    all_roots = [1, 4]
    expected_roots = [1, 2, 3, 4, 5]
    expected_shared = np.zeros((5, 5))
    check(children, parents, mcpid, all_roots, expected_roots, expected_shared)
    # now finaly actually a shared root
    # idx       0   1       2    3       4   5       6          7   8   9   10
    children = [[], [2, 3], [5], [6, 5], [], [],     [7, 8, 9], [], [], [], []]
    parents =  [[], [],     [0, 1], [1],    [], [2, 3], [3],       [6],[6],[6],[]]
    mcpid =    [5,  2,      1,   1,      3,  1,      1,         1,  1,  1,  4 ]
    all_roots = [2, 3, 4, 5]
    expected_roots = [2, 3, 4, 5]
    expected_shared = np.zeros((4, 4))
    expected_shared[3, 0] += 1
    expected_shared[0, 3] += 1
    check(children, parents, mcpid, all_roots, expected_roots, expected_shared)




def test_shared_ends():
    def check(children, parents, is_leaf, mcpid, expected_roots, expected_shared):
        with TempTestDir("tst") as dir_name:
            eventWise = Components.EventWise(os.path.join(dir_name, "tmp.awkd"))
            eventWise.append(Children=awkward.fromiter(children),
                             Parents=awkward.fromiter(parents),
                             Is_leaf=awkward.fromiter(is_leaf),
                             MCPID=awkward.fromiter(mcpid))
            eventWise.selected_index = 0
            all_roots, shared_counts = FormShower.shared_ends(eventWise)
            if len(all_roots) == 0:  # don't really care about the dimensions for 0 length
                assert len(expected_roots) == 0
                assert len(np.array(shared_counts).flatten()) == 0
            else:
                root_err_msg = f"Showers with parentage {parents}, expected {expected_roots}, found {all_roots}"
                count_err_msg = f"Showers with parentage {parents}, expected {expected_shared}, found {shared_counts}"
                # need to impost accending order on them
                root_order = np.argsort(all_roots)
                shared_counts = np.array(shared_counts)[root_order]
                shared_counts = shared_counts[:, root_order]
                all_roots = np.array(all_roots)[root_order]
                tst.assert_allclose(all_roots, expected_roots, err_msg=root_err_msg)
                tst.assert_allclose(shared_counts, expected_shared, err_msg=count_err_msg)
    # check the empty case
    children = mcpid = parents = is_leaf = [[]]
    expected_roots = []
    expected_shared = [[]]
    check(children, parents, is_leaf, mcpid, expected_roots, expected_shared)
    # shared root twice over
    # idx                        0   1       2    3       4   5       6          7   8   9   10
    children = awkward.fromiter([[], [2, 3], [5], [6, 5], [], [],     [7, 8, 9], [], [], [], []])
    children = [children, children]
    parents =  awkward.fromiter([[], [],     [0, 1], [1],    [], [2, 3], [3],       [6],[6],[6],[]])
    parents = [parents, parents]
    is_leaf = [awkward.fromiter([len(c) == 0 for c in chi]) for chi in children]
    mcpid =    awkward.fromiter([5,  2,      1,   1,      3,  1,      1,         1,  1,  1,  4 ])
    mcpid = [mcpid, mcpid]
    expected_roots = [2, 3, 4, 5]
    expected_shared = np.zeros((4, 4))
    expected_shared[3, 0] += 2
    expected_shared[0, 3] += 2
    check(children, parents, is_leaf, mcpid, expected_roots, expected_shared)

