# TODO need to rotat the jets
""" Preprocessig module takes prepared datafiles and performed generic processing tasks requird beofre data can be run as a net """
import numpy as np
from ipdb import set_trace as st
from sklearn import preprocessing
import os
import awkward
from tree_tagger import Components
from tree_tagger.Components import flatten, apply_array_func

def phi_rotation(eventWise):
    content = {}
    # pick out the columns to transform
    pxy_cols = [(c, c.replace('Px', 'Py').replace('px', 'py'))
                for c in eventWise.columns if "px" in c.lower()]
    phi_cols = [c for c in eventWise.columns if "phi" in c.lower()]
    content = {c:[] for c in phi_cols}
    for px_name, py_name in pxy_cols:
        content[px_name] = []
        content[py_name] = []
    # now rotate the events about the z axis
    # so that the overall momentum is down the x axis
    children = eventWise.Children
    no_decendants = apply_array_func(lambda lst: not bool(len(lst)), children)
    pxs = eventWise.Px
    pys = eventWise.Py
    def leaf_sum(values, no_dez):
        return np.sum(values[no_dez])
    px_sums = apply_array_func(leaf_sum, pxs, no_decendants)
    py_sums = apply_array_func(leaf_sum, pys, no_decendants)
    for event_n, (px, py) in enumerate(zip(px_sums, py_sums)):
        eventWise.selected_index = event_n
        angle = np.arctan(py/px)
        cos = np.cos(angle)
        sin = np.sin(angle)
        def rotate_x(xs, ys):
            return xs*cos - ys*sin
        def rotate_y(xs, ys):
            return xs*sin + ys*cos
        def rotate_phi(phis):
            return Components.confine_angle(phis - angle)
        for px_name, py_name in pxy_cols:
            this_pxs = getattr(eventWise, px_name)
            this_pys = getattr(eventWise, py_name)
            content[px_name].append(apply_array_func(rotate_x, this_pxs, this_pys))
            content[py_name].append(apply_array_func(rotate_y, this_pxs, this_pys))
        for phi_name in phi_cols:
            this_phis = getattr(eventWise, phi_name)
            content[phi_name].append(apply_array_func(rotate_phi, this_phis))
    content = {name: awkward.fromiter(a) for name, a in content.items()}
    columns = sorted(content.keys())
    eventWise.append(columns, content)


def normalize_jets(eventWise, jet_name):
    eventWise.selected_index = None
    jet_cols = [c for c in eventWise.columns if jet_name in c]
    normed_outs = {}
    for name in jet_cols:
        values = getattr(eventWise, name)
        flat_iter = flatten(values)
        first = next(flat_iter)
        # we ony want to normalise flaot columns
        if isinstance(first, (float, np.float)):
            print(f"Norming {name}")
            flat_values = [first] + [i for i in flat_iter]
            transformer = preprocessing.RobustScaler().fit(flat_values)
            out = apply_array_func(transformer.transform, values)
            normed_outs[name + "_normed"] = out
    columns = sorted(normed_outs.keys())
    eventWise.append(columns, normed_outs)


def make_targets(eventWise, jet_name):
    """ anything with at least one tag is considered signal """
    truth_tags = getattr(eventWise, jet_name+"_Tags")
    def target_func(array):
        return len(array) > 0
    contents = {jet_name + "_Target": apply_array_func(target_func, truth_tags)}
    columns = sorted(contents.keys())
    eventWise.append(columns, contents)


def event_wide_observables(eventWise):
    contents = {}
    cumulative_columns = ["Energy", "Rapidity", "PT",
                          "Px", "Py", "Pz"]
    for name  in cumulative_columns:
        values = getattr(eventWise, name)
        contents["Event_Sum"+name] = apply_array_func(np.sum, values)
        contents["Event_Ave"+name] = apply_array_func(np.mean, values)
        contents["Event_Std"+name] = apply_array_func(np.std, values)
    columns = sorted(contents.keys())
    eventWise.append(columns, contents)


def jet_wide_observables(eventWise, jet_name):
    # calculate averages and num hits
    eventWise.selected_index = None
    cumulative_components = ["PT", "Rapidity", "Phi", "Energy",
                             "Px", "Py", "Pz",
                             "JoinDistance"]
    contents = {}
    for name in cumulative_components:
        col_name = jet_name + '_' + name
        values = getattr(eventWise, col_name)
        contents[jet_name+"_Sum"+name] = apply_array_func(np.sum, values)
        contents[jet_name+"_Ave"+name] = apply_array_func(np.mean, values)
        contents[jet_name+"_Std"+name] = apply_array_func(np.std, values)
    # num_hits
    contents[jet_name+"_size"] = apply_array_func(len, values)
    columns = sorted(contents.keys())
    eventWise.append(columns, contents)

