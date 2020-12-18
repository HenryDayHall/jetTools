""" Testing module for the ShapeVariables module"""
from matplotlib import pyplot as plt
from jet_tools.src import ShapeVariables
import numpy as np
from ipdb import set_trace as st

def load_test_values(file_name='./test/test_shape.out'):
    # the expected start of each line
    start_length = 6
    start_key = {}
    start_key[" s  = "] = "invar_mass"
    start_key[" p3 = "] = "momentum1"
    start_key[" p4 = "] = "momentum2"
    start_key[" p5 = "] = "momentum3"
    start_key[" p6 = "] = "momentum4"
    start_key[" thr,o"] = "skip"
    start_key[" acopl"] = "skip"
    start_key[" *****"] = "new_event"
    invar_mass = []
    momenta = [[]]
    shape_variables = [[]]
    event_count = 0
    with open(file_name, 'r') as test_file:
        for line in test_file:
            start = line[:start_length]
            line_type = start_key.get(start, "shape")
            if line_type == "invar_mass":
                invar_mass.append(float(line.split('=', 1)[1]))
            elif line_type[:-1] == "momentum":
                momenta[-1].append([float(x) for x in line.split('=', 1)[1].split()])
            elif line_type == "shape":
                shape_variables[-1] += [float(x) for x in line.split()]
            elif line_type == "new_event":
                event_count += 1
                momenta.append([])
                shape_variables.append([])
            else:
                assert line_type == "skip"
    momenta = momenta[:-1]
    shape_variables = shape_variables[:-1]
    assert len(invar_mass) == event_count
    test_shape_order = ['Thrust', 'Oblateness',
                        'Heavyjetmass2', 'Lightjetmass2', 'Differencejetmass2', 'Cparameter',
                        'Sphericity', 'Alpanarity','Planarity', 'Acoplanarity',
                        'Transversethrust', 'Minor', 'Major', 'Dparameter', 'Spherocity']
    # turn the shape_variables into dictionaries
    shape_variables = [{name: val for name, val in zip(test_shape_order, line)} for line in shape_variables]
    return invar_mass, momenta, shape_variables


def mimic_test_values(momenta):
    momenta = np.array(momenta)
    py_shapes = []
    fr_shapes = []
    pyfix_shapes = []
    n_events = len(momenta)
    for i, event in enumerate(momenta):
        print(f"{i/n_events:%}", end='\r', flush=True)
        try:
            fortran_results = ShapeVariables.shape(event[:, 3],
                                                   event[:, 0], event[:, 1], event[:, 2])[1]
            fr_shapes.append(fortran_results)
        except ValueError: # tachyons
            print(f"{i} Tachyonic")
            print(event)
        else:
            python_results = ShapeVariables.python_shape(event[:, 3],
                                                         event[:, 0], event[:, 1], event[:, 2])

            py_shapes.append(python_results)
            thrust_axis = np.array([fortran_results['ThrustVector[1]'],
                                    fortran_results['ThrustVector[2]'],
                                    fortran_results['ThrustVector[3]']])
            python_results = ShapeVariables.python_shape(event[:, 3],
                                                         event[:, 0], event[:, 1], event[:, 2],
                                                         thrust_axis=thrust_axis)

            pyfix_shapes.append(python_results)
    return py_shapes, pyfix_shapes, fr_shapes


def name_fix(name):
    name = name.replace('_', '')
    name = name.replace(' ', '')
    name = name[0].capitalize() + name[1:]
    return name


def plot_shapevars(*dicts_data, labels=None):
    for data in dicts_data:
        data[:] = [{name_fix(key): val for key, val in dat.items()}
                     for dat in data]
    common_names = set(dicts_data[0][0].keys()).intersection(*[d[0].keys() for d in dicts_data])
    common_names = sorted(common_names)
    n_plots = len(common_names)
    n_cols = int(np.sqrt(n_plots))
    n_rows = int(np.ceil(n_plots/n_cols))
    fig, axarry = plt.subplots(n_rows, n_cols)
    axarry = axarry.flatten()
    if labels is None:
        labels = ["From python", "Test values"]
    colours = ['r', 'g', 'b', 'k', 'y', 'c'][:len(labels)]
    for name, ax in zip(common_names, axarry):
        vals = [np.fromiter((evt[name] for evt in data), dtype=float) for data in dicts_data]
        match = np.all([np.allclose(vals1, vals2) for vals1, vals2 in zip(vals[1:], vals[:-1])])
        _, _, patches = ax.hist(vals, density=True, histtype='step', label=labels, alpha=0.5, lw=2, color=colours)
        if match:
            ax.set_ylabel("Density ~ match")
        else:
            ax.set_ylabel("Density ~ differ")
        ax.set_xlabel(name)
    ax = axarry[-1]
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    for c, l in zip(colours, labels):
        ax.plot([0], [0], label=l, color=c)
    ax.legend()
    fig.set_size_inches(10, 11)
    plt.tight_layout()
    plt.show()

        
