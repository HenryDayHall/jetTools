from sklearn import tree, ensemble
import sklearn
from ipdb import set_trace as st
from matplotlib import pyplot as plt
import numpy as np
import pickle

def make_finite(ary):
    return np.nan_to_num(ary.astype('float32'))

def begin_training(run, viewer=None):
    assert 'bdt' in run.settings['net_type'].lower();
    # create the dataset
    dataset = run.dataset
    if not run.empty_run:
        bdt = run.last_nets[0]
    else:
        dtc = tree.DecisionTreeClassifier(max_depth=run.settings['max_depth'])
        bdt = ensemble.AdaBoostClassifier(dtc, algorithm=run.settings['algorithm_name'],
                                 n_estimators=run.settings['n_estimators'])
    bdt.fit(make_finite(dataset.jets), dataset.truth)
    run.last_nets = [bdt]
    run.set_best_state_dicts([pickle.dumps(bdt)])
    run.write()


def make_hist(run):
    bdt = run.best_nets[0]
    dataset = run.dataset
    output = bdt.decision_function(make_finite(dataset.test_jets))
    test_truth = dataset.test_truth.flatten()
    plot_range = (output.min(), output.max())
    plt.hist(output[test_truth>0.5],
             bins=10, range=plot_range,
             facecolor='g', label="Signal",
             alpha=.5, edgecolor='k', normed=True)
    plt.hist(output[test_truth<0.5],
             bins=10, range=plot_range,
             facecolor='r', label="Background",
             alpha=.5, edgecolor='k', normed=True)
    plt.ylabel("Percent out")
    plt.xlabel("BDT output")
    plt.title("Jet tagging BDT")
    plt.show()
    return run


def plot_rocs(runs, loglog=False, ax=None):
    #axis
    if ax is None:
        _, ax = plt.subplots()
    else:
        ax.clear()
    
    #calculation
    if isinstance(runs, (list, np.ndarray)):
        for run in runs:
            bdt = run.best_nets[0]
            dataset = run.dataset
            outputs = bdt.decision_function(make_finite(dataset.test_jets))
            MC_truth = dataset.test_truth.flatten()
            fpr, tpr, _ = sklearn.metrics.roc_curve(MC_truth, outputs)
            plt.plot(fpr, tpr, label=run.settings['pretty_name'])
        ax.legend()
    else:
        # N.B. except Exception ignorse KeyboardInterrupt, SystemExit and GeneratorExit
        bdt = runs.best_nets[0]
        dataset = runs.dataset
        outputs = bdt.decision_function(make_finite(dataset.test_jets))
        MC_truth = dataset.test_truth.flatten()
        fpr, tpr, _ = sklearn.metrics.roc_curve(MC_truth, outputs)
        plt.plot(fpr, tpr, label=runs.settings['pretty_name'])

    # label
    if loglog:
        ax.loglog()
    ax.set_title("Receiver Operator curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")

