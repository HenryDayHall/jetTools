""" compare two jet clustering techniques """
from ipdb import set_trace as st
from tree_tagger import Components
import sklearn.metrics
from matplotlib import pyplot as plt
import numpy as np

def rand_score(eventWise, jet_name1, jet_name2):
    # two jets clustered from the same eventWise should have
    # the same JetInput_SourceIdx, 
    # if I change this in the future I need to update this function
    selection1 = getattr(eventWise, jet_name1+"_InputIdx")
    selection2 = getattr(eventWise, jet_name2+"_InputIdx")
    num_common_events = min(len(selection1), len(selection2))
    num_inputs_per_event = Components.apply_array_func(len, eventWise.JetInputs_SourceIdx[:num_common_events])
    scores = []
    for event_n, n_inputs in enumerate(num_inputs_per_event):
        labels1 = -np.ones(n_inputs)
        for i, jet in enumerate(selection1[event_n]):
            labels1[jet[jet < n_inputs]] = i
        assert -1 not in labels1
        labels2 = -np.ones(n_inputs)
        for i, jet in enumerate(selection2[event_n]):
            labels2[jet[jet < n_inputs]] = i
        assert -1 not in labels2
        score = sklearn.metrics.adjusted_rand_score(labels1, labels2)
        scores.append(score)
    return scores

def visulise_scores(scores, jet_name1, jet_name2, score_name="Rand score"):
    plt.hist(scores, bins=40, density=True, histtype='stepfilled')
    mean = np.mean(scores)
    std = np.std(scores)
    plt.vlines([mean - 2*std, mean, mean + 2*std], 0, 1, colors=['orange', 'red', 'orange'])
    plt.xlabel(score_name)
    plt.ylabel("Density")
    plt.title(f"Similarity of {jet_name1} and {jet_name2} (for {len(scores)} events)")



def pseudovariable_differences(eventWise, jet_name1, jet_name2, var_name="Rapidity"):
    eventWise.selected_index = None
    selection1 = getattr(eventWise, jet_name1+"_InputIdx")
    selection2 = getattr(eventWise, jet_name2+"_InputIdx")
    var1 = getattr(eventWise, jet_name1 + "_" + var_name)
    var2 = getattr(eventWise, jet_name2 + "_" + var_name)
    num_common_events = min(len(selection1), len(selection2))
    num_inputs_per_event = Components.apply_array_func(len, eventWise.JetInputs_SourceIdx[:num_common_events])
    pseudojet_vars1 = []
    pseudojet_vars2 = []
    num_unconnected = 0
    for event_n, n_inputs in enumerate(num_inputs_per_event):
        values1 = {}
        for i, jet in enumerate(selection1[event_n]):
            children = tuple(sorted(jet[jet < n_inputs]))
            values = sorted(var1[event_n, i, jet >= n_inputs])
            values1[children] = values
        values2 = {}
        for i, jet in enumerate(selection2[event_n]):
            children = tuple(sorted(jet[jet < n_inputs]))
            values = sorted(var2[event_n, i, jet >= n_inputs])
            values2[children] = values
        for key1 in values1.keys():
            if key1 in values2:
                pseudojet_vars1+=values2[key1]
                pseudojet_vars2+=values2[key1]
            else:
                num_unconnected += 1
    pseudojet_vars1 = np.array(pseudojet_vars1)
    pseudojet_vars2 = np.array(pseudojet_vars2)
    return pseudojet_vars1, pseudojet_vars2, num_unconnected

