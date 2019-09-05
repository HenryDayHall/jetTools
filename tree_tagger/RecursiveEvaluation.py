""" evaluate the decisions of the linking NN """
from ipdb import set_trace as st
import numpy as np
from matplotlib import pyplot as plt
from tree_tagger import RecursiveNN, Constants
import sklearn
import torch

def apply_recursive_net(run, use_test=True, nets=None):
    if nets is None:
        net = run.best_nets[0]
    else:
        net = nets[0]
    # Device configuration
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    net.to(device)
    dataset = run.dataset
    if use_test:
        events = dataset.test_events
    else:
        events = dataset
    MC_truth = np.empty(len(events), dtype=int)
    outputs = np.empty(len(events), dtype=float)
    for i, event_data in enumerate(events):
        truth, walker = event_data
        outputs[i] = net(walker).detach().item()
        MC_truth[i] = truth
    # do the sigmoid
    outputs = 1/(1+np.exp(-outputs))
    return outputs, MC_truth


def plot_rocs(runs, loglog=False, ax=None):
    #axis
    if ax is None:
        _, ax = plt.subplots()
    else:
        ax.clear()
    
    #calculation
    try:  # try treating it as a list
        for run in runs:
            MC_truth, outputs = apply_recursive_net(run)
            fpr, tpr, _ = sklearn.metrics.roc_curve(MC_truth, outputs)
            plt.plot(fpr, tpr, label=run.settings['pretty_name'])
        ax.legend()
    except Exception: # try as a single run
        # N.B. except Exception ignorse KeyboardInterrupt, SystemExit and GeneratorExit
        MC_truth, outputs = apply_recursive_net(runs)
        fpr, tpr, _ = sklearn.metrics.roc_curve(MC_truth, outputs)
        plt.plot(fpr, tpr, label=runs.settings['pretty_name'])

    # label
    if loglog:
        ax.loglog()
    ax.set_title("Receiver Operator curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")



def plot_hist(outputs, MC_truth, ax=None, log_y=True):
    if ax is None:
        _, ax = plt.subplots()
    else:
        ax.clear()
    MC_truth = MC_truth.astype(bool) 
    num_signal = np.sum(MC_truth)
    num_bg = np.sum(~MC_truth)

    # plot
    n_bins = 50
    hist_type = 'step'
    distance_range = [0, 1]
    ax.hist(outputs[MC_truth], n_bins, distance_range, histtype=hist_type, color='green',
            label=f"{num_signal} signal", density=True) 
    ax.hist(outputs[~MC_truth], n_bins, distance_range, histtype=hist_type, color='red',
            label=f"{num_bg} background", density=True) 

    # label
    ax.legend()
    if log_y: ax.semilogy()
    ax.set_title("Output of classifier")
    ax.set_xlabel("Signal like charicter")
    ax.set_ylabel(f"Normed frequency per catigory")


class ResponsePlot:
    def __init__(self, run):
        self.run = run
        data = []
        self.on_launch()

    def on_launch(self):
        # set up plots
        self.figure, self.ax = plt.subplots()

    def update(self, _, nets):
        outputs, MC_truth = apply_recursive_net(self.run, nets=nets)
        # update the plot
        plot_hist(outputs, MC_truth, self.ax)
        #Need both of these in order to rescale
        self.ax.relim()
        self.ax.autoscale_view()
        #We need to draw *and* flush
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

