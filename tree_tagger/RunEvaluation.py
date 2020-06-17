""" evaluate the decisions of the various NN """
#from ipdb import set_trace as st
import numpy as np
from matplotlib import pyplot as plt
from tree_tagger import RecursiveNN, Constants
import sklearn
import torch


def plot_rocs(runs, loglog=False, ax=None):
    """
    

    Parameters
    ----------
    runs :
        param loglog: (Default value = False)
    ax :
        Default value = None)
    loglog :
        (Default value = False)

    Returns
    -------

    
    """
    #axis
    if ax is None:
        _, ax = plt.subplots()
    else:
        ax.clear()
    
    #calculation
    try:  # try treating it as a list
        for run in runs:
            MC_truth, outputs = run.apply_to_test()
            fpr, tpr, _ = sklearn.metrics.roc_curve(MC_truth, outputs)
            plt.plot(fpr, tpr, label=run.settings['pretty_name'])
        ax.legend()
    except Exception: # try as a single run
        # N.B. except Exception ignorse KeyboardInterrupt, SystemExit and GeneratorExit
        MC_truth, outputs = run.apply_to_test()
        fpr, tpr, _ = sklearn.metrics.roc_curve(MC_truth, outputs)
        plt.plot(fpr, tpr, label=runs.settings['pretty_name'])

    # label
    if loglog:
        ax.loglog()
    ax.set_title("Receiver Operator curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")



def plot_hist(run, ax=None, log_y=True):
    """
    

    Parameters
    ----------
    run :
        param ax: (Default value = None)
    log_y :
        Default value = True)
    ax :
        (Default value = None)

    Returns
    -------

    
    """
    outputs, truth = run.apply_to_test()
    if ax is None:
        _, ax = plt.subplots()
    else:
        ax.clear()
    truth = truth.astype(bool).flatten()
    num_signal = np.sum(truth)
    num_bg = np.sum(~truth)
    outputs = outputs.flatten()
    st()

    # plot
    n_bins = 50
    hist_type = 'step'
    distance_range = [min(*outputs, 0),max(*outputs, 1)]
    ax.hist(outputs[truth], n_bins, distance_range, histtype=hist_type, color='green',
            label=f"{num_signal} signal", density=True) 
    ax.hist(outputs[~truth], n_bins, distance_range, histtype=hist_type, color='red',
            label=f"{num_bg} background", density=True) 

    # label
    ax.legend()
    if log_y: ax.semilogy()
    ax.set_title("Output of classifier")
    ax.set_xlabel("Signal like charicter")
    ax.set_ylabel(f"Normed frequency per catigory")


class ResponsePlot:
    """ """
    def __init__(self, run):
        self.run = run
        data = []
        self.on_launch()

    def on_launch(self):
        """ """
        # set up plots
        self.figure, self.ax = plt.subplots()
        self.test_input = self.run.get_test_input()

    def update(self, _, nets):
        """
        

        Parameters
        ----------
        _ :
            param nets:
        nets :
            

        Returns
        -------

        
        """
        outputs, MC_truth = self.run.apply_to_test(nets, self.test_input)
        # update the plot
        plot_hist(outputs, MC_truth, self.ax)
        #Need both of these in order to rescale
        self.ax.relim()
        self.ax.autoscale_view()
        #We need to draw *and* flush
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

