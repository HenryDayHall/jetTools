""" Compare physical properties of jets and showers """
import numpy as np
import awkward
import scipy.spatial
from jet_tools.tree_tagger import FormShower, TrueTag, FormJets, Components, PlottingTools
import os
from matplotlib import pyplot as plt
from ipdb import set_trace as st

def width(linear, cyclic):
    """
    

    Parameters
    ----------
    linear :
        
    cyclic :
        

    Returns
    -------

    """
    if len(linear) < 2:
        return 0.
    cyclic_x = np.mean(np.cos(cyclic))
    cyclic_y = np.mean(np.sin(cyclic))
    mean_cyclic = np.arctan2(cyclic_y, cyclic_x)
    # then move all the cordinates so that the mean cyclic is 0
    cyclic_shift = cyclic - mean_cyclic
    cyclic_shift[cyclic_shift < -np.pi] += 2*np.pi
    cyclic_shift[cyclic_shift > np.pi] -= 2*np.pi
    cyclic_distances = scipy.spatial.distance.pdist(cyclic_shift.reshape((-1, 1)), metric='sqeuclidean')
    linear_distances = scipy.spatial.distance.pdist(linear.reshape((-1, 1)), metric='sqeuclidean')
    total_distances2 = cyclic_distances + linear_distances
    return np.sqrt(np.max(total_distances2))


def decendants_width(eventWise, *root_idxs, only_visible=True):
    """
    

    Parameters
    ----------
    eventWise :
        
    *root_idxs :
        

    Returns
    -------

    """
    assert eventWise.selected_index is not None
    decendants = list(FormShower.descendant_idxs(eventWise, *root_idxs))
    # select only the visible objects 
    if only_visible:
        decendants = [d for d in decendants if d in eventWise.JetInputs_SourceIdx]
    rapidity = eventWise.Rapidity
    phi = eventWise.Phi
    dwidth = width(rapidity[decendants], phi[decendants])
    return dwidth


def append_b_shower_widths(eventWise, silent=True):
    """
    

    Parameters
    ----------
    eventWise :
        
    silent :
         (Default value = True)

    Returns
    -------

    """
    if "BQuarkIdx" not in eventWise.columns:
        FormShower.append_b_idxs(eventWise, silent)
    eventWise.selected_index = None
    name = "BWidth"
    n_events = len(eventWise.BQuarkIdx)
    widths = list(getattr(eventWise, name, []))
    start_point = len(widths)
    if start_point >= n_events:
        print("Finished")
        return True
    end_point = n_events
    if not silent:
        print(f" Will stop at {100*end_point/n_events}%")
    for event_n in range(start_point, end_point):
        if event_n % 10 == 0 and not silent:
            print(f"{100*event_n/n_events}%", end='\r', flush=True)
        if os.path.exists("stop"):
            print(f"Completed event {event_n-1}")
            break
        eventWise.selected_index = event_n
        b_idxs = eventWise.BQuarkIdx
        widths_here = [decendants_width(eventWise, idx) for idx in b_idxs]
        widths.append(widths_here)
    widths = awkward.fromiter(widths)
    eventWise.append(**{name: widths})
        

def append_b_jet_widths(eventWise, jet_name, signal_only=False, silent=True):
    """
    

    Parameters
    ----------
    eventWise :
        
    jet_name :
        
    silent :
         (Default value = True)

    Returns
    -------

    """
    eventWise.selected_index = None
    if signal_only and "DetectableTag_Leaves" not in eventWise.columns:
        TrueTag.add_detectable_fourvector(eventWise)
    if signal_only:
        name = jet_name + "_BSigWidth"
    else:
        name = jet_name + "_BWidth"
    n_events = len(eventWise.DetectableTag_Leaves)
    widths = list(getattr(eventWise, name, []))
    start_point = len(widths)
    if start_point >= n_events:
        print("Finished")
        return True
    end_point = n_events
    if not silent:
        print(f" Will stop at {100*end_point/n_events}%")
    for event_n in range(start_point, end_point):
        if event_n % 10 == 0 and not silent:
            print(f"{100*event_n/n_events}%", end='\r', flush=True)
        if os.path.exists("stop"):
            print(f"Completed event {event_n-1}")
            break
        eventWise.selected_index = event_n
        # find the tagged particles
        tagged = [i for i, t in enumerate(getattr(eventWise, jet_name+"_Tags"))
                  if len(t)]
        if not tagged:
            widths.append([])
            continue
        # obtain the leaf particles
        is_leaf = getattr(eventWise, jet_name + "_Child1")[tagged] == -1
        if signal_only:
            source_idx = eventWise.JetInputs_SourceIdx
            detectable = eventWise.DetectableTag_Leaves.flatten()
            input_idxs = getattr(eventWise, jet_name+"_InputIdx")[tagged]
            signal = awkward.fromiter([[False if idx >= len(source_idx)  # not signal if not source
                                        else source_idx[idx] in detectable
                                        for idx in jet]
                                       for jet in input_idxs])
            is_leaf = is_leaf*signal
        rapidity = getattr(eventWise, jet_name + "_Rapidity")[tagged][is_leaf]
        phi = getattr(eventWise, jet_name + "_Phi")[tagged][is_leaf]
        widths_here = [width(r, p) for r, p in zip(rapidity, phi)]
        #if np.max(widths_here) > 5.:
        #    largest = np.argmax(widths_here)
        #    plt.scatter(rapidity[largest], phi[largest])
        #    st()
        #    width(rapidity[largest], phi[largest])
        widths.append(widths_here)
    widths = awkward.fromiter(widths)
    eventWise.append(**{name: widths})


def plot_jet_v_shower(eventWise, jet_name, silent=False, colour='PT', ax=None, signal_only=False):
    """
    

    Parameters
    ----------
    eventWise :
        
    jet_name :
        
    silent :
         (Default value = False)
    colour :
         (Default value = 'PT')

    Returns
    -------

    """
    eventWise.selected_index = None
    n_events = len(eventWise.MCPID)
    shower_name = "BWidth"
    if shower_name not in eventWise.columns:
        append_b_shower_widths(eventWise, silent=False)
    if signal_only:
        jet_wname = jet_name + "_BSigWidth"
    else:
        jet_wname = jet_name + "_BWidth"
    if jet_wname not in eventWise.columns:
        append_b_jet_widths(eventWise, jet_name, signal_only=signal_only, silent=False)
    jet_widths, shower_widths = [], []
    colour_var = []
    end_point = n_events
    if not silent:
        print(f" Will stop at {100*end_point/n_events}%")
    for event_n in range(end_point):
        if event_n % 10 == 0 and not silent:
            print(f"{100*event_n/n_events}%", end='\r', flush=True)
        eventWise.selected_index = event_n
        # add the jet widths directly
        jet_widths_here = getattr(eventWise, jet_wname)
        n_jets_here = len(jet_widths_here)
        if colour == 'Event Energy':
            colour_var += n_jets_here*[np.sum(eventWise.JetInputs_Energy)]
        jet_widths += jet_widths_here.tolist()
        # some sorting is needed to add the shower widths
        tags = getattr(eventWise, jet_name + "_Tags")
        # use only the shoer widths that corrispond to tagged jets
        for jet_n, ts in enumerate(tags):
            if len(ts) == 0:
                continue
            quarks_in_jet = TrueTag.tags_to_quarks(eventWise, ts)
            # this is not relyably the last quark
            if len(quarks_in_jet) == 0:
                raise RuntimeError
            if not set(quarks_in_jet).issubset(eventWise.BQuarkIdx):
                raise RuntimeError
            if colour == 'PT':
                input_root = getattr(eventWise, jet_name + "_RootInputIdx")[jet_n, 0]
                jet_root = getattr(eventWise, jet_name + "_InputIdx")[jet_n] == input_root
                colour_var.append(getattr(eventWise, jet_name + "_PT")[jet_n, jet_root][0])
            if colour == 'track multiplicity':
                child1 = getattr(eventWise, jet_name + "_Child1")[jet_n]
                colour_var.append(np.sum(child1 == -1))
            # if there is more than one tag calculate the width of the combined decendants
            if len(quarks_in_jet) > 1:
                shower_widths.append(decendants_width(eventWise, *quarks_in_jet))
            else:  # find the width in existing shower widths
                shower_mask = eventWise.BQuarkIdx == quarks_in_jet[0]
                shower_widths.append(eventWise.BWidth[shower_mask][0])
    if ax is None:
        fig, ax = PlottingTools.discribe_jet(eventWise, jet_name)
    scatter = ax.scatter(shower_widths, jet_widths, c=colour_var, alpha=0.1, linewidth=0, label=jet_name)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(colour)
    ax.set_xlabel("Shower width in deltaR")
    if signal_only:
        ax.set_ylabel("Found signal width in deltaR")
    else:
        ax.set_ylabel("Jet width in deltaR")
    #plt.title("Width of shower vs width of jet")
    #plt.show()
    return jet_widths, shower_widths, colour_var
        

