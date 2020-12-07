""" Do a single run of forming the jets, with specified parameters"""
from ipdb import set_trace as st
from jet_tools.tree_tagger import InputTools, Components, JoinHepMCRoot, ReadHepmc, FormJets, TrueTag, ShapeVariables, MassPeaks
import os
import shutil
import ast
import numpy as np
from matplotlib import pyplot as plt

BATCH_LENGTH = 10000

def get_data_file():
    """ """
    # to start with we know nothing
    root_file = hepmc_file = eventWise_file = False
    start_file = InputTools.get_file_name("Name the data file to start from; ").strip()
    if start_file.endswith('.root'):
        root_file = start_file
        hepmc_file = InputTools.get_file_name("Please give the corripsonding .hepmc file; ", '.hepmc')
    elif start_file.endswith('.hepmc'):
        hepmc_file = start_file
        root_file = InputTools.get_file_name("If desired enter a .root file (blank to skip); ", '.root').strip()
    if hepmc_file and root_file:
        eventWise = JoinHepMCRoot.marry(hepmc_file, root_file)
    elif hepmc_file:
        dir_name, save_name = os.path.split(hepmc_file)
        eventWise = ReadHepmc.Hepmc(dir_name, save_name)
    elif start_file.endswith('.awkd'):
        eventWise = Components.EventWise.from_file(start_file)
    else:
        raise ValueError(f"'{start_file}' not of recognised type,"
                          " please give a '.hepmc' (and optionally a '.root') or an '.awkd' file")
    return eventWise


def define_inputs(eventWise):
    """
    

    Parameters
    ----------
    eventWise :
        

    Returns
    -------

    """
    if 'JetInputs_Energy' in eventWise.columns:
        use_existing = InputTools.yesNo_question("There are already JetInputs, use these? ")
        if use_existing:
            return eventWise
        else:
            # make a copy of the eventWise with no clusters or JetInputs
            path = os.path.join(eventWise.dir_name, eventWise.save_name)
            print("Will make a copy of the eventWise without existing clusters to change JetInputs")
            print(f"Current file name is {eventWise.save_name}.")
            new_name = InputTools.list_complete("Name the copy; ", [''])
            if not new_name.endswith('.awkd'):
                new_name += '.awkd'
            new_path = os.path.join(eventWise.dir_name, new_name)
            shutil.copy(path, new_path)
            del eventWise
            eventWise = Components.EventWise.from_file(new_path)
            # remove any clusters
            clusters = {name.split('_', 1)[0] for name in eventWise.columns
                        if name.endswith('Parent')}
            for name in clusters:
                eventWise.remove_prefix(name)
            # remove the jet inputs
            eventWise.remove_prefix('JetInputs')
    # if we get here there are no current JetInputs
    if InputTools.yesNo_question("Filter the tracks on pT or eta? "):
        pt_cut = InputTools.get_literal("What is the minimum pT of the tracks? ", float)
        eta_cut = InputTools.get_literal("What is the absolute maximum of the tracks? ", float)
        pt_eta_cut = lambda *args: FormJets.filter_pt_eta(*args, min_pt=pt_cut, max_eta=eta_cut)
        filter_functions = [FormJets.filter_ends, pt_eta_cut]
    else:
        filter_functions = [FormJets.filter_ends]
    FormJets.create_jetInputs(eventWise, filter_functions=filter_functions, batch_length=BATCH_LENGTH)
    return eventWise


def get_existing_clusters(eventWise):
    """
    

    Parameters
    ----------
    eventWise :
        

    Returns
    -------

    """
    clusters = {name.split('_', 1)[0] for name in eventWise.columns
                if name.endswith('Parent')}
    if not clusters:
       return False
    choice = InputTools.list_complete("Do you want to use an existing cluster (blank for make new)? ", list(clusters)).strip()
    if choice == '':
        return False
    return choice


def pick_class_params():
    """ """
    cluster_options = FormJets.multiapply_input
    cluster_name = InputTools.list_complete("Which form of clustering? ", cluster_options.keys()).strip()
    cluster_function = cluster_options[cluster_name]
    if cluster_name not in FormJets.cluster_classes:
        if cluster_name in ["Fast", "Home"]:
            cluster_class = getattr(FormJets, "Traditional")
        else:
            raise NotImplementedError
    else:
        cluster_class = getattr(FormJets, cluster_name)
    default_parameters = cluster_class.default_params
    chosen_parameters = {}
    print(f"Select the parameters for {cluster_name}, blank for default.")
    for name, default in default_parameters.items():
        selection = InputTools.list_complete(f"{name} (default {default_parameters[name]}); ", [''])
        if selection == '':
            chosen_parameters[name] = default
            continue
        try:
            selection = ast.literal_eval(selection)
        except ValueError:
            pass
        if not InputTools.yesNo_question(f"Understood {selection}, is this correct? "):
            print("fix it manually")
            st()
            pass
        chosen_parameters[name] = selection
    return cluster_name, cluster_function, chosen_parameters


def make_new_cluster(eventWise):
    """
    

    Parameters
    ----------
    eventWise :
        

    Returns
    -------

    """
    cluster_name, cluster_function, chosen_parameters = pick_class_params()
    # now we have parameters, apply them
    jet_name = InputTools.list_complete("Name this cluster (empty for autoname); ", [''])
    if jet_name  == '':
        found = [name.split('_', 1)[0] for name in eventWise.hyperparameters
                 if name.startswith(cluster_name)]
        i = 0
        jet_name = cluster_name + "Jet" + str(i)
        while jet_name in found:
            i += 1
            jet_name = cluster_name + "Jet" + str(i)
        print(f"Naming this {jet_name}")
    FormJets.cluster_multiapply(eventWise, cluster_function, chosen_parameters, batch_length=BATCH_LENGTH, jet_name=jet_name)
    return jet_name


def get_make_tags(eventWise, jet_name):
    """
    

    Parameters
    ----------
    eventWise :
        
    jet_name :
        

    Returns
    -------

    """
    tag_name = jet_name + '_'
    if InputTools.yesNo_question("Apply pt cut to jets before tagging? "):
        jet_pt_cut = InputTools.get_literal("Jet pt cut; ", float)
        jet_name += str(int(jet_pt_cut))
    else:
        jet_pt_cut = None
    tag_name += "Tags"
    if hasattr(eventWise, tag_name):
        if InputTools.yesNo_question(f"{tag_name} already exists, use this? "):
            # just return here
            return jet_pt_cut
    dr = InputTools.get_literal("What is the maximum angle between the tag and the jet (dr)? ", float)
    min_tracks = InputTools.get_literal("Min tracks to tag jet; ", int)
    TrueTag.add_tags(eventWise, jet_name, dr, batch_length=BATCH_LENGTH, jet_pt_cut=jet_pt_cut, min_tracks=min_tracks, overwrite=True)
    return jet_pt_cut


def plot_results(eventWise, jet_name, pretag_jet_pt_cut, img_base):
    """
    

    Parameters
    ----------
    eventWise :
        
    jet_name :
        
    pretag_jet_pt_cut :
        
    img_base :
        

    Returns
    -------

    """
    tag_before_pt_cut = pretag_jet_pt_cut is None
    jet_pt_cut = pretag_jet_pt_cut
    if not tag_before_pt_cut:
        if InputTools.yesNo_question("Apply pt cut to jets after tagging? "):
            jet_pt_cut = InputTools.get_literal("Jet pt cut; ", float)
    if InputTools.yesNo_question("Plot shape variables? "):
        ShapeVariables.append_tagshapes(eventWise, batch_length=BATCH_LENGTH, jet_pt_cut=jet_pt_cut, tag_before_pt_cut=tag_before_pt_cut)
        ShapeVariables.append_jetshapes(eventWise, jet_name, batch_length=BATCH_LENGTH, jet_pt_cut=jet_pt_cut, tag_before_pt_cut=tag_before_pt_cut)
        if tag_before_pt_cut:
            ShapeVariables.plot_shapevars(eventWise, jet_name)
        else:
            ShapeVariables.plot_shapevars(eventWise, jet_name, jet_pt_cut)
        if img_base:
            plt.tight_layout()
            fig = plt.gcf()
            fig.set_size_inches(10, 10)
            plt.savefig(img_base + '_shape.png')
        else:
            plt.show()
    # mass peaks
    if not hasattr(eventWise, jet_name + "_Tags"):
        st()
    MassPeaks.plot_PT_pairs(eventWise, jet_name, jet_pt_cut=jet_pt_cut, show=not img_base)
    if img_base:
        plt.tight_layout()
        fig = plt.gcf()
        fig.set_size_inches(10, 10)
        plt.savefig(img_base + '_PTpairs.png')
    else:
        plt.show()
    #MassPeaks.plot_smallest_angles(eventWise, jet_name, jet_pt_cut=jet_pt_cut, show=not img_base)
    #if img_base:
    #    plt.tight_layout()
    #    plt.savefig(img_base + '_smallAngle.png')


def check_problem():
    """ """
    eventWise = get_data_file()
    BATCH_LENGTH = InputTools.get_literal("How long should the batch be (-1 for all events)? ", int)
    if BATCH_LENGTH == -1:
        BATCH_LENGTH = np.inf
    eventWise = define_inputs(eventWise)
    params = {'DeltaR': 0.4, 'ExponentMultiplier': 0.1, 'NumEigenvectors': 2, 'Laplacien': 'symmetric', "AffinityType": 'linear', "WithLaplacienScaling": False, "AffinityCutoff": ('distance', 5.2), "Invarient": 'normed'}
    jet_name = "ProblemJet"
    params["jet_name"] = jet_name
    FormJets.cluster_multiapply(eventWise, FormJets.SpectralFull, params, batch_length=BATCH_LENGTH)


if __name__ == '__main__':
    eventWise = get_data_file()
    BATCH_LENGTH = InputTools.get_literal("How long should the batch be (-1 for all events)? ", int)
    if BATCH_LENGTH == -1:
        BATCH_LENGTH = np.inf
    eventWise = define_inputs(eventWise)
    jet_name = get_existing_clusters(eventWise)
    if not jet_name:
        jet_name = make_new_cluster(eventWise)
    # now do tagging
    pretag_jet_pt_cut = get_make_tags(eventWise, jet_name)
    # now calculate shapes and mass peaks
    img_base = InputTools.get_file_name("Give a name to save the plots (empty for no save); ")
    plot_results(eventWise, jet_name, pretag_jet_pt_cut, img_base)



