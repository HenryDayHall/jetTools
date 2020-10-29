""" a script to update all datasets in the top level of megaIgnore to acount ro recent changes from FormTreesRefactor branch """
import os
from tree_tagger import Components, FormJets
# grab the datasets
dir_name = "megaIgnore"
file_names = [os.path.join(dir_name, name) for name in os.listdir(dir_name) if name.endswith('.awkd')]
#file_names = ["best.awkd"]
num_names = len(file_names)


for i, name in enumerate(file_names):
    try:
        eventWise = Components.EventWise.from_file(name)
    except Exception:
        print(f"{name} dosn't seem to be an EventWise")
        continue
    print(f"{i/num_names:.0%}\t{name}", flush=True)

    # Delele all knn items - known fault
    #affinitycutoffs = [(name, getattr(eventWise, name)) for name in eventWise.hyperparameter_columns
    #                   if name.endswith("AffinityCutOff")]
    #knn_jets = [name.split('_', 1)[0] for name, cutoff in affinitycutoffs
    #            if cutoff is not None and cutoff[0] == 'knn']
    #for jet in knn_jets:
    #    eventWise.remove_prefix(jet)


    # add parameters to datasets
    # change Luclus to angular and add Luclus
    new_hyper = {}
    #for name in FormJets.get_jet_names(eventWise):
    #    if name + "_ExpofPTFormat" not in eventWise.hyperparameter_columns:
    #        is_luclus = getattr(eventWise, name+"_PhyDistance") == 'Luclus'
    #        if is_luclus:
    #            new_hyper[name+"_PhyDistance"] = 'angular'
    #            new_hyper[name+"_ExpofPTFormat"] = 'Luclus'
    #        else:
    #            new_hyper[name+"_ExpofPTFormat"] = 'min'
    #    if name + "_Eigenspace" not in eventWise.hyperparameter_columns:
    #        new_hyper[name+"_Eigenspace"] = 'unnormalised'
    #    try:
    #        if getattr(eventWise, name + "_AffinityType") == 'angular':
    #            # broken :(
    #            new_hyper[name + "_AffinityType"] = 'unknown'
    #    except AttributeError:
    #        pass  # probably a traditional jet
    for name in FormJets.get_jet_names(eventWise):
        if name + "_EigDistance" not in eventWise.hyperparameter_columns:
            new_hyper[name + "_EigDistance"] = 'euclidien'
    if new_hyper:
        eventWise.append_hyperparameters(**new_hyper)

        


