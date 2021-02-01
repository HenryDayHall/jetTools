from jet_tools import Components, FormJets, InputTools
import os
import numpy as np

path = InputTools.get_file_name("Name the eventWise or directy of eventWise to fix; ")

if path.endswith(".awkd"):
    path_list = [path]
else:
    path_list = [os.path.join(path, name) for name in os.listdir(path)
                 if name.endswith(".awkd")]


def fix_jet(ew, jet_name):
    params = FormJets.get_jet_params(ew, jet_name)
    new_hypers = {}
    if "Spectral" in jet_name:
        if "EigenvalueLimit" not in params:
            new_hypers[f"{name}_EigenvalueLimit"] = np.inf
    if "AffinityType" in params:
        if params["AffinityType"] == "exponent2":
            new_hypers[f"{name}_AffinityType"] = "exponent"
            new_hypers[f"{name}_AffinityExp"] = 2.
        elif params["AffinityType"] == "exponent":
            if "AffinityExp" not in params:
                new_hypers[f"{name}_AffinityExp"] = 1.
    return new_hypers


for path in path_list:
    print(path)
    try:
        ew = Components.EventWise.from_file(path)
    except Exception as e:
        print(e)
        continue
    names = FormJets.get_jet_names(ew)
    hypers = {}
    for name in names:
        try:
            new_hypers = fix_jet(ew, name)
        except Exception as e:
            print(e)
            continue
        hypers.update(new_hypers)
    if hypers:
        try:
            ew.append_hyperparameters(**hypers)
        except Exception as e:
            print(e)
            continue





