import numpy as np
from ipdb import set_trace as st
from ReadSQL import readSelected
from ReadHepmc import Hepmc_event

def indexWithoutMask(index, mask):
    counting = -1
    for idx, m in enumerate(mask):
        counting += m
        if counting == index:
            return idx

# Can I check that up to the inital masking the particles maintain their order?
# Which catigory produces te missmatch?

def seekMatch(rootValues, hepmc):
    mask = np.full_like(hepmc.IDs, True, dtype=bool)
    # start by looking for the same PID
    mask *= hepmc.intParticles[:, hepmc.intParticle_columns.index("MCPID")] == rootValues["MCPID"]
    # then look to the status 
    mask *= hepmc.intParticles[:, hepmc.intParticle_columns.index("status_code")] == rootValues["status"]
    # now we are onto soft values
    soft_list = ["energy", "generated_mass", "px", "py", "pz"]
    min_indices = []
    scaled_distances = np.zeros(sum(mask))
    for value_name in soft_list:
        hepmc_vals = hepmc.floatParticles[:, hepmc.floatParticle_columns.index(value_name)][mask]
        distances = np.abs(hepmc_vals - rootValues[value_name])
        scaled_distances += distances/rootValues[value_name]
        min_indices.append(np.argmin(distances))
    if len(set(min_indices)) == 1:
        return True, indexWithoutMask(min_indices[0], mask)
    else:  # not an exact match
        return False, indexWithoutMask(np.argmin(scaled_distances), mask)

def getRootValues(id_num, databaseName="/home/henry/lazy/29delphes_events.db"):
    fields = ["MCPID", "status", "E", "Mass", "Px", "Py", "Pz"]
    out = readSelected(databaseName, fields, where=f"ID=={id_num}")
    dict_keys = ["MCPID", "status", "energy", "generated_mass", "px", "py", "pz"]
    root_dict = {key: out[0][i] for i, key in enumerate(dict_keys)}
    return root_dict
    

def seekAll():
    hepmc_name = "/home/henry/lazy/29pythia8_events.hepmc"
    hepmc = Hepmc_event()
    hepmc.read_file(hepmc_name)
    hepmc.assign_heritage()

    all_matches = {}
    perfect_match = {}
    for i in range(len(hepmc.intParticles)):
        root_dict = getRootValues(i)
        perf, idx = seekMatch(root_dict, hepmc)
        all_matches[i] = idx
        perfect_match[i] = perf
        if i % 100:
            print('.', flush=True, end='')
    return perfect_match, all_matches

