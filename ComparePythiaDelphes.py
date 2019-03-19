import numpy as np
from matplotlib import pyplot as plt
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
def checkStatePID(rootValues, hepmc):
    hepmc_state = hepmc.intParticles[:, hepmc.intParticle_columns.index("status_code")] 
    state_match = hepmc_state == np.array(rootValues["status"])
    print(f"{100*sum(state_match)/len(state_match)} % of particle states match")
    hepmc_pid = hepmc.intParticles[:, hepmc.intParticle_columns.index("MCPID")] 
    pid_match = hepmc_pid == np.array(rootValues["MCPID"])
    print(f"{100*sum(pid_match)/len(pid_match)} % of particle pids match")
# yes they do
    

# ave percent missmatch
def showMissmatch(rootValues, hepmc):
    soft_list = ["energy", "generated_mass", "px", "py", "pz"]
    fig, axes = plt.subplots(len(soft_list), 1)
    for ax, value_name in zip( axes, soft_list):
        hepmc_vals = hepmc.floatParticles[:, hepmc.floatParticle_columns.index(value_name)]
        ax.scatter(hepmc_vals, rootValues[value_name])
        ax.set_xlabel(value_name + " hepmc")
        ax.set_ylabel(value_name + " root")
        x = np.linspace(*ax.get_xlim())
        ax.plot(x, x)
        percentage_diffs = 2*(hepmc_vals - rootValues[value_name])/(hepmc_vals + rootValues[value_name])
        percent_diff = np.nanmean(percentage_diffs)
        ax.set_title(f"average percent diff {percent_diff}")
    plt.show()
# this si v v small


# this dosn't work, too many particles close together.
def seekMatch(rootValues, hepmc):
    mask = np.full_like(hepmc.ids, True, dtype=bool)
    # start by looking for the same pid
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

def getRootValues(id_num=None, databaseName="/home/henry/lazy/29delphes_events.db"):
    fields = ["MCPID", "status", "E", "Mass", "Px", "Py", "Pz"]
    dict_keys = ["MCPID", "status", "energy", "generated_mass", "px", "py", "pz"]
    if id_num is None:
        out = readSelected(databaseName, fields)
        index = slice(None, None)
    else:
        out = readSelected(databaseName, fields, where=f"ID=={id_num}")
        index = 0
    root_dict = {key: out[index, i] for i, key in enumerate(dict_keys)}
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


