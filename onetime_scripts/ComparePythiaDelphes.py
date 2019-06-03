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



def makeCheckfile(databaseName):
    """
    Pakes a csv file of all the data in the database,
    with format as close as possible to the c++ output.
    

    Parameters
    ----------
    databaseName : string
        file name os the database to be read.

    """
    fields = ["MCPID", "Status", "IsPU", "Charge", "Mass", "E", "Px", "Py", "Pz", "P", "PT", "Eta", "Phi", "Rapidity", "CtgTheta", "D0", "DZ", "T", "X", "Y", "Z"]
    out = readSelected(databaseName, fields)
    cppOut = [', '.join((cppLike(x) for x in line)) for line in out]
    testName = databaseName.split('.')[0] + "_python.txt"
    with open(testName, 'w') as outFile:
        for line in cppOut:
            outFile.write(str(line) + "\n")
    print(f"Written to {testName}")

def cppLike(x):
    """
    Format a number to be closer to the format of a c++ print
    

    Parameters
    ----------
    x : float or int
        number to be printed
        

    Returns
    -------
    : str
        the c++ like string format

    """
    if x == 0:
        x = abs(x)
    if x == int(x):
        return str(int(x))
    else:
        return str(x)


def checkReflection(databaseName=None, **kwargs):
    """
    Check for agreement between parent and child fields

    Parameters
    ----------
    databaseName : str
        path and file name of the database to be read
        

    """
    if databaseName is not None:
        fields = ["ID", "M1", "M2", "D1", "D2"]
        fromDatabase = readSelected(databaseName, fields)
        IDs = np.array([d[0] for d in fromDatabase])
        parents = np.array([d[1:3] for d in fromDatabase])
        children = np.array([d[3:5] for d in fromDatabase])
    else:
        IDs = kwargs.get('IDs')
        parents = kwargs.get('parents')
        children = kwargs.get('children')
    listIDs = list(IDs)
    for row, this_id in enumerate(IDs):
        for p in parents[row]:
            if p is not None:
                assert p in IDs, f"The parent {p} of {this_id} is invalid"
                p_row = listIDs.index(p)  # find the row of the parent
                assert this_id in children[p_row], f"Parent {p} of {this_id} not acknowledging child"
        for c in children[row]:
            if c is not None:
                assert c in IDs, f"The child {c} of {this_id} is invalid"
                c_row = listIDs.index(c)  # find the row of the child
                if row == c_row:
                    print(f"Particle {this_id} appears to be it's own daughter")
                    print("Ignoring")
                    continue
                assert this_id in parents[c_row], f"Child {c} of {this_id} not acknowledging parent"

def checkPIDMatch(databaseName, table1, refField, table2, PIDfield="MCPID"):
    # in table 1 we need the ref field and the pid field
    out1 = readSelected(databaseName, [refField, PIDfield], table1)
    # in table2 we need the ID field and the pid field
    out2 = readSelected(databaseName, ["ID", PIDfield], table2)
    # convert the second table to a dict
    out2 = dict(out2)
    for foreignKey, PID in out1:
        assert PID == out2[foreignKey], f"Error, track PID mismatch for particle {foreignKey}"

