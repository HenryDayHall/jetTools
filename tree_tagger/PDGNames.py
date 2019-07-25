import numpy as np

class IDConverter:
    def __init__(self, table_file="tree_tagger/pdg_names.csv"):
        table = np.genfromtxt(table_file, skip_header=1, dtype=np.str)
        self.ID_dict = {int(line[0]): line[1] for line in table}
        # include antiparticles
        self.ID_dict.update({-key: "~" + name for key, name in self.ID_dict.items()})

    def __getitem__(self, key):
        # if we don't find it in the dict return the key
        return self.ID_dict.get(key, key)

def match(pid_list, desired, partial=True):
    desired = str(desired)
    converter = IDConverter()
    name_list = [str(converter[pid]) for pid in pid_list]
    if partial:
        return [desired in name for name in name_list]
    else:
        return [desired == name for name in name_list]
