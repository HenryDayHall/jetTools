import numpy as np

class IDConverter:
    def __init__(self, table_file="pdg_names.csv"):
        table = np.genfromtxt(table_file, skip_header=1, dtype=np.str)
        self.ID_dict = {int(line[0]): line[1] for line in table}
        # include antiparticles
        self.ID_dict.update({-key: "~" + name for key, name in self.ID_dict.items()})

    def __getitem__(self, key):
        # if we don't find it in the dict return the key
        return self.ID_dict.get(key, key)
