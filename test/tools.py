""" tools used in various tests """
import numpy as np
#import collections
from ipdb import set_trace as st
import os
import awkward
import pickle

data_dir = "./base_data"
if not os.path.exists(data_dir):
    data_dir = "../base_data"


def generic_equality_comp(x, y):
    """ an atempt to generalise checking equality """
    strx = pickle.dumps(x)
    stry = pickle.dumps(y)
    return strx == stry


# context manager for test directory
class TempTestDir:
    def __init__(self, base_name):
        self.base_name = base_name
        self.num = 1

    def __enter__(self):
        dir_name = f"{self.base_name}{self.num}"
        made_dir = False
        while not made_dir:
            try:
                os.makedirs(dir_name)
                made_dir = True
            except FileExistsError:
                self.num += 1
                dir_name = f"{self.base_name}{self.num}"
        return dir_name

    def __exit__(self, *args):
        dir_name = f"{self.base_name}{self.num}"
        for root, dirs, files in os.walk(dir_name, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(dir_name)

