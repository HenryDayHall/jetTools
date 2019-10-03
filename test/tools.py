""" tools used in various tests """
import numpy as np
import collections
from ipdb import set_trace as st
import os


def generic_equality_comp(x, y, strict=True):
    """ an atempt to generalise checking equality """
    xdict = x.__dict__
    ydict = y.__dict__
    if len(xdict) != len(ydict): return False
    if set(xdict.keys()) != set(ydict.keys()): return False
    for key in xdict:
        if isinstance(xdict[key], np.ndarray):
            # if possible use the numpy inbuilt comparison
            np.allclose(xdict[key], ydict[key])
        elif isinstance(xdict[key], collections.Hashable):
            if xdict[key] != ydict[key]: return False
        else:
            try:
                # if it's a list of numbers go back to numpy
                if isinstance(xdict[key][0], (int, float, np.int, np.float)):
                    if not np.allclose(xdict[key], ydict[key]): return False
            except:
                # finally give up and compare reprs
                if repr(xdict[key]) != repr(ydict[key]): return False
    return True


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

