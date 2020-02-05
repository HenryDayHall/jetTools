""" you need to compile the shape program to use this with 
f77 shape2.f shape3.f shape4.f -o shape """
import numpy as np
import subprocess
import os
from ipdb import set_trace as st

def shape(energies, pxs, pys, pzs, my_dir='./'):
    # these need to be stacked into a momentum vector that 
    # has energy last
    momentums = np.vstack((pxs, pys, pzs, energies)).T
    try:
        s = np.sum(np.sum(momentums, axis=0)**2)
    except ValueError:
        st()
    call = [my_dir + "shape", str(s)] + np.hstack(momentums).astype(str).tolist()
    process = subprocess.Popen(call, stdout=subprocess.PIPE)
    results = {}
    for line in process.stdout.readlines():
        # some names have spaces in :((
        name, value = line.decode().strip().rsplit(' ', 1)
        name = name.strip()
        results[name] = float(value)
    return results

