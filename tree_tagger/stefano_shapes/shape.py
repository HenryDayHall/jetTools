""" you need to compile the shape program to use this with 
f77 shape2.f shape3.f shape4.f -o shape """
import numpy as np
import subprocess
import os

def shape(*momentums, my_dir='./'):
    s = np.sum(np.sum(momentums, axis=0)**2)
    call = [my_dir + "shape", str(s)] + np.hstack(momentums).astype(str).tolist()
    process = subprocess.Popen(call, stdout=subprocess.PIPE)
    results = {}
    for line in process.stdout.readlines():
        # some names have spaces in :((
        name, value = line.decode().strip().rsplit(' ', 1)
        name = name.strip()
        results[name] = float(value)
    return results

