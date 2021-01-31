""" you need to compile the shape program to use this with 
f77 shape2.f shape3.f shape4.f -o shape """
import numpy as np
import subprocess
import os
#from ipdb import set_trace as st

def shape(energies, pxs, pys, pzs, my_dir='./'):
    """
    

    Parameters
    ----------
    energies :
        param pxs:
    pys :
        param pzs:
    my_dir :
        Default value = './')
    pxs :
        
    pzs :
        

    Returns
    -------

    
    """
    # these need to be stacked into a momentum vector that 
    # has energy last
    momentums = np.vstack((pxs, pys, pzs, energies)).T
    # check for tachyons, probably edit this out at some point
    s = energies**2 - np.sum(momentums[:, :3]**2, axis=1)
    tollerance = -0.001
    if np.any(s <= tollerance):
        raise ValueError(f"Tachyons in {momentums}, s={s}")
    s = np.sum(s)
    #s = np.sum(energies**2) - np.sum(pxs**2 + pys**2 + pzs**2)
    call = [my_dir + "shape", str(s)] + np.hstack(momentums).astype(str).tolist()
    process = subprocess.Popen(call, stdout=subprocess.PIPE)
    results = {}
    for line in process.stdout.readlines():
        # some names have spaces in :((
        name, value = line.decode().strip().rsplit(' ', 1)
        name = name.strip()
        results[name] = float(value)
    return call, results

