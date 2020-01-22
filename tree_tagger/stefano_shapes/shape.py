""" you need to compile the shape program to use this with 
f77 shape2.f shape3.f shape4.f -o shape """
import numpy as np
import subprocess

def shape(*momentums):
    s = np.sum(np.sum(momentums, axis=0)**2)
    call = ["./shape", str(s)] + np.hstack(momentums).astype(str).tolist()
    process = subprocess.Popen(call, stdout=subprocess.PIPE)
    results = {}
    for line in process.stdout.readlines():
        name, value = line.decode().split()
        results[name] = float(value)
    return results

