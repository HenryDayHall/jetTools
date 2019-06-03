# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


data = np.genfromtxt("./per_trackData.csv", skip_header=1)
weights = data[:, 0]
pts = data[:, 1]
etas = data[:, 2]

min_eta = np.min(etas)
max_eta = np.max(etas)
min_pt = np.min(pts)
max_pt = np.max(pts)

twoD=True
if twoD:
    pt_catigories = ['pt<1', '1<pt<2', '2<pt<5', '5<pt']
    masks = [pts<1, np.logical_and(pts>1, pts<2), np.logical_and(pts>2, pts<5), pts>5]
    eta_grid = np.linspace(min_eta, max_eta, 200)
    for name, mask in zip(pt_catigories, masks):
        kde = stats.gaussian_kde(etas[mask], weights=weights[mask])
        plt.plot(eta_grid, kde.evaluate(eta_grid), label=name)
    plt.legend()
    import matplotlib
    matplotlib.rc('text', usetex=True)
    plt.title("Psudorapidity ($\eta$) of tracks from 9999 events, weighted by inverse of event size")
    plt.xlabel("$\eta$")
    plt.ylabel("Frequency")
    plt.grid()
    plt.show()
else:
    pts = np.log(pts)
    min_pt = np.min(pts)
    max_pt = np.max(pts)
    X, Y = np.mgrid[min_eta:max_eta:70j, min_pt:max_pt:70j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack((etas, pts))
    kernal = stats.gaussian_kde(values)
    Z = np.reshape(kernal(positions).T, X.shape)
    plt.imshow(np.rot90(Z), cmap='cubehelix', extent=[min_eta, max_eta, min_pt, max_pt])
    plt.xlabel("Psuorapidity $\eta$")
    plt.ylabel("$log(p_T)$")
    plt.title("Density plot of tracks from 9999 events, weighted by inverse of event size")
    plt.show()
