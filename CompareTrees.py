""" Calculate proximity between two binary rooted trees with distinct leaves"""
import InputTools
import subprocess
import time
import os
from matplotlib import pyplot as plt
import matplotlib
from ipdb import set_trace as st
import numpy as np
import operator
from unholyMatrimony import TreeWalker
import Components
import FormJets

def proximity(root1, root2):
    """ number of maximium matched items divided by number of items at each depth """
    levels1 = [[root1]]
    levels2 = [[root2]]
    all_decendants1 = root1.decendants
    all_decendants2 = root2.decendants
    overlap_percent = (len(all_decendants1.intersection(all_decendants2))/
                      len(all_decendants1.union(all_decendants2)))
    print(f"Percantage of leaves in common = {100*overlap_percent}%")

    ratios = [overlap_percent]
    while len(levels1[-1]) > 0 and len(levels2[-1]) > 0:
        levels1.append([])
        levels2.append([])
        matched = 0
        twice_total = 0
        for node1, node2 in zip(levels1[-2], levels2[-2]):
            if node1.is_leaf or node2.is_leaf:
                continue
            left1_set = set(node1.left.decendants)
            right1_set = set(node1.right.decendants)
            left2_set = set(node2.left.decendants)
            right2_set = set(node2.right.decendants)
            twice_total += len(left1_set) + len(right1_set) + \
                           len(left2_set) + len(right2_set)
            congruent = len(left1_set.intersection(left2_set)) + \
                        len(right1_set.intersection(right2_set))
            switch = len(left1_set.intersection(right2_set)) + \
                        len(right1_set.intersection(left2_set))
            if congruent > switch:
                levels1[-1].append(node1.left)
                levels2[-1].append(node2.left)
                levels1[-1].append(node1.right)
                levels2[-1].append(node2.right)
                matched += congruent
            else:
                levels1[-1].append(node1.left)
                levels2[-1].append(node2.right)
                levels1[-1].append(node1.right)
                levels2[-1].append(node2.left)
                matched += switch
        # print(f"matched = {matched}/{twice_total/2}")
        if matched > 0:
            ratios.append(2*matched/twice_total)
    weights = [2**(-depth) for depth in range(len(ratios))]
    ratio = np.average(ratios, weights=weights)
    return ratio

def canopy_proximity(jets1, jets2):
    canopies1 = [set(jet.global_obs_ids[jet.global_obs_ids!=-1])
                 for jet in jets1]
    canopies2 = [set(jet.global_obs_ids[jet.global_obs_ids!=-1])
                 for jet in jets2]
    selected1 = np.full(len(canopies1), False)
    selected2 = np.full(len(canopies2), False)
    dif_grid = np.full((len(canopies1), len(canopies2)), np.inf)
    matched_grid = np.zeros((len(canopies1), len(canopies2)))
    for x, c1 in enumerate(canopies1):
        for y, c2 in enumerate(canopies2):
            if len(c1.intersection(c2)) > 0:
                matched_grid[x, y] = len(c1.intersection(c2))
                dif_grid[x, y] = len(c1.symmetric_difference(c2))/matched_grid[x, y]
    num_matched = 0
    while False in selected1 and False in selected2 and np.any(np.isfinite(dif_grid)):
        x_min, y_min = np.unravel_index(np.argmin(dif_grid), dif_grid.shape)
        num_matched += matched_grid[x_min, y_min]
        # zero out that index
        dif_grid[x_min, :] = np.inf
        dif_grid[: , y_min] = np.inf
        selected1[x_min] = True
        selected2[y_min] = True
    total_number = np.sum(matched_grid)
    return num_matched, total_number

def compare_psudoFast(dir_name="./test", deltaR=1., exponent_multiplyer=-1):
    import Components
    observables = Components.Observables.from_file(dir_name)
    import FormJets
    psudojets = FormJets.PsudoJets(observables, deltaR, exponent_multiplyer=exponent_multiplyer)
    psudojets.assign_mothers()
    psudojets = psudojets.split()
    fastjets = FormJets.run_FastJet(dir_name, deltaR, exponent_multiplyer)
    fastjets = fastjets.split()
    matched, total = canopy_proximity(psudojets, fastjets)
    print(f"Matched {matched} out of {total}")


def plot_psudofast(dir_name="./test/", lines=False, deltaR=1., exponent_multiplyer=-1):
    # colourmap
    colours = plt.get_cmap('gist_rainbow')
    import ReadSQL
    # get some psudo jets
    import Components
    observables = Components.Observables.from_file(dir_name)
    import FormJets
    psudojets = FormJets.PsudoJets(observables, deltaR=deltaR, exponent_multiplyer=exponent_multiplyer)
    psudojets.assign_mothers()
    psudojets = psudojets.split()
    fastjets = FormJets.run_FastJet(dir_name, deltaR, exponent_multiplyer)
    fastjets = fastjets.split()
    # plot the psudojets
    psudo_colours = [colours(i) for i in np.linspace(0, 0.4, len(psudojets))]
    for c, pjet in zip(psudo_colours, psudojets):
        if lines:
            c_eta = pjet.eta
            c_phi = pjet.phi
            for eta, phi in zip(pjet.obs_etas, pjet.obs_phis):
                plt.plot([eta, c_eta], [phi, c_phi], c=c)
        else:
            plt.scatter(pjet.obs_etas, pjet.obs_phis,
                        c=[c], marker='v', s=30, alpha=0.6)
    plt.scatter([], [], c=[c], marker='v', s=30, alpha=0.6, label="PsudoJets")
    # get soem fast jets
    directory = "./test/"
    # plot the fastjet
    psudo_colours = [colours(i) for i in np.linspace(0.6, 1., len(fastjets))]
    for c, fjet in zip(psudo_colours, fastjets):
        if lines:
            c_eta = fjet.eta
            c_phi = fjet.phi
            for eta, phi in zip(fjet.obs_etas, fjet.obs_phis):
                plt.plot([eta, c_eta], [phi, c_phi], c=c)
        else:
            plt.scatter(fjet.obs_etas, fjet.obs_phis,
                        c=[c], marker='^', s=25, alpha=0.6)
    plt.scatter([], [], c=[c], marker='^', s=25, alpha=0.6, label="FastJets")
    plt.legend()
    plt.title("Jets")
    plt.xlabel("eta")
    plt.ylabel("phi")
    plt.show()

def find_tree_prox(dir_name):
    observables = Components.Observables.from_file(dir_name)
    psudojets = FormJets.PsudoJets(observables)
    psudojets.assign_mothers()
    psudojets = psudojets.split()
    fastjets = FormJets.run_FastJet(dir_name, 1., 1)
    fastjets = fastjets.split()
    fast_coordinates = np.array([[j.phi, j.eta] for j in fastjets])
    means = np.mean(fast_coordinates, axis=0)
    fast_coordinates /= means
    for psudojet in psudojets:
        psudo_coord = [psudojet.phi, psudojet.eta]
        # normalise points
        psudo_coord /= means
        # pick closest match
        diff2 = np.sum((fast_coordinates - psudo_coord)**2, axis=1)
        min_idx = np.argmin(diff2)
        fastjet = fastjets[min_idx]
        psudo_tree_walker = TreeWalker.TreeWalker(psudojet, psudojet.root_psudojetIDs[0])
        fast_tree_walker = TreeWalker.TreeWalker(fastjet, fastjet.root_psudojetIDs[0])
        #prox = proximity(psudo_tree_walker, psudo_tree_walker)
        prox = proximity(psudo_tree_walker, fast_tree_walker)
        print(f"Proximity is {prox}")


# context manager for test directory
class TestDir:
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


def get_psudoFast(dir_name, deltaR, exponent_multiplyer):
    observables = Components.Observables.from_file(dir_name)
    psudojets = FormJets.PsudoJets(observables, deltaR=deltaR,
                                   exponent_multiplyer=exponent_multiplyer)
    psudojets.assign_mothers()
    psudojets = psudojets.split()
    fastjets = FormJets.run_FastJet(dir_name, deltaR, exponent_multiplyer)
    fastjets = fastjets.split()
    return psudojets, fastjets

def speed_test():
    print("Running a speed test")
    # make some data
    n_pts = 100
    n_loops = 100
    n_trials = 20
    deltaR = 1.
    psudo_time = []
    fast_time = []
    for t in range(n_trials):
        print(f"Starting trial {t}")
        exponent_multiplyer = -1
        algorithm_num = 1
        pts = np.random.uniform(0, 100, n_pts)
        etas = np.random.uniform(0, 5, n_pts)
        phis = np.random.uniform(-np.pi, np.pi, n_pts)
        es = np.random.uniform(10**(-10), 10**(-8), n_pts)
        with TestDir("speed") as dir_name:
            Components.make_observables(pts, etas, phis, es, dir_name)
            observables = Components.Observables.from_file(dir_name)
            psudo_start = time.time()
            for _ in range(n_loops):
                psudojets = FormJets.PsudoJets(observables, deltaR=deltaR,
                                               exponent_multiplyer=exponent_multiplyer)
                psudojets.assign_mothers()
                psudojets = psudojets.split()
            psudo_end = time.time()
            psudo_time.append(psudo_end - psudo_start)
            fast_start = time.time()
            for _ in range(n_loops):
                program_name = "unholyMatrimony/applyFastJet"
                subprocess.run([program_name, dir_name, str(deltaR), str(algorithm_num)])
                # readding the data back in is not part of fastjet
                #fastjets = FormJets.PsudoJets.read(dir_name, fastjet_format=True)
            fast_end = time.time()
            fast_time.append(fast_end - fast_start)
    psudo_ave = np.average(psudo_time)
    fast_ave = np.average(fast_time)
    psudo_result = (f"ran psudojet on {n_pts} points for {n_loops} loops in {n_trials} trials\n" +
            f"Average time for {n_loops} loops = {psudo_ave}")
    fast_result = (f"ran fastjet on {n_pts} points for {n_loops} loops in {n_trials} trials\n" +
            f"Average time for {n_loops} loops = {fast_ave}")
    print(psudo_result)
    print(fast_result)
    results = np.array(list(zip(psudo_time, fast_time)))
    np.savetxt("speed_test.csv", results, header="psudo_time fast_time")

def coord_check():
    observables = Components.Observables.from_file("test")
    program = "unholyMatrimony/particle_print"
    found_out = []
    expected_out = []
    for particle in observables.objects:
        out = subprocess.run([program, str(particle.px),
                                       str(particle.py),
                                       str(particle.pz),
                                       str(particle.e)],
                             stdout=subprocess.PIPE)
        out = [float(x) for x in out.stdout.split()]
        found_out.append(out)
        e_out = [particle.px, particle.py, particle.pz, particle.e,
                        particle.pt, particle.eta, particle.phi(), particle.m]
        expected_out.append(e_out)
        # np.testing.assert_allclose(out, e_out, rtol=0.01)
    found_out = np.array(found_out)
    expected_out = np.array(expected_out)
    average = np.average((found_out + expected_out)/2, axis=0)
    distance = found_out - expected_out
    ave_distances = np.average(distance/average, axis=0)
    std_distances = np.std(distance/average, axis=0)
    max_distances = np.nanmax(distance/average, axis=0)
    print(f"average distances {ave_distances}")
    print(f"standard devations {std_distances}")
    print(f"max {max_distances}")




def simple_test():
    trivial_tests = False
    if trivial_tests:
        with TestDir("test") as dir_name:
            print("Test to see if an empty observables list creates no jets")
            print("(ignore UserWarnings from genfromtxt)")
            Components.make_observables([], [], [], [], dir_name)
            psudojets, fastjets = get_psudoFast(dir_name, 1., 1)
            assert len(psudojets) == 0
            assert len(fastjets) == 0
        with TestDir("test") as dir_name:
            print("Test to see if one observabel creates one jet")
            Components.make_observables([0.5], [0], [0], [0], dir_name)
            psudojets, fastjets = get_psudoFast(dir_name, 1., 1)
            assert len(psudojets) == 1
            assert len(fastjets) == 1
            find_tree_prox(dir_name)
    deltaR = 1.
    exponent_multiplier = -1
    pt = 1.
    e = 10**(-9)
    expected_join_distance = 1.
    # start along the phi axis
    phi_axis_test= True
    if phi_axis_test:
        print("Make two points and move them together until they merge")
        phi = 0.
        eta1 = 0.
        for eta2 in np.linspace(-5., 5., 200):
            should_join = eta2 < expected_join_distance
            with TestDir("test") as dir_name:
                coords = [[pt, pt], [eta1, eta2], [phi, phi], [e, e]]
                Components.make_observables(*coords, dir_name)
                psudojets, fastjets = get_psudoFast(dir_name, deltaR, exponent_multiplier)
                # if should_join:
                #     assert len(psudojets) == 1, f"Expect merge at {coords} but have {len(psudojets)} jets"
                #     assert len(fastjets) == 1, f"Expect merge at {coords} but have {len(fastjets)} jets" 
                # else:
                #     assert len(psudojets) == 2, f"Expect no merge at {coords} but have {len(psudojets)} jets"
                #     assert len(fastjets) == 2, f"Expect no merge at {coords} but have {len(fastjets)} jets"
    # eta axis
    eta_axis_test = False
    if eta_axis_test:
        phi1 = 0.
        eta = 0.
        for phi2 in np.linspace(0., 2*np.pi, 30):
            # remeber to account for cyclic behavior
            should_join = phi2 < expected_join_distance or (np.pi*2 - phi2) < expected_join_distance
            with TestDir("test") as dir_name:
                coords = [[pt, pt], [eta, eta], [phi1, phi2], [e, e]]
                Components.make_observables(*coords, dir_name)
                psudojets, fastjets = get_psudoFast(dir_name, deltaR, exponent_multiplier)
                if should_join:
                    assert len(psudojets) == 1, f"Expect merge at {coords} but have {len(psudojets)} jets"
                    assert len(fastjets) == 1, f"Expect merge at {coords} but have {len(fastjets)} jets" 
                else:
                    assert len(psudojets) == 2, f"Expect no merge at {coords} but have {len(psudojets)} jets"
                    assert len(fastjets) == 2, f"Expect no merge at {coords} but have {len(fastjets)} jets"

def joins(eta, phi, pt=1., e=0.1, deltaR=1., exponent_multiplier=-1):
    """ find out if the discribed jet would be joined with a jet at the origin under fastjet of psudojet """
    with TestDir("test") as dir_name:
        coords = [[pt, pt], [0, eta], [0, phi], [e, e]]
        Components.make_observables(*coords, dir_name)
        psudojets, fastjets = get_psudoFast(dir_name, deltaR, exponent_multiplier)
        return len(psudojets) == 1, len(fastjets) == 1

def plot_joins():
    recalculate = InputTools.yesNo_question("Recalculate points? ")
    eta_lims = (-5, 5)
    phi_lims = (-np.pi, np.pi)
    if recalculate:
        n_pts = 50
        etas = np.linspace(*eta_lims, n_pts)
        phis = np.linspace(*phi_lims, n_pts)
        fast_join = np.zeros((n_pts, n_pts))
        psudo_join = np.zeros((n_pts, n_pts))
        for neta, eta in enumerate(etas):
            for nphi, phi in enumerate(phis):
                psu, fas = joins(eta, phi)
                fast_join[nphi, neta] = fas
                psudo_join[nphi, neta] = psu
        np.savetxt("fast_joins.csv", fast_join)
        np.savetxt("psudo_joins.csv", psudo_join)
    else:
        fast_join = np.genfromtxt("fast_joins.csv")
        psudo_join = np.genfromtxt("psudo_joins.csv")

    plot_colours, colors_used = area_heatmap2(fast_join, psudo_join) 
    plt.imshow(plot_colours, extent = (*eta_lims, *phi_lims), interpolation='none')
    legend_elements = [matplotlib.lines.Line2D([0], [0], marker='o',
                                               label='fastjet joins',
                                               markerfacecolor=colors_used[0],
                                               markersize=15),
                       matplotlib.lines.Line2D([0], [0], marker='o',
                                               label='homejet joins',
                                               markerfacecolor=colors_used[1],
                                               markersize=15),
                       matplotlib.lines.Line2D([0], [0], marker='o',
                                               label='second track',
                                               markerfacecolor='black',
                                               markersize=15)]
    plt.scatter([0], [0], c='black') 
    plt.legend(handles=legend_elements)
    plt.xlabel("$\\eta$")
    plt.ylabel("$\\phi$")
    plt.title("Behavor comparison")
    plt.show()

def area_heatmap(*heat_arrays):
    colourmap = matplotlib.cm.get_cmap("gist_rainbow")
    num_heats = len(heat_arrays)
    heat_colours = [colourmap(i) for i in np.linspace(0., 1., num_heats)]
    plot_colours = np.zeros((*heat_arrays[0].shape, 4))
    for xn in range(plot_colours.shape[0]):
        for yn in range(plot_colours.shape[1]):
            weights = [h[xn, yn] for h in heat_arrays]
            plot_colours[xn, yn] = colour_average(heat_colours, weights)
    return plot_colours
    
def area_heatmap2(*heat_arrays):
    colourmap = matplotlib.cm.get_cmap("tab10")
    #colourmap = matplotlib.cm.get_cmap("gist_rainbow")
    num_heats = len(heat_arrays)
    heat_colours = [np.array(colourmap(i)) for i in np.linspace(0., 1., num_heats)]
    #heat_colours = [np.random.uniform(0, 1, 4) for i in range(num_heats)]
    shape = heat_arrays[0].shape
    plot_colours = np.zeros((shape[0], shape[1]*num_heats, 4))
    for xn in range(plot_colours.shape[0]):
        for yn in range(plot_colours.shape[1]):
            temp = heat_arrays[yn%num_heats][xn, int(yn/num_heats)]
            plot_colours[xn, yn] = heat_colours[yn%num_heats]*temp
    return plot_colours, heat_colours
    
def colour_average(colours, weights):
    rgb = np.array([c[:3] for c in colours])
    w_rgb = np.array([w * c for w, c in zip(weights, rgb)])
    newrgba = np.sqrt(np.sum(w_rgb**2, axis=0))
    newalpha = np.average([c[-1] for c in colours])
    return (*newrgba, newalpha)


if __name__ == '__main__':
    #compare_psudoFast(deltaR=0.8, exponent_multiplyer=1)
    #plot_psudofast(lines=True, deltaR=0.8, exponent_multiplyer=1)
    #simple_test()
    plot_joins()
    # coord_check()
    #speed_test()

        
