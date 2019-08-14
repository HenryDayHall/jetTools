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
import TreeWalker
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
            c_rap = pjet.rap
            c_phi = pjet.phi
            for rap, phi in zip(pjet.obs_raps, pjet.obs_phis):
                plt.plot([rap, c_rap], [phi, c_phi], c=c)
        else:
            plt.scatter(pjet.obs_raps, pjet.obs_phis,
                        c=[c], marker='v', s=30, alpha=0.6)
    plt.scatter([], [], c=[c], marker='v', s=30, alpha=0.6, label="PsudoJets")
    # get soem fast jets
    directory = "./test/"
    # plot the fastjet
    psudo_colours = [colours(i) for i in np.linspace(0.6, 1., len(fastjets))]
    for c, fjet in zip(psudo_colours, fastjets):
        if lines:
            c_rap = fjet.rap
            c_phi = fjet.phi
            for rap, phi in zip(fjet.obs_raps, fjet.obs_phis):
                plt.plot([rap, c_rap], [phi, c_phi], c=c)
        else:
            plt.scatter(fjet.obs_raps, fjet.obs_phis,
                        c=[c], marker='^', s=25, alpha=0.6)
    plt.scatter([], [], c=[c], marker='^', s=25, alpha=0.6, label="FastJets")
    plt.legend()
    plt.title("Jets")
    plt.xlabel("rap")
    plt.ylabel("phi")
    plt.show()


def find_tree_prox(dir_name):
    observables = Components.Observables.from_file(dir_name)
    psudojets = FormJets.PsudoJets(observables)
    psudojets.assign_mothers()
    psudojets = psudojets.split()
    fastjets = FormJets.run_FastJet(dir_name, 1., 1)
    fastjets = fastjets.split()
    fast_coordinates = np.array([[j.phi, j.rap] for j in fastjets])
    means = np.mean(fast_coordinates, axis=0)
    fast_coordinates /= means
    for psudojet in psudojets:
        psudo_coord = [psudojet.phi, psudojet.rap]
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



def safe_rap(obs):
    if obs.perp2 == 0 and obs.e == abs(obs.pz):
        large_num = 10**10
        return np.sign(obs.pz)*large_num + obs.pz
    if obs.pz == 0.:
        return 0.
    m2 = max(obs.m2, 0.)
    mag_rap = 0.5*np.log((obs.perp2 + m2)/((obs.e + abs(obs.pz))**2))
    return -np.sign(obs.pz) * mag_rap

# not working, fine for valid rapidities, dosn;t replicate the behavior forbad rapidities
def safe_eta(rap, pt, e):
    if rap == 0.:
        return 0.
    #pz = e*(1 - np.exp(2*rap))/(1 + np.exp(2*rap))
    pz = e* np.tanh(rap)
    mag_p = np.sqrt(pt*pt + pz*pz)
    eta = 0.5*np.log((mag_p + pz)/(mag_p - pz))
    return eta, pz


def three_way_comp():
    dir_name = "./test/"
    observables = Components.Observables.from_file(dir_name)
    program= "./test_rapidity"
    rapidities = []
    towers = []
    for obs in observables.objects:
        is_tower = isinstance(obs, Components.MyTower)
        towers.append(is_tower)
        py_rap = safe_rap(obs)
        if np.isnan(py_rap):
            st()
            safe_rap(obs)
        out = subprocess.run([program, str(obs.px),
                                       str(obs.py),
                                       str(obs.pz),
                                       str(obs.e)],
                             stdout=subprocess.PIPE)
        out = [float(x) for x in out.stdout.split()]
        rapidities.append([py_rap] + out)
    rapidities = np.array(rapidities)
    plt.plot([np.min(rapidities), np.max(rapidities)],
             [np.min(rapidities), np.max(rapidities)])
    plt.scatter( rapidities[:, -1], rapidities[:, 1], label="c++")
    plt.scatter( rapidities[:, 0], rapidities[:, -1], label="python")
    plt.legend()
    plt.show()
    return rapidities, towers

# probably a sig fig issue
def tachyons_in_obs(dir_name="./test/"):
    observables = Components.Observables.from_file(dir_name)
    is_tach = [obj.e**2 < obj.pt**2 + obj.pz**2 for obj in observables.objects]
    print(f"fraction tachyions = {sum(is_tach)/len(is_tach)}")
    return is_tach


def test_tach_theory():
    """ for any particle that has a non tachionic mass we expect a match,
    for particles with tachionic mass I expect chaos """
    deltaR = 1.
    exponent_multiplier = 1
    n_pts = 500
    etas = np.random.uniform(-5., 5., (n_pts, 2))
    phis =  np.random.uniform(-np.pi, np.pi, (n_pts, 2))
    es = np.random.uniform(0., 1000., (n_pts, 2))
    pts = es * np.random.uniform(0., 2., (n_pts, 2)) 
    is_tach = []
    agree_join = []
    psudo_join = []
    fast_join = []
    raps = []
    for pt, eta, phi, e in zip(pts, etas, phis, es):
        with TestDir("test") as dir_name:
            observables = Components.make_observables(pt, eta, phi, e,
                                                      dir_name)
            tach = observables.es**2 < observables.pxs**2 + observables.pys**2 + observables.pzs**2
            raps.append([safe_rap(observables.objects[0]),
                         safe_rap(observables.objects[1])])

            is_tach.append(np.any(tach))
            psudojets, fastjets = get_psudoFast(dir_name, deltaR, exponent_multiplier)
            psudo_join.append(len(psudojets) == 1)
            fast_join.append(len(fastjets) == 1)

            agree_join.append(len(psudojets) == len(fastjets))
    is_tach = np.array(is_tach, dtype=bool)
    agree_join = np.array(agree_join)
    fast_join = np.array(fast_join)
    psudo_join = np.array(psudo_join)
    raps = np.array(raps)
    eta_dis = etas[:, 0] - etas[:, 1]
    phi_dis = phis[:, 0] - phis[:, 1]
    rap_dis = raps[:, 0] - raps[:, 1]
    return is_tach, agree_join, fast_join, psudo_join, (eta_dis, phi_dis, rap_dis)
# remeber that is everything has the same value tey al come out red....
def plt_test_tach_theory(is_tach, agree_join, fast_join, psudo_join, dis, deltaR=1., exponent_multiplier=1): 
    (eta_dis, phi_dis, rap_dis) = dis
    colour_name = "agree join"
    if "fast" in colour_name:
        colour=fast_join
    elif "home" in colour_name:
        colour= psudo_join
    elif "agree" in colour_name:
        colour = agree_join
    plt.scatter(rap_dis[is_tach], phi_dis[is_tach], c=colour[is_tach],
                marker="P", cmap='RdYlGn', label="is tachyonic")
    plt.scatter(rap_dis[~is_tach], phi_dis[~is_tach], c=colour[~is_tach],
                marker="o", cmap='RdYlGn', label="Real mass")
    # plot a circle at deltaR
    xs = np.cos(np.linspace(0, 2*np.pi, 50))*deltaR
    ys = np.sin(np.linspace(0, 2*np.pi, 50))*deltaR
    plt.plot(xs, ys, c='k')
    plt.xlabel("rap displacement")
    plt.xlim(-5, 5); plt.ylim(-np.pi, np.pi)
    plt.ylabel("$\\phi$ displacement")
    plt.legend()
    plt.title(f"Colour={colour_name} R={deltaR}, exponMul={exponent_multiplier}")
    plt.show()
            

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
    program = "./particle_print"
    found_out = []
    expected_out = []
    # also do a sweep
    pt = 10.
    e = 100.
    phi_axis_test= True
    phi = 0.
    n_pts = 100
    ones = np.ones(n_pts)
    with TestDir("test") as dir_name:
        coords = [pt*ones, np.linspace(-5., 5., n_pts), phi*ones, e*ones]
        observables2 = Components.make_observables(*coords, dir_name)
    all_objs = observables.objects + observables2.objects
    for particle in all_objs:
        out = subprocess.run([program, str(particle.px),
                                       str(particle.py),
                                       str(particle.pz),
                                       str(particle.e)],
                             stdout=subprocess.PIPE)
        out = [float(x) for x in out.stdout.split()]
        rap = safe_rap(particle)
        e_out = [particle.px, particle.py, particle.pz, particle.e,
                        particle.pt, particle.eta, particle.phi(), particle.m, rap]
        found_out.append(out[:len(e_out)])
        expected_out.append(e_out)
        np.testing.assert_allclose(out, e_out, rtol=0.01, atol=0.001)
        # check the invese eta works
        # broken
        s_eta, s_pz = safe_eta(rap, particle.pt, particle.e)
        other_rap = 0.5*np.log((particle.e + particle.pz)/(particle.e - particle.pz))
        print(f"e {particle.e}, {out[3]}")
        print(f"rap {other_rap}, {rap}, {out[8]}")
        print(f"eta {s_eta}, {particle.eta}, {out[5]}")
        print(f"pz {s_pz}, {particle.pz}, {out[2]}")
        if abs(particle.pz) < particle.e :
            print("valid combo \n")
            np.testing.assert_allclose(s_eta, particle.eta, atol =0.001)
            np.testing.assert_allclose(s_eta, out[5], atol =0.001)
        else:
            print()
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


def compare_rapidity_pseudorapidity():
    pt_set = [.1, 1., 10.]
    e_set = [1., 10., 100.]
    n_pts = 50
    etas = np.linspace(-5., 5., n_pts)
    phis = np.linspace(-np.pi, np.pi, n_pts)
    eta_grid, phi_grid = np.meshgrid(etas, phis)
    #std::cout << p.e() << sep
    #          << p.pt()  << sep
    #          << p.m() << sep
    #          << p.pz() << sep
    #          << p.eta()  << sep
    #          << p.rap() << std::endl;
    columns = ['e', 'p_T', 'm', 'p_z', '\\eta', 'rap']
    plt_col = 4
    ones = np.ones(n_pts*n_pts)
    outs = np.empty((len(e_set), len(pt_set), n_pts*n_pts, len(columns)))
    #outs = np.empty((n_pts*n_pts, len(columns)))
    observable_sets = []
    for n_e, e in enumerate(e_set):
        observable_sets.append([])
        for n_pt, pt in enumerate(pt_set):
            label = f"e={e};pt={pt}"
            print(label)
            with TestDir("test") as dir_name:
                observables = Components.make_observables(ones*pt,
                                                          eta_grid.flat,
                                                          phi_grid.flat,
                                                          ones*e,
                                                          dir_name)
                observable_sets[-1].append(observables)
                outs[n_e, n_pt] = get_cpp(observables.objects)
                print()
    return pt_set, e_set, etas, phis, outs, columns, observable_sets, 

def get_cpp(particles):
    outs = np.empty((len(particles), 6))
    program = "./rapidities"
    for i, particle in enumerate(particles):
        out = subprocess.run([program, str(particle.px),
                                       str(particle.py),
                                       str(particle.pz),
                                       str(particle.e)],
                             stdout=subprocess.PIPE)
        outs[i] = [float(x) for x in out.stdout.split()]
        if i%100 == 0:
            print('.', end='', flush=True)
    return outs


def plot_rapidity_psudorapidity(plt_col, etas, observable_sets, pt_set, e_set, outs, columns):
    fig, axarry = plt.subplots((len(e_set)))
    for n_e, (e, ax) in enumerate(zip(e_set, axarry)):
        ax.set_title(f"e={e}| Input python vs. output c++")
        for n_pt, pt in enumerate(pt_set):
            obs = observable_sets[n_e][n_pt]
            calc_rap = [safe_rap(o) for o in obs.objects]
            label = f"pt={pt}"
            print(label)
            #ax.scatter(obs.raps, outs[n_e, n_pt, :, plt_col], label=label, alpha=0.5)
            ax.scatter(calc_rap, outs[n_e, n_pt, :, plt_col], label=label, alpha=0.5)
            #plt.scatter(eta_grid.flat, outs[:, plt_col], label=label)
        ax.set_xlabel("Calculated Rapidity"); ax.set_ylabel(f"Output ${columns[plt_col]}$")
        lims = (np.min(etas), np.max(etas))
        ax.legend()
        ax.set_xlim(lims); ax.set_ylim(lims)
    return axarry


def simple_test():
    trivial_tests = True
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
    pt = 100.
    e = 10.
    expected_join_distance = 1.
    # start along the phi axis
    phi_axis_test= True
    fast_distances = []
    psudo_distances = []
    rapidity0 = []
    rapidity1 = []
    if phi_axis_test:
        print("Make two points and move them together until they merge")
        phi = 0.
        eta1 = 0.
        for eta2 in np.linspace(-5., 5., 200):
            with TestDir("test") as dir_name:
                coords = [[pt, pt], [eta1, eta2], [phi, phi], [e, e]]
                observables = Components.make_observables(*coords, dir_name)
                psudojets, fastjets, psudo_d, fast_d = compare_distances(dir_name, deltaR, exponent_multiplier, observables)
                # calcuate the safe rapidity
                rap0 = safe_rap(observables.objects[0])
                rap1 = safe_rap(observables.objects[1])
                should_join = abs(rap0 - rap1) < expected_join_distance
                fast_distances.append(fast_d)
                psudo_distances.append(psudo_d)
                rapidity0.append(rap0)
                rapidity1.append(rap1)
                if should_join:
                    print(f"Psudo distance = {psudo_d}, fast distance = {fast_d}")
                    assert len(psudojets) == 1, f"Expect merge at {coords} but have {len(psudojets)} jets"
                    # assert len(fastjets) == 1, f"Expect merge at {coords} but have {len(fastjets)} jets" 
                else:
                    assert len(psudojets) == 2, f"Expect no merge at {coords} but have {len(psudojets)} jets"
                    # assert len(fastjets) == 2, f"Expect no merge at {coords} but have {len(fastjets)} jets"
    # rap axis
    rap_axis_test = False
    if rap_axis_test:
        phi1 = 0.
        rap = 0.
        for phi2 in np.linspace(0., 2*np.pi, 30):
            # remeber to account for cyclic behavior
            should_join = phi2 < expected_join_distance or (np.pi*2 - phi2) < expected_join_distance
            with TestDir("test") as dir_name:
                coords = [[pt, pt], [rap, rap], [phi1, phi2], [e, e]]
                Components.make_observables(*coords, dir_name)
                psudojets, fastjets = get_psudoFast(dir_name, deltaR, exponent_multiplier)
                if should_join:
                    assert len(psudojets) == 1, f"Expect merge at {coords} but have {len(psudojets)} jets"
                    assert len(fastjets) == 1, f"Expect merge at {coords} but have {len(fastjets)} jets" 
                else:
                    assert len(psudojets) == 2, f"Expect no merge at {coords} but have {len(psudojets)} jets"
                    assert len(fastjets) == 2, f"Expect no merge at {coords} but have {len(fastjets)} jets"
    return fast_distances, psudo_distances, rapidity0, rapidity1

def joins(eta, phi, pt=100., e=.1, deltaR=1., exponent_multiplier=-1):
    """ find out if the discribed jet would be joined with a jet at the origin under fastjet of psudojet """
    with TestDir("test") as dir_name:
        coords = [[pt, pt], [0, eta], [0, phi], [e, e]]
        Components.make_observables(*coords, dir_name)
        psudojets, fastjets = get_psudoFast(dir_name, deltaR, exponent_multiplier)
        return len(psudojets) == 1, len(fastjets) == 1


def plot_joins(pt=10., e=100.):
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
                psu, fas = joins(eta, phi, pt=pt, e=e)
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
    plt.title(f"Behavor comparison; pt={pt}, e={e}")
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

def compare_distances(dir_name, deltaR, exponent_multiplyer, observables=None):
    if observables is None:
        observables = Components.Observables.from_file(dir_name)
    psudojets = FormJets.PsudoJets(observables, deltaR=deltaR,
                                   exponent_multiplyer=exponent_multiplyer)
    row, column = np.unravel_index(np.argmin(psudojets._distances), psudojets._distances.shape)
    psudo_merge = row != column
    psudo_d = psudojets._distances[row, column]

    psudojets.assign_mothers()
    psudojets = psudojets.split()
    fastjets, out = FormJets.run_FastJet(dir_name, deltaR, exponent_multiplyer, capture_out=True)
    lines = out.split('\n')
    fast_d = float(lines[-2])
    if len(lines) > 2:
        print(lines[0])
    fastjets = fastjets.split()
    return psudojets, fastjets, psudo_d, fast_d


if __name__ == '__main__':
    #compare_psudoFast(deltaR=0.8, exponent_multiplyer=1)
    #plot_psudofast(lines=True, deltaR=0.8, exponent_multiplyer=1)
    #simple_test()
    #plot_joins()
    # coord_check()
    #speed_test()
    pass

        
