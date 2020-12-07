"""somewhat messy script to play with things related to the clustering objective in a live way """
from jet_tools.tree_tagger import Components, InputTools, FormJets
import matplotlib
from ipdb import set_trace as st
import numpy as np
from matplotlib import pyplot as plt
import sklearn.metrics
from scipy.spatial.distance import pdist, squareform
import scipy.spatial
import scipy.linalg
import awkward
import os

eventWise_path = InputTools.get_file_name("EventWise with formed jets; ", '.awkd').strip()
ew = Components.EventWise.from_file(eventWise_path)

# Physical space manipulations ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# leave them as false till we calculate them
taxicab = None; euclidien = None

def get_phys_distances(eventWise, metric):
    save_name = "calculations/physdistance" + metric + ".awkd"
    if os.path.exists(save_name):
        distance = awkward.load(save_name)
        return distance
    eventWise.selected_index = None
    distance = []
    n_events = len(eventWise.X)
    for i in range(n_events):
        print(i, end='\r')
        eventWise.selected_index = i
        rap = eventWise.JetInputs_Rapidity.reshape((-1, 1))
        rap_distance = squareform(pdist(rap))
        phi = eventWise.JetInputs_Phi.reshape((-1, 1))
        phi_distance = Components.angular_distance(phi, phi.T)
        if metric == 'taxicab':
            distance.append(rap_distance + phi_distance)
        elif metric in ['angular', 'euclidien']:
            distance.append(np.sqrt(rap_distance**2 + phi_distance**2))
    distance = awkward.fromiter(distance)
    awkward.save(save_name, distance)
    return distance


def get_cluster_crossings(eventWise):
    save_name = "calculations/crossings.awkd"
    if os.path.exists(save_name):
        crossings = awkward.load(save_name)
        return crossings
    eventWise.selected_index = None
    n_events = len(eventWise.X)
    crossings = []
    for i in range(n_events):
        print(i, end='\r')
        eventWise.selected_index = i
        source_idxs = eventWise.JetInputs_SourceIdx
        sol = eventWise.Solution
        cross = np.empty((len(source_idxs), len(source_idxs)), dtype=bool)
        for r, row in enumerate(source_idxs):
            jet_n = next(n for n, jet in enumerate(sol) if row in jet)
            for c, col in enumerate(source_idxs):
                cross[r, c] = col not in sol[jet_n]
        crossings.append(cross)
    crossings = awkward.fromiter(crossings)
    awkward.save(save_name, crossings)
    return crossings
print("Calculating group crossings")
crossings = get_cluster_crossings(ew)



#if InputTools.yesNo_question("Plot raw distance distributions?"):
if False:
    print("Calculating taxicab distances")
    taxicab = get_phys_distances(ew, 'taxicab')
            
    print("Calculating euclidien distances")
    euclidien = get_phys_distances(ew, 'euclidien')
                        

    taxicross = taxicab[crossings]
    taxiin = taxicab[~crossings]
    euclicross = euclidien[crossings]
    eucliin = euclidien[~crossings]
                        
    taxi_roc = sklearn.metrics.roc_auc_score(crossings.flatten().flatten(), taxicab.flatten().flatten())
    euc_roc = sklearn.metrics.roc_auc_score(crossings.flatten().flatten(), euclidien.flatten().flatten())

    print("Plotting raw physical disances")
    plt.hist([euclicross.flatten().flatten(), eucliin.flatten().flatten(), taxicross.flatten().flatten(), taxiin.flatten().flatten()], bins=50, histtype='step', label=['euclidien crossing', 'euclidien internal', 'taxicab crossing', 'taxicab internal'])
    plt.legend()
    plt.xlabel("Euclidien distance")
    plt.ylabel("Frequency")
    plt.title(f"AUC ROC; euclidien={euc_roc:.4f}, taxicab={taxi_roc:.4f}")
    print(f"AUC ROC; euclidien={euc_roc:.4f}, taxicab={taxi_roc:.4f}")
    plt.show()
    input()

def calculate_kt_factors(eventWise):
    save_names = ["calculations/ktfactor" + m + ".awkd" for m in ["antikt", "kt", "luclus"]]
    if os.path.exists[save_names[0]]:
        antikt, kt, luclus = [awkward.load(name) for name in save_names]
        return antikt, kt, luclus
    eventWise.selected_index = None
    antikt = []
    kt = []
    luclus = []
    n_events = len(eventWise.X)
    for i in range(n_events):
        print(i, end='\r')
        eventWise.selected_index = i
        pt = eventWise.JetInputs_PT.reshape((-1, 1))
        min_pt = np.minimum(pt, pt.T)
        max_pt = np.maximum(pt, pt.T)
        antikt.append(max_pt**-2)
        kt.append(min_pt**2)
        luclus.append((pt*pt.T)/(pt+pt.T))
    antikt =awkward.fromiter(antikt)
    kt = awkward.fromiter(kt)
    luclus = awkward.fromiter(luclus)
    for name, data in zip(save_names, [antikt, kt, luclus]):
        awkward.save(name, data)
    return antikt, kt, luclus
print("Calculating kt factors")
antikt, kt, luclus = calculate_kt_factors(ew)

def cost_function(affinity, weight=None, solution_input=None):
    costs = []
    if solution_input is None:
        solution_input = solution_inputidx
    n_events = len(affinity)
    for event_n in range(n_events):
        cost = []
        aff = affinity[event_n]
        groups = solution_input[event_n]
        if len(aff) == 0:
            for jet in groups:
                cost.append(np.nan)
            costs.append(cost)
            continue
        if weight is None:
            wei = np.ones(len(aff))
        else:
            wei = np.array(weight[event_n].tolist())
        all_idx = set(list(range(len(wei))))
        if len(wei.shape) == 2:
            wei = np.sum(wei, axis=1)
        wei = np.maximum(wei - 40, 1.5)
        for jet in groups:
            outside = all_idx - set(jet)
            numerator = 0
            denominator = 0
            for in_track in jet:
                denominator += wei[in_track]
                for out_track in outside:
                    numerator += aff[in_track][out_track]
            if denominator == 0:
                cost.append(0)
            else:
                cost.append(numerator/denominator)
        costs.append(cost)
    costs = awkward.fromiter(costs)
    return costs

# creating affinities ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
jet_costs = {}
ew.selected_index = None
solution = ew.Solution
print("Converting solution to jet indices")
solution_inputidx = awkward.fromiter([[[np.where(source_idx == track)[0][0] for track in jet] for jet in jets]for jets, source_idx in zip(solution, ew.JetInputs_SourceIdx)])
signal_sets = [set(dec.flatten()) for dec in ew.DetectableTag_Leaves]
jet_signal = {}
costs = {}
#while True:
#    jet_names = FormJets.get_jet_names(ew)
#    jet_name = InputTools.list_complete("Pick a jet; ", jet_names).strip()
#    while not jet_name:
#        if not InputTools.yesNo_question("Do you want to quit? "):
#            jet_name = InputTools.list_complete("Pick a jet; ", jet_names).strip()
#        else:
#            break
#    if not jet_name:
#        break
#for jet_name in ['AntiKTp63Jet', 'AntiKTp77Jet', 'AntiKT1p0Jet', 'SpFullCAExp2TaxiJet', 'SolDenominatorJet']:
for jet_name in ['AntiKTp63Jet', 'AntiKT1p0Jet', 'SpFullCAExp2TaxiJet']:
    print(jet_name)
    params = FormJets.get_jet_params(ew, jet_name)

    if 'PhyDistance' in params:
        if params['PhyDistance'] in ['euclidien', 'angular']:
            if euclidien is None:
                euclidien = get_phys_distances(ew, 'euclidien')
            phydistance = euclidien
        elif params['PhyDistance'] == 'taxicab':
            if taxicab is None:
                taxicab = get_phys_distances(ew, 'taxicab')
            phydistance = taxicab
        else:
            if euclidien is None:
                euclidien = get_phys_distances(ew, 'euclidien')
            print(f"Don't recognise {params['PhyDistance']}")
            phydistance = euclidien
    else:
        phydistance = euclidien
        

    #ptroc = {f'{nf}': f'aucroc={sklearn.metrics.roc_auc_score(crossings.flatten().flatten(), (euclidien*f).flatten().flatten()):.5f}' for nf, f in [('kt', kt), ('antikt', antikt), ('Luclus', luclus), ('antiLuclus', luclus**-1)]}
    #print("ROC AUC are;")
    #print(ptroc)
    #print("Plotting this")
    #ptfactors = {f'{nc} {nf}': (euclidien*f)[c] for nf, f in [('kt', kt), ('antikt', antikt), ('Luclus', luclus), ('antiLuclus', luclus**-1)] for nc, c in [('crossing', crossings), ('internal', ~crossings)]}
    #plt.hist([v.flatten().flatten() for v in ptfactors.values()], bins=np.logspace(np.log(0.001), np.log(10), 50), histtype='step', label=ptfactors.keys())
    #plt.legend()
    #plt.xlabel("Euclidien distance")
    #plt.ylabel("Frequency")
    #plt.semilogx()
    #plt.show()
    #input()

    phydistance_pt = phydistance
    if 'ExpofPTPosition' in params:
        if params['ExpofPTPosition'] == 'input':
            multiplier = params['ExpofPTMultiplier']
            if not np.isclose(multiplier, 0):
                print("multiplying by pt factor")
                ptformat = params['ExpofPTFormat']
                if ptformat == 'min':
                    if multiplier < 1:
                        factor = antikt**abs(multiplier)
                    else:
                        factor = kt**multiplier
                else:
                    factor = luclus**multiplier
                # also need a counterfactor to make unitless
                event_mass2 = awkward.fromiter([np.sum(energy)**2
                                                - np.sum(px)**2 - np.sum(py)**2 - np.sum(pz)**2
                                                for energy, px, py, pz
                                                in zip(ew.JetInputs_Energy, ew.JetInputs_Px,
                                                       ew.JetInputs_Py, ew.JetInputs_Pz)])
                phydistance_pt = phydistance*factor*(event_mass2**-multiplier)

            

    affinities = {}
    print("Calculating affinities")
    values = [0.5, 1., 2., 3., 4.]
    for i in values:
        print(f"sigma = {i}")
        aff_exp_sigi = np.exp(-np.abs(phydistance_pt)/i)
        aff_exp2_sigi = np.exp(-phydistance_pt**2/i)
        affinities[f'exp(-|distance|/{i})']= aff_exp_sigi
        affinities[f'exp(-distance^2/{i})']= aff_exp2_sigi
        affinities[f'exp(-|distance|/{i}) crossing']= aff_exp_sigi[crossings]
        affinities[f'exp(-|distance|/{i}) internal']= aff_exp_sigi[~crossings]
        affinities[f'exp(-distance^2/{i}) internal']= aff_exp2_sigi[~crossings]
        affinities[f'exp(-distance^2/{i}) crossing'] = aff_exp2_sigi[crossings]

    ew.selected_index = None

    affinity_name = 'exp(-|distance|/1.0)' 
    weights = ew.JetInputs_PerfectDenominator
    #weights = affinities[affinity_name]
    #weights = awkward.fromiter([[2 for _ in event] for event in ew.JetInputs_PerfectDenominator])

    if costs:
        perfext_sum = costs[affinity_name]
    else:
        dog = cost_function(affinities[affinity_name], weights)
        perfext_sum = np.sum(dog.flatten().flatten())
    print(f"The perfect sum is {perfext_sum}")
    jet_allocations = [idxs[idxs<len(inps)] for idxs, inps in zip(getattr(ew, jet_name+"_InputIdx"), ew.JetInputs_Energy)]
    cat = cost_function(affinities[affinity_name], weights, jet_allocations)
    found_sum = np.sum(cat.flatten().flatten())
    print(f"{jet_name} has score {found_sum}")

    if not costs:
        costs = {key: cost_function(affinities[key], weights)
                     for key in [k for k in affinities if ' ' not in k]}
        costs_signal = awkward.fromiter([[len(set(source[jet]).intersection(sig)) > 0
                                             for jet in event] 
                                           for event, source, sig in 
                                           zip(solution_inputidx, ew.JetInputs_SourceIdx, signal_sets)])
    jet_costs[jet_name] = {key: cost_function(affinities[key], weights, jet_allocations)
                 for key in [k for k in affinities if ' ' not in k]}

    jet_signal[jet_name] = awkward.fromiter([[len(set(source[jet]).intersection(sig)) > 0
                                              for jet in event] 
                                             for event, source, sig in 
                                             zip(jet_allocations, ew.JetInputs_SourceIdx, signal_sets)])
                                            


# plot stuff
jet_costs2 = {k: {ke:np.nansum(va.flatten().flatten()) for ke, va in v.items() if '^2' in ke}
              for k, v in jet_costs.items()}
jet_costsa = {k: {ke:np.nansum(va.flatten().flatten()) for ke, va in v.items() if '^2' not in ke}
              for k, v in jet_costs.items()}
costsa = {k: np.nansum(v.flatten().flatten()) for k, v in costs.items() if '^2' not in k}
costs2 = {k: np.nansum(v.flatten().flatten()) for k, v in costs.items() if '^2' in k}
colours = {n: matplotlib.cm.get_cmap('hsv')(x/(len(jet_costs2)+0.2)) for x,n in enumerate(jet_costs2)}

plt.plot(values, list(costsa.values()), label='Solution, $e^{-|d|/\\sigma}$', marker='X', color='k')
plt.plot(values, list(costs2.values()), label='Solution, $e^{-d^2/\\sigma}$', marker='X', ls='--', color='k')

for jet_name in jet_costs2:
    c = colours[jet_name]
    j2 = jet_costs2[jet_name]
    ja = jet_costsa[jet_name]
    plt.plot(values, list(ja.values()), label=f'{jet_name}, $e^{{-|d|/\\sigma}}$', color=c)
    plt.plot(values, list(j2.values()), label=f'{jet_name}, $e^{{d^2/\\sigma}}$', color=c, ls='--')
    
    
plt.legend()
plt.xlabel("$\\sigma$")
plt.ylabel("Cost function")
input()
plt.clf()

# plot stuff
jet_costs2 = {k: {ke:np.nansum(va.flatten()[jet_signal[k].flatten()]) for ke, va in v.items() if '^2' in ke}
              for k, v in jet_costs.items()}
jet_costsa = {k: {ke:np.nansum(va.flatten()[jet_signal[k].flatten()]) for ke, va in v.items() if '^2' not in ke}
              for k, v in jet_costs.items()}
costsa = {k: np.nansum(v[costs_signal].flatten().flatten()) for k, v in costs.items() if '^2' not in k}
costs2 = {k: np.nansum(v[costs_signal].flatten().flatten()) for k, v in costs.items() if '^2' in k}
colours = {n: matplotlib.cm.get_cmap('hsv')(x/(len(jet_costs2)+0.2)) for x,n in enumerate(jet_costs2)}

plt.plot(values, list(costsa.values()), label='Solution, $e^{-|d|/\\sigma}$', marker='X', color='k')
plt.plot(values, list(costs2.values()), label='Solution, $e^{-d^2/\\sigma}$', marker='X', ls='--', color='k')

for jet_name in jet_costs2:
    c = colours[jet_name]
    j2 = jet_costs2[jet_name]
    ja = jet_costsa[jet_name]
    plt.plot(values, list(ja.values()), label=f'{jet_name}, $e^{{-|d|/\\sigma}}$', color=c)
    plt.plot(values, list(j2.values()), label=f'{jet_name}, $e^{{d^2/\\sigma}}$', color=c, ls='--')
    
    
plt.legend()
plt.xlabel("$\\sigma$")
plt.ylabel("Cost function")
input()

common = affinities['exp(-distance^2/1.0)']

def make_laplaciens(affinity, weights):
    laplaciens = []
    for aff, wei in zip(affinity, weights):
        if len(aff) == 0:
            laplaciens.append([])
            continue
        aff = np.array(aff.tolist())
        wei = np.array(wei.tolist())
        basic = np.diag(np.sum(aff, axis=1)) -aff
        if len(wei.shape) == 2:
            wei = np.sum(wei, axis=1)
        inverted = wei **(-0.5)
        inverted[wei==0] = 0
        inverted = np.diag(inverted)
        lap = np.matmul(inverted, np.matmul(basic, inverted))
        laplaciens.append(lap.tolist())
    laplaciens = awkward.fromiter(laplaciens)
    return laplaciens

laplaciens = make_laplaciens(common, weights)


def make_eigenvectors(laplaciens):
    eigenvectors = []; eigenvalues = []
    first_vectors = []
    for lap in laplaciens:
        try:
            values, vectors = scipy.linalg.eigh(lap.tolist())
        except ValueError:
            eigenvectors.append([])
            eigenvalues.append([])
            continue
        first_vectors.append(vectors[:, 0].tolist())
        eigenvalues.append(values[1:].tolist())
        eigenvectors.append(vectors[:, 1:].tolist())
    eigenvectors = awkward.fromiter(eigenvectors)
    eigenvalues = awkward.fromiter(eigenvalues)
    first_vectors = awkward.fromiter(first_vectors)
    return first_vectors, eigenvalues, eigenvectors

first_vectors, eigenvalues, eigenvectors = make_eigenvectors(laplaciens)


def get_distances(eigenvalues, eigenvectors, crossings, normed=True, num_eigenvetors=np.inf):
    if np.isinf(num_eigenvetors):
        num_eigenvetors = None
    if normed:
        distances = awkward.fromiter([[] if len(s) == 0 else
                                      scipy.spatial.distance.squareform(
                                          scipy.spatial.distance.pdist(np.array(s[:, :num_eigenvetors].tolist())/
                                                                       np.array(v[:num_eigenvetors].tolist())))
                                      for s, v in zip(eigenvectors, eigenvalues)])
    else:
        distances = awkward.fromiter([[] if len(s) == 0 else
                                      scipy.spatial.distance.squareform(
                                          scipy.spatial.distance.pdist(np.array(s[:, :num_eigenvetors].tolist())))
                                      for s, v in zip(eigenvectors, eigenvalues)])
    distances_inter = distances[~crossings].flatten().flatten()
    distances_cross = distances[crossings].flatten().flatten()
    plt.hist([distances_cross.tolist(), distances_inter.tolist()], histtype='step', label=['crossing', 'internal'], bins=700, normed=True)
    plt.legend()
    plt.xlabel("Eigenspace distance")
    plt.ylabel("count")
    return distances, distances_cross, distances_inter

distance, distances_cross, distances_inter = get_distances(eigenvalues, eigenvectors, crossings, True)
plt.title('exp(-distance^2/1.0), perfect weights, normed embedding space, all eigenvectors')
