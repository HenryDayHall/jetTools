# coding: utf-8
from src.Components import *
from src.FormJets import *
from src.FormShower import *

ew = EventWise.from_file("megaIgnore/arthur_Scan_perfect.awkd")
total_detectables = len(ew.DetectableTag_Roots.flatten().flatten())
print(f"Total detectable Tags {total_detectables}")


over20 = awkward.fromiter([False if not len(pt) else np.max(pt) > 20 for pt in ew.DetectableTag_PT])
num_over20 = len(ew.DetectableTag_Roots[over20].flatten().flatten())

num_under20 = total_detectables - num_over20

print(f"Of these {num_under20} lack a lead jet with PT over 20 GeV, {num_under20/total_detectables:.2%}")
sublead_over15 = ew.DetectableTag_PT[over20] > 15
sublead_over15
num_under15 = len(ew.DetectableTag_Roots[over20][~sublead_over15].flatten().flatten())

print(f"Of those with a lead jet over 20 GeV, {num_under15} have PT under 15GeV, {num_under15/total_detectables:.2%}")

over2tracks = awkward.fromiter([[len(j) > 1 for j in e] for e in ew.DetectableTag_Leaves[over20][sublead_over15]])
num_under2tracks = np.sum(~over2tracks.flatten())

print(f"Of those matching all PT requirements {num_under2tracks} do not have at least 2 tracks, {num_under2tracks/total_detectables:.2%}")

assert num_under2tracks == 0, "Need to rewrite later sections"

print("Working out which roots could be found in theory")
print("N.B. it may not be possible to find all simaltaniously")


over20_idxs = np.where(over20)[0].tolist()
possible_in_theory = []
u20 = 0; u15 = 0; over = 0
for i, event in enumerate(ew.DetectableTag_Roots):
    possible_in_theory.append([])
    for ir, roots in enumerate(event):
        
        if not over20[i]:
            u20 += len(roots)
            possible_in_theory[-1].append([False for _ in roots])
        else:
            keep_bundle = sublead_over15[over20_idxs.index(i)][ir]
           
            u15 += len(roots)*~keep_bundle
            over += len(roots)*keep_bundle
            possible_in_theory[-1].append([keep_bundle for _ in roots])
possible_in_theory = awkward.fromiter(possible_in_theory)
num_possible_in_thoery = np.sum(possible_in_theory.flatten().flatten())
print(f"There are {num_possible_in_thoery} tags for which a jet could be constructed, this is from {total_detectables} which looses {u20} to the 20PT cut and {u15} to the 15PT cut")

soljet_masses = np.sqrt(awkward.fromiter([[np.sum(e[i])**2-np.sum(x[i])**2 - np.sum(y[i])**2 - np.sum(z[i])**2 for i in idxs] for idxs, e, x, y, z in zip(ew.Solution, ew.Energy, ew.Px, ew.Py, ew.Pz)]))
soljet_pts = np.sqrt(awkward.fromiter([[np.sum(x[i])**2 + np.sum(y[i])**2 - np.sum(z[i])**2 for i in idxs] for idxs, e, x, y, z in zip(ew.Solution, ew.Energy, ew.Px, ew.Py, ew.Pz)]))

possible_roots = ew.DetectableTag_Roots[possible_in_theory]
possible_leaves = []
for i, event in enumerate(possible_roots):
    ew.selected_index = i
    jetinputs = ew.JetInputs_SourceIdx
    possible_leaves.append([])
    for bundle in event:
        leaves = FormShower.descendant_idxs(ew, *bundle)
        possible_leaves[-1].append(sorted(leaves.intersection(jetinputs)))
possible_leaves = awkward.fromiter(possible_leaves)

    
ew.selected_index = None
possible_soljets = []
for i, sol in enumerate(ew.Solution):
    possible_soljets.append([])
    leaves = set(possible_leaves[i].flatten())
    for jet in sol:
        possible_soljets[-1].append( len(leaves.intersection(jet)) > 0)
possible_soljets = awkward.fromiter(possible_soljets)        

num_possible_soljets = np.sum(possible_soljets.flatten().flatten())
print(f"Considering only jets that overlap with theroetically detectable tags the number of jets that may contain tag matterial is {num_possible_soljets}. This will inclue jets taht contain matterial from multiple tags and tags that have been split between jets")
    
    
root_in_soljet = []
total_overlaps = []
over_split = []
for i, event in enumerate(possible_roots):
    ew.selected_index = i
    here = []
    soljets = ew.Solution.tolist()
    for root in event.flatten():
        produces = FormShower.descendant_idxs(ew, root)
        found_in = [len(produces.intersection(jet)) > 0 for jet in soljets]
        here.append(np.where(found_in)[0])
    stack = np.copy(here).tolist()
    total_overlaps.append(0)
    over_split.append(0)
    while stack:
        seed = stack.pop()
        connected = np.array([len(set(seed).intersection(found_in))>0 for found_in in stack])
        group = awkward.fromiter([seed] + [stack[i] for i in np.where(connected)[0]])
        stack = [x for x, m in zip(stack, connected) if not m]
        jets_minus_roots = len(set(group.flatten())) - len(group)
        if jets_minus_roots>0:
            over_split[-1] += jets_minus_roots
        else:
            total_overlaps[-1] -= jets_minus_roots
    root_in_soljet.append(here)
root_in_soljet = awkward.fromiter(root_in_soljet)
total_overlaps = np.array(total_overlaps)
over_split = np.array(over_split)

print(f"In total there are {np.sum(total_overlaps)} tags that share one jet and {np.sum(over_split)} jets resulting from a tag split across many jets")

soljet_over20 = np.array([np.max(m, initial=0) > 20 for m in soljet_pts[possible_soljets]])
num_soljet_over20 = np.sum(possible_soljets[soljet_over20].flatten())
num_soljet_under20 = num_possible_soljets - num_soljet_over20
print(f"Of these {num_soljet_under20} lack a lead jet with PT over 20 GeV, {num_soljet_under20/num_possible_soljets:.2%}")

soljet_sublead_over15 = soljet_pts[possible_soljets][soljet_over20] > 15
num_soljet_over15 = np.sum(soljet_sublead_over15.flatten())
num_soljet_under15 = num_soljet_over20 - num_soljet_over15
print(f"Of those with a lead jet over 20 GeV, {num_soljet_under15} have PT under 15GeV, {num_soljet_under15/num_possible_soljets:.2%}")

ew.selected_index = None
soljet_tracks = awkward.fromiter([[len(i) for i in idxs] for idxs in ew.Solution])
soljet_enough_tracks = soljet_tracks[possible_soljets][soljet_over20][soljet_sublead_over15] > 1
num_soljet_enough_tracks = np.sum(soljet_enough_tracks.flatten().flatten())
num_soljet_undertracks = num_soljet_over15 - num_soljet_enough_tracks
print(f"Of those matching all PT requirements {num_soljet_undertracks} do not have at least 2 tracks, {num_soljet_undertracks/num_possible_soljets:.2%}")


print(f"We are left with {num_soljet_enough_tracks} remaining jets containing signal. Out of the total number of etectable objects this is {num_soljet_enough_tracks/total_detectables:.2%}")

"""Total detectable Tags 6971
Of these 1146 lack a lead jet with PT over 20 GeV, 16.44%
Of those with a lead jet over 20 GeV, 839 have PT under 15GeV, 12.04%
Of those matching all PT requirements 0 do not have at least 2 tracks, 0.00%
Working out which roots could be found in theory
N.B. it may not be possible to find all simaltaniously
There are 4986 tags for which a jet could be constructed, this is from 6971 which looses 1146 to the 20PT cut and 839 to the 15PT cut
/usr/local/lib/python3.6/dist-packages/awkward/array/jagged.py:1043: RuntimeWarning: invalid value encountered in sqrt
  result = getattr(ufunc, method)(*inputs, **kwargs)
  Considering only jets that overlap with theroetically detectable tags the number of jets that may contain tag matterial is 2937. This will inclue jets taht contain matterial from multiple tags and tags that have been split between jets
  In total there are 2209 tags that share one jet and 160 jets resulting from a tag split across many jets
  Of these 2555 lack a lead jet with PT over 20 GeV, 86.99%
  Of those with a lead jet over 20 GeV, 48 have PT under 15GeV, 1.63%
  Of those matching all PT requirements 0 do not have at least 2 tracks, 0.00%
  We are left with 334 remaining jets containing signal. Out of the total number of etectable objects this is 4.79%
  """
