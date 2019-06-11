""" Calculate proximity between two binary rooted trees with distinct leaves"""
from ipdb import set_trace as st
import numpy as np
import operator
from unholyMatrimony import TreeWalker
import FormJets

def proximity(root1, root2):
    """ number of maximium matched items divided by number of items at each depth """
    levels1 = [[root1]]
    levels2 = [[root2]]
    all_decendants1 = root1.decendants
    all_decendants2 = root2.decendants
    overlap_percent = (len(all_decendants1.intersection(all_decendants2))/
                      len(all_decendants1.union(all_decendants2)))
    print(f"Percantage of leaves in common = {overlap_percent}%")

    ratios = []
    while len(levels1[-1]) > 0 and len(levels2[-1]) > 0:
        levels1.append([])
        levels2.append([])
        matched = 0
        twice_total = 0
        for node1, node2 in zip(levels1[-2], levels2[-2]):
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
                if (len(left1_set) > 1 and # left1 not a leaf
                    len(left2_set) > 1): # left2 not a leaf
                    levels1[-1].append(node1.left)
                    levels2[-1].append(node2.left)
                if (len(right1_set) > 1 and # right1 not a leaf
                    len(right2_set) > 1): # right2 not a leaf
                    levels1[-1].append(node1.right)
                    levels2[-1].append(node2.right)
                matched += congruent
            else:
                if (len(left1_set) > 1 and # left1 not a leaf
                    len(right2_set) > 1): # right2 not a leaf
                    levels1[-1].append(node1.left)
                    levels2[-1].append(node2.right)
                if (len(right1_set) > 1 and # right1 not a leaf
                    len(left2_set) > 1): # left2 not a leaf
                    levels1[-1].append(node1.right)
                    levels2[-1].append(node2.left)
                matched += switch
        print(f"matched = {matched}/{twice_total/2}")
        ratios.append(2*matched/twice_total)
    weights = [2**(-depth) for depth in range(len(ratios))]
    ratio = np.average(ratios, weights=weights)
    return ratio


def main():
    directory = "./test/"
    psudojet = FormJets.PsudoJets.read(directory, save_number=1, fastjet_format=False)
    fastjets = FormJets.PsudoJets.read(directory, fastjet_format=True)
    jets = fastjets.split()
    coordinates = np.array([[j.e, j.pt, j.phi, j.eta] for j in jets])
    means = np.mean(coordinates, axis=0)
    psudo_coord = [psudojet.e, psudojet.pt, psudojet.phi, psudojet.eta]/means
    coordinates = coordinates/means
    diff2 = np.sum((coordinates - psudo_coord)**2, axis=1)
    min_idx = np.argmin(diff2)
    fastjet = jets[min_idx]

    psudo_tree_walker = TreeWalker.TreeWalker(psudojet, psudojet.root_psudojetIDs[0])
    fast_tree_walker = TreeWalker.TreeWalker(fastjet, fastjet.root_psudojetIDs[0])
    #prox = proximity(psudo_tree_walker, psudo_tree_walker)
    prox = proximity(psudo_tree_walker, fast_tree_walker)
    print(f"Proximity is {prox}")
    return prox

if __name__ == '__main__':
    main()

        
