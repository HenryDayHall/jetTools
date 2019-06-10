""" Calculate seperation between two binary rooted trees with distinct leaves"""
import numpy as np
import operator

def seperation(root1, root2):
    """ number of maximium matched items divided by number of items at each depth """
    levels1 = [[root1]]
    levels2 = [[root2]]
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
        ratios.append(2*matched/twice_total)
    weights = [2**(-depth) for depth in range(len(ratios))]
    ratio = np.average(ratios, weights=weights)
    return ratio





        
