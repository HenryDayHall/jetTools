# Jets are formed using applyFastJet.cc
# this takes in the jets, clusters them and writes to the text form of parentedTree
# read in the data
# make the jets into parented trees 
# there should be a Jet class that is an extention of ParentedTree from nltk
# use Scikit learn RobustScalar to rescale the data
import numpy as np
from matplotlib import pyplot as plt
from ipdb import set_trace as st
import os

class TreeWalker:
    def __init__(self, jet, node_id):
        self.jet = jet
        self.id = node_id
        psudojet_ids = [ints[jet.psudojet_id_col] for ints in jet._ints]
        node_index = psudojet_ids.index(node_id)
        self.global_obs_id = jet._ints[node_index][jet.obs_id_col]
        self.left_id = jet._ints[node_index][jet.daughter1_col]
        self.right_id = jet._ints[node_index][jet.daughter2_col]
        self.label = jet.distances
        self.is_leaf = (self.left_id not in psudojet_ids) and (self.right_id not in psudojet_ids)
        self.leaf = [jet._floats[node_index][i] for i in
                      [jet.pt_col, jet.eta_col, jet.phi_col, jet.energy_col]]
        self._decendants = None

    @property
    def left(self):
        return TreeWalker(self.jet, self.left_id)

    @property
    def right(self):
        return TreeWalker(self.jet, self.right_id)

    @property
    def decendants(self):
        if self._decendants is None:
            if self.is_leaf:
                self._decendants = set([self.global_obs_id])
            else:
                left_decendants = self.left.decendants
                right_decendants = self.right.decendants
                self._decendants = left_decendants.union(right_decendants)
        return self._decendants


# its impossible to to this properly retrospectivly... 
# the clustering has time order to it
# somthing is clearly not right here....
def tree_motion(start, root, steps_between):
    if root.is_leaf:
        return [[start]]*steps_between
    else:
        # find out what is beflow this point
        next_left = root.left.leaf[1:3]
        below_left = tree_motion(next_left, root.left, steps_between)
        next_right = root.right.leaf[1:3]
        below_right = tree_motion(next_right, root.right, steps_between)
        # if either were leaves pad to the right depth
        if len(below_left) < len(below_right):
            pad_height = int(len(below_right)-len(below_left))
            pad = pad_height * [below_left[0]]
            below_left = pad + below_left
        elif len(below_right) < len(below_left):
            pad_height = int(len(below_left)-len(below_right))
            pad = pad_height * [below_right[0]]
            below_right = pad + below_right
        levels = [r+l for r, l in zip(below_left, below_right)]
        # now make the this level
        left_stack = np.vstack((np.linspace(next_left[0], start[0], steps_between),
                                np.linspace(next_left[1], start[1], steps_between))).tolist()
        right_stack = np.vstack((np.linspace(next_right[0], start[0], steps_between),
                                 np.linspace(next_right[1], start[1], steps_between))).tolist()
        this_level = [[left, right] for left, right in zip(left_stack, right_stack)]
        levels += this_level
        return levels


def plot_motion(motion, dir_name):
    os.mkdir(dir_name)
    name_format = os.path.join(dir_name, "frame{}.png")
    last = plt.scatter(*zip(*motion[0]))
    plt.xlabel("$\\eta$")
    plt.ylabel("$\\phi$")
    xlim = plt.gca().get_xlim()
    ylim = plt.gca().get_ylim()
    for i, level in enumerate(motion):
        plt.cla()
        plt.xlabel("$\\eta$")
        plt.ylabel("$\\phi$")
        plt.xlim(xlim)
        plt.ylim(ylim)
        last = plt.scatter(*zip(*level))
        plt.savefig(name_format.format(i))

        
def quick_vid():
    import FormJets
    jet = 
