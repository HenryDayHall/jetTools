# Jets are formed using applyFastJet.cc
# this takes in the jets, clusters them and writes to the text form of parentedTree
# read in the data
# make the jets into parented trees 
# there should be a Jet class that is an extention of ParentedTree from nltk
# use Scikit learn RobustScalar to rescale the data

class TreeWalker:
    def __init__(self, jet, node_index):
        self.jet = jet
        psudojet_ids = [ints[jet.psudojet_id_col] for ints in jet._ints]
        self.left_idx = jet._ints[node_index][jet.daughter1_col]
        self.right_idx = jet._ints[node_index][jet.daughter2_col]
        self._label = jet._floats[node_index][jet.join_distance_col]
        self._isLeaf = (self.left_idx not in psudojet_ids) and (self.right_idx not in psudojet_ids)
        self._leaf = [jet._floats[node_index][i] for i in
                      [jet.pt_col, jet.eta_col, jet.phi_col, jet.energy_col]]


    def isLeaf(self):
        return self._isLeaf

    def getLeaf(self):
        return self._leaf

    def left(self):
        return TreeWalker(self.jet, self.left_idx)

    def right(self):
        return TreeWalker(self.jet, self.right_idx)

    def label(self):
        return self._label
