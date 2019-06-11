# Jets are formed using applyFastJet.cc
# this takes in the jets, clusters them and writes to the text form of parentedTree
# read in the data
# make the jets into parented trees 
# there should be a Jet class that is an extention of ParentedTree from nltk
# use Scikit learn RobustScalar to rescale the data

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

