import numpy as np
from matplotlib import pyplot as plt
from ipdb import set_trace as st


class PsudoJets:
    def __init__(self, observables, deltaR=.4, exponent_multiplyer=-1, **kwargs):
        self.deltaR = deltaR
        self.exponent_multiplyer = exponent_multiplyer
        self.exponent = 2 * exponent_multiplyer
        # make a table of ints and a table of floats
        # lists not arrays, becuase they will grow
        # int columns
        # psudojet_id global_obs_id mother daughter1 daughter2 rank
        self.psudojet_id_col = 0
        self.obs_id_col = 1
        self.mother_col = 2
        self.daughter1_col = 3
        self.daughter2_col = 4
        self.rank_col = 5
        # float columns
        # pt eta phi energy join_distance
        self.pt_col = 0
        self.eta_col = 1
        self.phi_col = 2
        self.energy_col = 3
        self.join_distance_col = 4
        if 'ints' in kwargs:
            self._ints = kwargs['ints']
            self._floats = kwargs['floats']
        else:
            self._ints = [[i, oid, -1, -1, -1, -1] for i, oid in enumerate(observables.global_obs_ids)]
            self._floats = np.hstack((observables.pts.reshape((-1, 1)),
                                      observables.etas.reshape((-1, 1)),
                                      observables.phis.reshape((-1, 1)),
                                      observables.es.reshape((-1, 1)),
                                      np.zeros((len(observables), 1)))).tolist()
        # keep track of how many clusters don't yet have a mother
        self.currently_avalible = sum([p[self.mother_col]==-1 for p in self._ints])
        self._distances = None
        self._calculate_distances()
        # as we go note the root notes of the psudojets
        self.root_psudojetIDs = []

    @property
    def global_obs_ids(self):
        return np.array([ints[self.obs_id_col] for ints in self._ints])

    @property
    def global_jet_ids(self):
        return np.array([ints[self.psudojet_id_col] for ints in self._ints])

    @property
    def ranks(self):
        return np.array([ints[self.rank_col] for ints in self._ints])

    @property
    def distances(self):
        return np.array([floats[self.join_distance_col] for floats in self._floats])

    @property
    def mothers(self):
        return np.array([ints[self.mother_col] for ints in self._ints])

    def split(self):
        self.grouped_psudojets = [self.get_decendants(lastOnly=False, psudojetID=psudojetID)
                                  for psudojetID in self.root_psudojetIDs]
        self.JetList = []
        for group in self.grouped_psudojets:
            group_idx = [self.idx_from_ID(ID) for ID in group]
            ints = [self._ints[i] for i in group_idx]
            floats = [self._floats[i] for i in group_idx]
            jet = PsudoJets(self.deltaR, self.exponent_multiplyer,
                            ints=ints, floats=floats)
            self.JetList.append(jet)
        return self.JetList
    
    def _calculate_distances(self):
        # this is caluculating all the distances
        self._distances = np.full((self.currently_avalible, self.currently_avalible), np.inf)
        # for speed, make local variables
        pt_col  = self.pt_col 
        eta_col = self.eta_col
        phi_col = self.phi_col
        exponent = self.exponent
        deltaR = self.deltaR
        for row in range(self.currently_avalible):
            for column in range(self.currently_avalible):
                if column > row:
                    continue  # we only need a triangular matrix due to symmetry
                elif column == row:
                    distance = self._floats[row][pt_col]**exponent * deltaR
                else:
                    angular_diffrence = self._floats[row][phi_col] - self._floats[column][phi_col]
                    angular_distance = min(abs(angular_diffrence), abs(2*np.pi - angular_diffrence))
                    distance = min(self._floats[row][pt_col]**exponent, self._floats[column][pt_col]**exponent) *\
                               ((self._floats[row][eta_col] - self._floats[column][eta_col])**2 +
                               (angular_distance)**2)
                self._distances[row, column] = distance

    
    def _recalculate_one(self, cluster_num):
        for row in range(self.currently_avalible):
            column = cluster_num
            if column > row:
                row, column = column, row  # keep the upper triangular form
            if column == row:
                distance = self._floats[row][self.pt_col]**self.exponent * self.deltaR
            else:
                angular_diffrence = self._floats[row][self.phi_col] - self._floats[column][self.phi_col]
                angular_distance = min(abs(angular_diffrence), abs(2*np.pi - angular_diffrence))
                distance = min(self._floats[row][self.pt_col]**self.exponent, self._floats[column][self.pt_col]**self.exponent) *\
                           ((self._floats[row][self.eta_col] - self._floats[column][self.eta_col])**2 +
                           (angular_distance)**2)
            self._distances[row, column] = distance

    def _merge_psudojets(self, psudojet_index1, psudojet_index2, distance):
        new_psudojet_ints, new_psudojet_floats = self._combine(psudojet_index1, psudojet_index2, distance)
        # move the first psudojet to the back without replacement
        psudojet1_ints = self._ints.pop(psudojet_index1)
        psudojet1_floats = self._floats.pop(psudojet_index1)
        self._ints.append(psudojet1_ints)
        self._floats.append(psudojet1_floats)
        # move the second psudojet to the back but replace it with the new psudojet
        psudojet2_ints = self._ints[psudojet_index2]
        psudojet2_floats = self._floats[psudojet_index2]
        self._ints.append(psudojet2_ints)
        self._floats.append(psudojet2_floats)
        self._ints[psudojet_index2] = new_psudojet_ints
        self._floats[psudojet_index2] = new_psudojet_floats
        # one less psudojet avalible
        self.currently_avalible -= 1
        
        # delete the first row and column of the merge
        self._distances = np.delete(self._distances, (psudojet_index1), axis=0)
        self._distances = np.delete(self._distances, (psudojet_index1), axis=1)

        # now recalculate for the new psudojet
        self._recalculate_one(psudojet_index2)

    def _remove_psudojet(self, psudojet_index):
        # move the first psudojet to the back without replacement
        psudojet_ints = self._ints.pop(psudojet_index)
        psudojet_floats = self._floats.pop(psudojet_index)
        self._ints.append(psudojet_ints)
        self._floats.append(psudojet_floats)
        self.root_psudojetIDs.append(psudojet_ints[self.psudojet_id_col])
        # delete the row and column
        self._distances = np.delete(self._distances, (psudojet_index), axis=0)
        self._distances = np.delete(self._distances, (psudojet_index), axis=1)
        # one less psudojet avalible
        self.currently_avalible -= 1
        

    def assign_mothers(self):
        while self.currently_avalible > 0:
            # now find the smallest distance
            row, column = np.unravel_index(np.argmin(self._distances), self._distances.shape)
            if row == column:
                self._remove_psudojet(row)
            else:
                self._merge_psudojets(row, column, self._distances[row, column])

    def plt_assign_mothers(self):
        # dendogram < this should be
        plt.axis([-5, 5, -np.pi-0.5, np.pi+0.5])
        etas = [p[self.eta_col] for p in self._floats]
        phis = [p[self.phi_col] for p in self._floats]
        es = [p[self.energy_col] for p in self._floats]
        pts = [1/p[self.pt_col]**2 for p in self._floats]
        plt.scatter(etas, phis, pts, marker='D', c='w')
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.ylabel(r"$\phi$ - barrel angle")
        plt.xlabel(r"$\eta$ - psudo rapidity")
        plt.title("Detected Hits")
        plt.gca().set_facecolor('gray')
        # for getting rid of the axis
        #plt.gca().get_xaxis().set_visible(False)
        #plt.gca().get_yaxis().set_visible(False)
        #plt.gca().spines['top'].set_visible(False)
        #plt.gca().spines['right'].set_visible(False)
        #plt.gca().spines['bottom'].set_visible(False)
        #plt.gca().spines['left'].set_visible(False)
        plt.pause(0.05)#
        input("Press enter to start psudojeting")
        while self.currently_avalible > 0:
            # now find the smallest distance
            row, column = np.unravel_index(np.argmin(self._distances), self._distances.shape)
            if row == column:
                decendents = self.get_decendants(lastOnly=True, psudojet_idx=row)
                decendents_idx = [self.idx_from_ID(d) for d in decendents]
                detas = [self._floats[d][self.eta_col] for d in decendents_idx]
                dphis = [self._floats[d][self.phi_col] for d in decendents_idx]
                des = [self._floats[d][self.energy_col] for d in decendents_idx]
                dpts = [1/self._floats[d][self.pt_col]**2 for d in decendents_idx]
                plt.scatter(detas, dphis, dpts, marker='D')
                print(f"Added jet of {len(decendents)} tracks, {self.currently_avalible} psudojets unfinished")
                plt.pause(0.05)
                input("Press enter for next psudojet")
                self._remove_psudojet(row)
            else:
                self._merge_psudojets(row, column, self._distances[row, column])
        plt.show()

    def idx_from_ID(self, psudojetID):
        ids = [p[self.psudojet_id_col] for p in self._ints]
        psudojet_idx = next((idx for idx, psu_id in enumerate(ids)
                             if psu_id == psudojetID),
                       None)
        if psudojet_idx is not None:
            return psudojet_idx
        raise ValueError(f"No psudojet with ID {psudojetID}")

    def get_decendants(self, lastOnly=True, psudojetID=None, psudojet_idx=None):
        if psudojetID is None and psudojet_idx is None:
            raise TypeError("Need to specify a psudojet")
        elif psudojet_idx is None:
            psudojet_idx = self.idx_from_ID(psudojetID)
        elif psudojetID is None:
            psudojetID = self._ints[psudojet_idx][self.psudojet_id_col]
        decendents = []
        if not lastOnly:
            decendents.append(psudojetID)
        # make local variables for speed
        daughter1_col = self.daughter1_col
        daughter2_col = self.daughter2_col
        # bu this point we have the first psudojet
        if (self._ints[psudojet_idx][daughter1_col] < 0
            and self._ints[psudojet_idx][daughter2_col] < 0):
            # just the one
            return [psudojetID]
        to_check = []
        d1 = self._ints[psudojet_idx][daughter1_col]
        d2 = self._ints[psudojet_idx][daughter2_col]
        if d1 >= 0:
            to_check.append(d1)
        if d2 >= 0:
            to_check.append(d2)
        while len(to_check) > 0:
            psudojetID = to_check.pop()
            psudojet_idx = self.idx_from_ID(psudojetID)
            if ((self._ints[psudojet_idx][daughter1_col] < 0
                 and self._ints[psudojet_idx][daughter2_col] < 0)
                or not lastOnly):
                decendents.append(psudojetID)
            d1 = self._ints[psudojet_idx][daughter1_col]
            d2 = self._ints[psudojet_idx][daughter2_col]
            if d1 >= 0:
                to_check.append(d1)
            if d2 >= 0:
                to_check.append(d2)
        return decendents
            

    def _combine(self, psudojet_index1, psudojet_index2, distance):
        new_id = max([ints[self.psudojet_id_col] for ints in self._ints]) + 1
        self._ints[psudojet_index1][self.mother_col] = new_id
        self._ints[psudojet_index2][self.mother_col] = new_id
        rank = max(self._ints[psudojet_index1][self.rank_col],
                   self._ints[psudojet_index2][self.rank_col]) + 1
        # psudojet_id global_obs_id mother daughter1 daughter2 rank
        ints = [new_id, -1, -1,
                self._ints[psudojet_index1][self.psudojet_id_col],
                self._ints[psudojet_index2][self.psudojet_id_col],
                rank]
        # pt eta phi energy join_distance
        pt1 = self._floats[psudojet_index1][self.pt_col]
        pt2 = self._floats[psudojet_index2][self.pt_col]
        e1 = self._floats[psudojet_index1][self.energy_col]
        e2 = self._floats[psudojet_index2][self.energy_col]
        eta1 = self._floats[psudojet_index1][self.eta_col]
        eta2 = self._floats[psudojet_index2][self.eta_col]
        phi1 = self._floats[psudojet_index1][self.phi_col]
        phi2 = self._floats[psudojet_index2][self.phi_col]
        floats = [pt1 + pt2,
                  (pt1*eta1 + pt2*eta2)/(pt1+pt2),
                  (pt1*phi1 + pt2*phi2)/(pt1+pt2),
                  e1 + e2,
                  distance]
        # one scheme (recombination schemes) look this up get fastjet manuel to see how it does for each algorithm
        return ints, floats


def main():
    pass


if __name__ == '__main__':
    pass #main()
