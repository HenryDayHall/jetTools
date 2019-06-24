import numpy as np
import subprocess
import os
from matplotlib import pyplot as plt
from ipdb import set_trace as st
from skhep import math as hepmath


class PsudoJets:
    def __init__(self, observables=None, deltaR=1., exponent_multiplyer=-1, **kwargs):
        self.deltaR = deltaR
        self.exponent_multiplyer = exponent_multiplyer
        self.exponent = 2 * exponent_multiplyer
        # make a table of ints and a table of floats
        # lists not arrays, becuase they will grow
        self._set_column_numbers()
        if 'ints' in kwargs:
            self._ints = kwargs['ints']
            self._floats = kwargs['floats']
            if isinstance(self._ints, np.ndarray):
                self._ints = self._ints.tolist()
                self._floats = self._floats.tolist()
        else:
            assert observables is not None, "Must give observables or floats and ints"
            self._ints = [[i, oid, -1, -1, -1, -1] for i, oid in enumerate(observables.global_obs_ids)]
            self._floats = np.hstack((observables.pts.reshape((-1, 1)),
                                      observables.raps.reshape((-1, 1)),
                                      observables.phis.reshape((-1, 1)),
                                      observables.es.reshape((-1, 1)),
                                      np.zeros((len(observables), 1)))).tolist()
        # keep track of how many clusters don't yet have a mother
        self._calculate_currently_avalible()
        self._distances = None
        self._calculate_distances()
        # as we go note the root notes of the psudojets
        self.root_psudojetIDs = []

    def _set_column_numbers(self):
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
        self.rap_col = 1
        self.phi_col = 2
        self.energy_col = 3
        self.join_distance_col = 4

    def _calculate_currently_avalible(self):
        # keep track of how many clusters don't yet have a mother
        self.currently_avalible = sum([p[self.mother_col]==-1 for p in self._ints])

    @property
    def pt(self):
        leaf_pts = [floats[self.pt_col] for floats, ints in zip(self._floats, self._ints)
                    if ints[self.daughter1_col] == -1 and
                       ints[self.daughter2_col] == -1]
        return sum(leaf_pts)

    @property
    def rap(self):
        leaf_etas = [floats[self.rap_col] for floats, ints in zip(self._floats, self._ints)
                     if ints[self.daughter1_col] == -1 and
                        ints[self.daughter2_col] == -1]
        return np.average(leaf_etas)

    @property
    def obs_raps(self):
        etas = [floats[self.rap_col] for floats, ints in zip(self._floats, self._ints)
                     if ints[self.obs_id_col] != -1]
        return etas

    @property
    def obs_phis(self):
        phis = [floats[self.phi_col] for floats, ints in zip(self._floats, self._ints)
                     if ints[self.obs_id_col] != -1]
        return phis

    @property
    def phi(self):
        leaf_phis = [floats[self.phi_col] for floats, ints in zip(self._floats, self._ints)
                     if ints[self.daughter1_col] == -1 and
                        ints[self.daughter2_col] == -1]
        return np.average(leaf_phis)

    @property
    def e(self):
        leaf_es = [floats[self.energy_col] for floats, ints in zip(self._floats, self._ints)
                   if ints[self.daughter1_col] == -1 and
                      ints[self.daughter2_col] == -1]
        return np.average(leaf_es)

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

    def write(self, dir_name, note="No note given."):
        note.replace('|', '!')  # because | will be used as a seperator
        technical_specs = f"|deltaR={self.deltaR}|exponent_multiplyer={self.exponent_multiplyer}|"
        # the directory should exist becuase we expect the observables to be written first
        assert os.path.exists(dir_name), "Write the observables first"
        iname_format = os.path.join(dir_name, "psuedojets_ints{}.csv")
        fname_format = os.path.join(dir_name, "psuedojets_floats{}.csv")
        name_number = 1
        # find a filename that is not taken (warning; race conditions here)
        while os.path.exists(iname_format.format(name_number)):
            name_number += 1
        ifile_name = iname_format.format(name_number)
        ffile_name = fname_format.format(name_number)
        # if the int file is free the float file should be too
        assert not os.path.exists(ffile_name), "How do you have unmatched int and float files?"
        int_columns = ["psudojet_id", "global_obs_id", "mother", "daughter1", "daughter2", "rank"]
        int_header = ' '.join([note, technical_specs, "Columns;", *int_columns])
        np.savetxt(ifile_name, self._ints,
                   header=int_header, fmt='%d')
        float_columns = ["pt", "rapidity", "phi", "energy", "join_distance"]
        float_header = ' '.join([note, technical_specs, "Columns;", *float_columns])
        np.savetxt(ffile_name, self._floats,
                   header=float_header)

    def _calculate_roots(self):
        # should only bee needed for reading from file
        assert self.currently_avalible == 0, "Assign mothers before you calculate roots"
        psudojet_ids = [ints[self.psudojet_id_col] for ints in self._ints]
        mother_ids = [ints[self.mother_col] for ints in self._ints]
        for mid, pid in zip(mother_ids, psudojet_ids):
            if (mid == -1 or
                mid not in psudojet_ids or
                mid == pid):
                self.root_psudojetIDs.append(pid)

    @classmethod
    def read(cls, dir_name, save_number=1, fastjet_format=False):
        if not fastjet_format:
            ifile_name = os.path.join(dir_name, f"psuedojets_ints{save_number}.csv")
            ffile_name = os.path.join(dir_name, f"psuedojets_floats{save_number}.csv")
            # first line will be the note, tech specs and columns
            with open(ifile_name, 'r') as ifile:
                header = ifile.readline()[1:]
            with open(ffile_name, 'r') as ffile:
                header2 = ffile.readline()[1:]
            header = header.split('|')
            header2 = header2.split('|')
            assert header[0] == header2[0], f"Notes should match for {ifile_name, ffile_name}"
            print(f"Note; {header[0]}")
            # now pull the tech specs
            deltaR = float(header[1].split('=')[1])
            exponent_multiplyer = float(header[2].split('=')[1])
            # the floats and ints
            ints = np.genfromtxt(ifile_name, skip_header=1, dtype=int)
            floats = np.genfromtxt(ffile_name, skip_header=1)
        else:  #  fastjet format
            ifile_name = os.path.join(dir_name, f"fastjet_ints.csv")
            ffile_name = os.path.join(dir_name, f"fastjet_doubles.csv")
            # first line will be the tech specs and columns
            with open(ifile_name, 'r') as ifile:
                header = ifile.readline()[1:]
            header = header.split(' ')
            deltaR = float(header[1].split('=')[1])
            algorithm_name = header[2]
            if algorithm_name == 'kt_algorithm':
                exponent_multiplyer = 1
            elif algorithm_name == 'cambridge_algorithm':
                exponent_multiplyer = 0
            elif algorithm_name == 'antikt_algorithm':
                exponent_multiplyer = -1
            else:
                raise ValueError(f"Algorithm {algorithm_name} not recognised")
            fast_ints = np.genfromtxt(ifile_name, skip_header=1, dtype=int)
            fast_floats = np.genfromtxt(ffile_name, skip_header=1)
            if len(fast_ints.shape) > 1:
                num_rows = fast_ints.shape[0]
                assert len(fast_ints) == len(fast_floats), f"len({ifile_name}) != len({ffile_name})"
            elif len(fast_ints) > 0:
                num_rows = 1
            else:
                num_rows = 0
            ints = np.full((num_rows, 6), -1, dtype=int)
            floats = np.full((num_rows, 5), -1, dtype=float)
            if len(fast_ints) > 0:
                ints[:, :5] = fast_ints
                floats[:, :4] = fast_floats
        new_psudojet = cls(ints=ints, floats=floats, deltaR=deltaR, exponent_multiplyer=exponent_multiplyer)
        new_psudojet.currently_avalible = 0
        new_psudojet._calculate_roots()
        return new_psudojet


    def split(self):
        assert self.currently_avalible == 0, "Need to assign_mothers before splitting"
        self.grouped_psudojets = [self.get_decendants(lastOnly=False, psudojetID=psudojetID)
                                  for psudojetID in self.root_psudojetIDs]
        self.JetList = []
        for root in self.root_psudojetIDs:
            group = self.get_decendants(lastOnly=False, psudojetID=root)
            group_idx = [self.idx_from_ID(ID) for ID in group]
            ints = [self._ints[i] for i in group_idx]
            floats = [self._floats[i] for i in group_idx]
            jet = PsudoJets(deltaR=self.deltaR, exponent_multiplyer=self.exponent_multiplyer,
                            ints=ints, floats=floats)
            jet.currently_avalible = 0
            jet.root_psudojetIDs = [root]
            self.JetList.append(jet)
        return self.JetList
    
    def _calculate_distances(self):
        # this is caluculating all the distances
        self._distances = np.full((self.currently_avalible, self.currently_avalible), np.inf)
        # for speed, make local variables
        pt_col  = self.pt_col 
        rap_col = self.rap_col
        phi_col = self.phi_col
        exponent = self.exponent
        deltaR = self.deltaR
        for row in range(self.currently_avalible):
            for column in range(self.currently_avalible):
                if column > row:
                    continue  # we only need a triangular matrix due to symmetry
                elif self._floats[row][pt_col] == 0:
                    distance = 0  # soft radation might as well be at 0 distance
                elif column == row:
                    distance = self._floats[row][pt_col]**exponent * deltaR
                else:
                    angular_diffrence = self._floats[row][phi_col] - self._floats[column][phi_col]
                    angular_distance = min(abs(angular_diffrence), abs(2*np.pi - angular_diffrence))
                    distance = min(self._floats[row][pt_col]**exponent, self._floats[column][pt_col]**exponent) *\
                               ((self._floats[row][rap_col] - self._floats[column][rap_col])**2 +
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
                           ((self._floats[row][self.rap_col] - self._floats[column][self.rap_col])**2 +
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
        raps = [p[self.rap_col] for p in self._floats]
        phis = [p[self.phi_col] for p in self._floats]
        es = [p[self.energy_col] for p in self._floats]
        pts = [1/p[self.pt_col]**2 for p in self._floats]
        plt.scatter(raps, phis, pts, marker='D', c='w')
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
                draps = [self._floats[d][self.rap_col] for d in decendents_idx]
                dphis = [self._floats[d][self.phi_col] for d in decendents_idx]
                des = [self._floats[d][self.energy_col] for d in decendents_idx]
                dpts = [1/self._floats[d][self.pt_col]**2 for d in decendents_idx]
                plt.scatter(draps, dphis, dpts, marker='D')
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
        rap1 = self._floats[psudojet_index1][self.rap_col]
        rap2 = self._floats[psudojet_index2][self.rap_col]
        phi1 = self._floats[psudojet_index1][self.phi_col]
        phi2 = self._floats[psudojet_index2][self.phi_col]
        # do not try to be clever with the angles.....
        # just do it all explicitly
        phi_shift = phi2 - phi1
        if phi_shift > np.pi: phi_shift -= 2.*np.pi
        elif phi_shift < -np.pi: phi_shift += 2.*np.pi
        mid_phi = phi1 + phi_shift/2.
        if mid_phi > np.pi: mid_phi -= 2.*np.pi
        elif mid_phi < -np.pi: mid_phi += 2.*np.pi
        floats = [pt1 + pt2,
                  (rap1+rap2)/2,
                  mid_phi,
                  e1 + e2,
                  distance]
        # floats = [pt1 + pt2,
        #           (pt1*eta1 + pt2*eta2)/(pt1+pt2),
        #           (pt1*phi1 + pt2*phi2)/(pt1+pt2),
        #           e1 + e2,
        #           distance]
        #temp_vector1 = hepmath.LorentzVector()
        #temp_vector1.setptetaphie(pt1, eta1, phi1, e1)
        #temp_vector2 = hepmath.LorentzVector()
        #temp_vector2.setptetaphie(pt2, eta2, phi2, e2)
        #comb = temp_vector1 + temp_vector2
        #floats = [comb.pt, comb.eta, comb.phi(), comb.e, distance]
        # one scheme (recombination schemes) look this up get fastjet manuel to see how it does for each algorithm
        return ints, floats

    def __len__(self):
        return len(self._ints)


def run_FastJet(dir_name, deltaR, exponent_multiplyer):
    if exponent_multiplyer == -1:
        # antikt algorithm
        algorithm_num = 1
    elif exponent_multiplyer == 0:
        algorithm_num = 2
    elif exponent_multiplyer == 1:
        algorithm_num = 0
    else:
        raise ValueError(f"exponent_multiplyer should be -1, 0 or 1, found {exponent_multiplyer}")
    program_name = "./applyFastJet"
    subprocess.run([program_name, dir_name, str(deltaR), str(algorithm_num)])
    fastjets = PsudoJets.read(dir_name, fastjet_format=True)
    return fastjets


def main():
    # colourmap
    colours = plt.get_cmap('gist_rainbow')
    import ReadSQL
    # get some psudo jets
    event, tracks, towers, observables = ReadSQL.main()
    psudojet = PsudoJets(observables)
    psudojet.assign_mothers()
    pjets = psudojet.split()
    # plot the psudojets
    psudo_colours = [colours(i) for i in np.linspace(0, 0.4, len(pjets))]
    for c, pjet in zip(psudo_colours, pjets):
        obs_idx = [ints[pjet.obs_id_col] != -1 for ints in pjet._ints]
        plt.scatter(np.array(pjet._floats)[obs_idx, pjet.eta_col],
                    np.array(pjet._floats)[obs_idx, pjet.phi_col],
                    c=[c], marker='v', s=30, alpha=0.6)
        #plt.scatter([psudojet.eta], [psudojet.phi], c='black', marker='o', s=10)
        #plt.scatter([psudojet.eta], [psudojet.phi], c='red', marker='o', s=9)
    plt.scatter([], [], c=[c], marker='v', s=30, alpha=0.6, label="PsudoJets")
    # get soem fast jets
    directory = "./test/"
    fastjets = PsudoJets.read(directory, fastjet_format=True)
    # plot the fastjet
    fjets = fastjets.split()
    psudo_colours = [colours(i) for i in np.linspace(0.6, 1., len(fjets))]
    for c, fjet in zip(psudo_colours, fjets):
        obs_idx = [ints[fjet.obs_id_col] != -1 for ints in fjet._ints]
        plt.scatter(np.array(fjet._floats)[obs_idx, fjet.eta_col],
                    np.array(fjet._floats)[obs_idx, fjet.phi_col],
                    c=[c], marker='^', s=25, alpha=0.6)
        #plt.scatter([jet.eta], [jet.phi], c=['black'], marker='o', s=10)
        #plt.scatter([jet.eta], [jet.phi], c=[colour], marker='o', s=9)
    plt.scatter([], [], c=[c], marker='^', s=25, alpha=0.6, label="FastJets")
    plt.legend()
    plt.title("Jets")
    plt.xlabel("eta")
    plt.ylabel("phi")
    plt.show()
    return pjets
    


if __name__ == '__main__':
    main()
