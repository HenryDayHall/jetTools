import numpy as np
import awkward
import subprocess
import os
import csv
from matplotlib import pyplot as plt
from ipdb import set_trace as st
from skhep import math as hepmath
from tree_tagger import Components


class PseudoJet:
    int_columns = ["Pseudojet_InputIdx",
                   "Pseudojet_Parent", "Pseudojet_Child1", "Pseudojet_Child2",
                   "Pseudojet_Rank"]
    float_columns = ["Pseudojet_PT",
                     "Pseudojet_Rapidity",
                     "Pseudojet_Phi",
                     "Pseudojet_Energy",
                     "Pseudojet_Px",
                     "Pseudojet_Py",
                     "Pseudojet_Pz",
                     "Pseudojet_JoinDistance"]
    def __init__(self, eventWise=None, deltaR=1., exponent_multiplyer=-1, **kwargs):
        self.deltaR = deltaR
        self.exponent_multiplyer = exponent_multiplyer
        self.exponent = 2 * exponent_multiplyer
        # jets can have a varient name
        # this allows multiple saves in the same file
        self.jet_name = kwargs.get('jet_name', 'Pseudojet')
        self.int_columns = [c.replace('Pseudojet', self.jet_name) for c in self.int_columns]
        self.float_columns = [c.replace('Pseudojet', self.jet_name) for c in self.float_columns]
        self.from_PseudoRapidity = kwargs.get("from_PseudoRapidity", False)
        if self.from_PseudoRapidity:
            idx = next(i for i, c in enumerate(self.float_columns) if c.endswith("_Rapidity"))
            self.float_columns[idx] = self.float_columns[idx].replace("_Rapidity", "_PseudoRapidity")
        # make a table of ints and a table of floats
        # lists not arrays, becuase they will grow
        self._set_column_numbers()
        if eventWise is not None:
            self.eventWise = eventWise
        else:
            assert 'eventWise_dir_name' in kwargs, "Must give eventWise or the instruction for it"
            eventWise_dir_name = kwargs["eventWise_dir_name"]
            eventWise_save_name = kwargs["eventWise_save_name"]
            eventWise_selected_index = kwargs["eventWise_selected_index"]
            self.eventWise = Components.EventWise(eventWise_dir_name, eventWise_save_name)
            self.eventWise.selected_index = eventWise_selected_index
        if 'ints' in kwargs:
            self._ints = kwargs['ints']
            self._floats = kwargs['floats']
            if isinstance(self._ints, np.ndarray):
                # these are forced into list format
                self._ints = self._ints.tolist()
                self._floats = self._floats.tolist()
            self.root_jetInputIdxs = kwargs.get('root_jetInputIdxs', [])
        else:
            assert "JetInputs_PT" in eventWise.columns, "eventWise must have JetInputs"
            assert isinstance(eventWise.selected_index, int), "selected index should be int"
            
            self.n_inputs = len(eventWise.JetInputs_PT)
            self._ints = [[i, -1, -1, -1, -1] for i in range(self.n_inputs)]
            if self.from_PseudoRapidity:
                rapidity_var = eventWise.JetInputs_PseudoRapidity
            else:
                rapidity_var = eventWise.JetInputs_Rapidity
            self._floats = np.hstack((eventWise.JetInputs_PT.reshape((-1, 1)),
                                      rapidity_var.reshape((-1, 1)),
                                      eventWise.JetInputs_Phi.reshape((-1, 1)),
                                      eventWise.JetInputs_Energy.reshape((-1, 1)),
                                      eventWise.JetInputs_Px.reshape((-1, 1)),
                                      eventWise.JetInputs_Py.reshape((-1, 1)),
                                      eventWise.JetInputs_Pz.reshape((-1, 1)),
                                      np.zeros((self.n_inputs, 1)))).tolist()
            # as we go note the root notes of the pseudojets
            self.root_jetInputIdxs = []
        # keep track of how many clusters don't yet have a parent
        self._calculate_currently_avalible()
        self._distances = None
        self._calculate_distances()

    def _set_column_numbers(self):
        prefix_len = len(self.jet_name) + 1
        # int columns
        self._int_contents = {}
        for i, name in enumerate(self.int_columns):
            attr_name = '_' + name[prefix_len:] + "_col"
            self._int_contents[name[prefix_len:]] = attr_name
            setattr(self, attr_name, i)
        # float columns
        self._float_contents = {}
        for i, name in enumerate(self.float_columns):
            if "PseudoRapidity" == name[prefix_len:]:
                name = "Rapidity"
            attr_name = '_' + name[prefix_len:] + "_col"
            self._float_contents[name[prefix_len:]] = attr_name
            setattr(self, attr_name, i)

    def __dir__(self):
        new_attrs = set(super().__dir__())
        return sorted(new_attrs)

    def _calculate_currently_avalible(self):
        # keep track of how many clusters don't yet have a parent
        self.currently_avalible = sum([p[self._Parent_col]==-1 for p in self._ints])

    def __getattr__(self, attr_name):
        """ the float columns form whole jet attrs"""
        # capitalise raises the case of the first letter
        attr_name = attr_name[0].upper() + attr_name[1:]
        if attr_name in self._float_contents:
            if len(self) == 0:  # if there are no pesudojets, there are no contents
                return np.nan
            # floats return a cumulative value
            col_num = getattr(self, self._float_contents[attr_name])
            cumulative = 0
            num_obs = 0
            for floats, ints in zip(self._floats, self._ints):
                  if (ints[self._Child1_col] == -1 and
                      ints[self._Child2_col] == -1):
                      cumulative += floats[col_num]
                      num_obs += 1
            if num_obs == 0:
                return 0.
            # some attrs should behave diferently
            if attr_name in ["Rapidity", "Pseudorapidity"]:
                print("Aaaaaaah")  # how did it end up like this
                raise AttributeError  # it was only a kiss, it was only a kiss
            if attr_name == 'Phi':  # make sure it's -pi to pi
                cumulative = Components.confine_angle(cumulative)
            return cumulative
        elif attr_name in self._int_contents:
            # ints return every value
            col_num = getattr(self, self._int_contents[attr_name])
            return np.array([ints[col_num] for ints in self._ints])
        raise AttributeError(f"{self.__class__.__name__} does not have {attr_name}")

    @property
    def P(self):
        if len(self) == 0:
            return np.nan
        return np.linalg.norm([self.Px, self.Py, self.Pz])

    @property
    def Theta(self):
        if len(self) == 0:
            return np.nan
        theta = Components.ptpz_to_theta(self.PT, self.Pz)
        return theta

    @property
    def Pseudorapidity(self):
        if len(self) == 0:
            return np.nan
        pseudorapidity = Components.theta_to_pseudorapidity(self.Theta)
        return pseudorapidity

    @property
    def Rapidity(self):
        if len(self) == 0:
            return np.nan
        rapidity = Components.ptpze_to_rapidity(self.PT, self.Pz, self.Energy)
        return rapidity

    @classmethod
    def write_event(cls, pseudojets, jet_name="Pseudojet", event_index=None, eventWise=None):
        """Save a handful of jets together """
        if eventWise is None:
            eventWise = pseudojets[0].eventWise
        if event_index is None:
            event_index = eventWise.selected_index
        save_columns = [ "_DeltaRs", "_ExponentMulti",
                        "_RootInputIdx"]
        save_columns = [jet_name + c for c in save_columns]
        int_columns = [c.replace('Pseudojet', jet_name) for c in cls.int_columns]
        float_columns = [c.replace('Pseudojet', jet_name) for c in cls.float_columns]
        save_columns += float_columns
        save_columns += int_columns
        eventWise.selected_index = None
        arrays = {name: list(getattr(eventWise, name, [])) for name in save_columns}
        # check there are enough event rows
        for name in save_columns:
            while len(arrays[name]) <= event_index:
                arrays[name].append([])
        for jet in pseudojets:
            assert jet.eventWise == pseudojets[0].eventWise
            arrays[jet_name + "_DeltaRs"][event_index].append(jet.deltaR)
            arrays[jet_name + "_ExponentMulti"][event_index].append(jet.exponent_multiplyer)
            arrays[jet_name + "_RootInputIdx"][event_index].append(awkward.fromiter(jet.root_jetInputIdxs))
            # if an array is deep it needs converting to an awkward array
            ints = awkward.fromiter(jet._ints)
            for col_num, name in enumerate(jet.int_columns):
                arrays[name][event_index].append(ints[:, col_num])
            floats = awkward.fromiter(jet._floats)
            for col_num, name in enumerate(jet.float_columns):
                arrays[name][event_index].append(floats[:, col_num])
        arrays = {name: awkward.fromiter(arrays[name]) for name in arrays}
        eventWise.append(arrays)

    @classmethod
    def multi_from_file(cls, file_name, jet_name="Pseudojet", batch_start=None, batch_end=None):
        """ read a handful of jets from file """
        int_columns = [c.replace('Pseudojet', jet_name) for c in cls.int_columns]
        float_columns = [c.replace('Pseudojet', jet_name) for c in cls.float_columns]
        # could write a version that just read one jet if needed
        eventWise = Components.EventWise.from_file(file_name)
        # check if its a pseudorapidty jet
        if jet_name + "_Rapidity" not in eventWise.columns:
            assert jet_name + "_PseudoRapidity" in eventWise.columns
            idx = float_columns.index(jet_name + "_Rapidity")
            float_columns[idx] = float_columns[idx].replace("_Rapidity", "_PseudoRapidity")
        save_name = eventWise.save_name
        dir_name = eventWise.dir_name
        avalible = len(getattr(eventWise, int_columns[0]))
        # decide on the start and stop points
        if batch_start is None:
            batch_start = 0
        if batch_end is None:
            batch_end = avalible
        elif batch_end > avalible:
            batch_end = avalible
        # get from the file
        jets = []
        for i in range(batch_start, batch_end):
            deltaR = getattr(eventWise, jet_name + "_DeltaRs")[i]
            exponent_multi = getattr(eventWise, jet_name + "_ExponentMulti")[i]
            roots = getattr(eventWise, jet_name + "_RootInputIdx")[i]
            # need to reassemble to ints and the floats
            int_values = []
            for name in int_columns:
                int_values.append(np.array(getattr(eventWise, name)[i]).reshape((-1, 1)))
            ints = np.hstack(int_values)
            float_values = []
            for name in float_columns:
                float_values.append(np.array(getattr(eventWise, name)[i]).reshape((-1, 1)))
            floats = np.hstack(float_values)
            new_jet = cls(deltaR=deltaR,
                          exponent_multiplier=exponent_multi,
                          eventWise_save_name=save_name,
                          eventWise_dir_name=dir_name,
                          eventWise_selected_index=i,
                          ints=ints.tolist(),
                          floats=floats.tolist(),
                          root_jetInputIdxs=roots)
            new_jet.currently_avalible = 0  # assumed since we are reading from file
            jets.append(new_jet)
        return jets

    def _calculate_roots(self):
        self.root_jetInputIdxs = []
        # should only bee needed for reading from file
        assert self.currently_avalible == 0, "Assign parents before you calculate roots"
        pseudojet_ids = self.InputIdx
        parent_ids = self.Parent
        for mid, pid in zip(parent_ids, pseudojet_ids):
            if (mid == -1 or
                mid not in pseudojet_ids or
                mid == pid):
                self.root_jetInputIdxs.append(pid)

    @classmethod
    def read_fastjet(cls, dir_name, selected_index=0, eventWise=None, jet_name="PseudoJet"):
        #  fastjet format
        assert eventWise is not None
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
        # the file of fast_ints contains
        fast_ints = np.genfromtxt(ifile_name, skip_header=1, dtype=int)
        assert len(fast_ints) > 0, "No ints found!"
        if len(fast_ints.shape) == 1:
            fast_ints = fast_ints.reshape((1, -1))
        # check that all the input idx have come through
        assert set(eventWise.JetInputs_SourceIdx).issubset(set(fast_ints[:, 1])), "Problem with inpu idx"
        next_free = max(fast_ints[:, 1]) + 1
        fast_idx_dict = {}
        for line_idx, i in fast_ints[:, :2]:
            if i == -1:
                fast_idx_dict[line_idx] = next_free
                next_free += 1
            else:
                fast_idx_dict[line_idx] = i
        fast_idx_dict[-1]=-1
        fast_ints = np.vectorize(fast_idx_dict.__getitem__)(fast_ints[:, [0, 2, 3, 4]])
        # check that the parent child relationship is reflexive
        for line in fast_ints:
            identifier = f"pseudojet inputIdx={line[0]} "
            if line[2] == -1:
                assert line[3] == -1, identifier + "has only one child"
            else:
                assert line[2] != line[3], identifier + " child1 and child1 are same"
                child1_line = fast_ints[fast_ints[:, 0] == line[2]][0]
                assert child1_line[1] == line[0], identifier + " first child dosn't acknowledge parent"
                child2_line = fast_ints[fast_ints[:, 0] == line[3]][0]
                assert child2_line[1] == line[0], identifier + " second child dosn't acknowledge parent"
            if line[1] != -1:
                assert line[0] != line[1], identifier + "is it's own mother"
                parent_line = fast_ints[fast_ints[:, 0] == line[1]][0]
                assert line[0] in parent_line[2:4], identifier + " parent doesn't acknowledge child"
        fast_floats = np.genfromtxt(ffile_name, skip_header=1)
        if len(fast_floats.shape) == 1:
            fast_floats = fast_floats.reshape((1, -1))
        if len(fast_ints.shape) > 1:
            num_rows = fast_ints.shape[0]
            assert len(fast_ints) == len(fast_floats), f"len({ifile_name}) != len({ffile_name})"
        elif len(fast_ints) > 0:
            num_rows = 1
        else:
            num_rows = 0
        ints = np.full((num_rows, len(cls.int_columns)), -1, dtype=int)
        floats = np.full((num_rows, len(cls.float_columns)), -1, dtype=float)
        if len(fast_ints) > 0:
            ints[:, :4] = fast_ints
            floats[:, :7] = fast_floats
        new_pseudojet = cls(ints=ints, floats=floats,
                            eventWise=eventWise,
                            deltaR=deltaR,
                            exponent_multiplyer=exponent_multiplyer,
                            jet_name=jet_name)
        new_pseudojet.currently_avalible = 0
        new_pseudojet._calculate_roots()
        # make ranks
        rank = 0
        ints[ints[:, 2]==-1, 4] = rank
        this_rank = ints[ints[:, 2] == -1, 1]
        while len(this_rank) > 0:
            rank += 1
            next_rank = []
            for i in this_rank:
                ints[ints[:, 0] == i, 4] = rank
                parent = ints[ints[:, 0] == i, 1]
                if parent != -1 and parent not in next_rank:
                    next_rank.append(parent)
            this_rank = next_rank
        return new_pseudojet

    def split(self):
        assert self.currently_avalible == 0, "Need to assign_parents before splitting"
        if len(self) == 0:
            return []
        self.JetList = []
        # ensure the split has the same order every time
        self.root_jetInputIdxs = sorted(self.root_jetInputIdxs)
        for root in self.root_jetInputIdxs:
            group = self.get_decendants(lastOnly=False, jetInputIdx=root)
            group_idx = [self.idx_from_inpIdx(ID) for ID in group]
            ints = [self._ints[i] for i in group_idx]
            floats = [self._floats[i] for i in group_idx]
            jet = PseudoJet(deltaR=self.deltaR, exponent_multiplyer=self.exponent_multiplyer,
                            ints=ints, floats=floats,
                            jet_name=self.jet_name,
                            eventWise_selected_index=self.eventWise.selected_index,
                            eventWise_dir_name=self.eventWise.dir_name,
                            eventWise_save_name=self.eventWise.save_name)
            jet.currently_avalible = 0
            jet.root_jetInputIdxs = [root]
            self.JetList.append(jet)
        return self.JetList
    
    def _calculate_distances(self):
        # this is caluculating all the distances
        self._distances = np.full((self.currently_avalible, self.currently_avalible), np.inf)
        # for speed, make local variables
        pt_col  = self._PT_col 
        rap_col = self._Rapidity_col
        phi_col = self._Phi_col
        exponent = self.exponent
        deltaR2 = self.deltaR**2
        for row in range(self.currently_avalible):
            for column in range(self.currently_avalible):
                if column > row:
                    continue  # we only need a triangular matrix due to symmetry
                elif self._floats[row][pt_col] == 0:
                    distance = 0  # soft radation might as well be at 0 distance
                elif column == row:
                    distance = self._floats[row][pt_col]**exponent * deltaR2
                else:
                    angular_diffrence = abs(self._floats[row][phi_col] - self._floats[column][phi_col]) % (2*np.pi)
                    angular_distance = min(angular_diffrence, 2*np.pi - angular_diffrence)
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
                distance = self._floats[row][self._PT_col]**self.exponent * self.deltaR**2
            else:
                angular_diffrence = abs(self._floats[row][self._Phi_col] - self._floats[column][self._Phi_col]) % (2*np.pi)
                angular_distance = min(angular_diffrence, 2*np.pi - angular_diffrence)
                distance = min(self._floats[row][self._PT_col]**self.exponent, self._floats[column][self._PT_col]**self.exponent) *\
                           ((self._floats[row][self._Rapidity_col] - self._floats[column][self._Rapidity_col])**2 +
                           (angular_distance)**2)
            self._distances[row, column] = distance

    def _merge_pseudojets(self, pseudojet_index1, pseudojet_index2, distance):
        new_pseudojet_ints, new_pseudojet_floats = self._combine(pseudojet_index1, pseudojet_index2, distance)
        # move the first pseudojet to the back without replacement
        pseudojet1_ints = self._ints.pop(pseudojet_index1)
        pseudojet1_floats = self._floats.pop(pseudojet_index1)
        self._ints.append(pseudojet1_ints)
        self._floats.append(pseudojet1_floats)
        # move the second pseudojet to the back but replace it with the new pseudojet
        pseudojet2_ints = self._ints[pseudojet_index2]
        pseudojet2_floats = self._floats[pseudojet_index2]
        self._ints.append(pseudojet2_ints)
        self._floats.append(pseudojet2_floats)
        self._ints[pseudojet_index2] = new_pseudojet_ints
        self._floats[pseudojet_index2] = new_pseudojet_floats
        # one less pseudojet avalible
        self.currently_avalible -= 1
        
        # delete the first row and column of the merge
        self._distances = np.delete(self._distances, (pseudojet_index1), axis=0)
        self._distances = np.delete(self._distances, (pseudojet_index1), axis=1)

        # now recalculate for the new pseudojet
        self._recalculate_one(pseudojet_index2)

    def _remove_pseudojet(self, pseudojet_index):
        # move the first pseudojet to the back without replacement
        pseudojet_ints = self._ints.pop(pseudojet_index)
        pseudojet_floats = self._floats.pop(pseudojet_index)
        self._ints.append(pseudojet_ints)
        self._floats.append(pseudojet_floats)
        self.root_jetInputIdxs.append(pseudojet_ints[self._InputIdx_col])
        # delete the row and column
        self._distances = np.delete(self._distances, (pseudojet_index), axis=0)
        self._distances = np.delete(self._distances, (pseudojet_index), axis=1)
        # one less pseudojet avalible
        self.currently_avalible -= 1
        

    def assign_parents(self):
        while self.currently_avalible > 0:
            # now find the smallest distance
            row, column = np.unravel_index(np.argmin(self._distances), self._distances.shape)
            if row == column:
                self._remove_pseudojet(row)
            else:
                self._merge_pseudojets(row, column, self._distances[row, column])

    def plt_assign_parents(self):
        # dendogram < this should be
        plt.axis([-5, 5, -np.pi-0.5, np.pi+0.5])
        inv_pts = [1/p[self._PT_col]**2 for p in self._floats]
        plt.scatter(self.Rapidity, self.Phi, inv_pts, c='w')
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.ylabel(r"$\phi$ - barrel angle")
        if self.from_PseudoRapidity:
            plt.xlabel(r"$\eta$ - pseudo rapidity")
        else:
            plt.xlabel(r"Rapidity")
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
        input("Press enter to start pseudojeting")
        while self.currently_avalible > 0:
            # now find the smallest distance
            row, column = np.unravel_index(np.argmin(self._distances), self._distances.shape)
            if row == column:
                decendents = self.get_decendants(lastOnly=True, pseudojet_idx=row)
                decendents_idx = [self.idx_from_inpIdx(d) for d in decendents]
                draps = [self._floats[d][self._Rapidity_col] for d in decendents_idx]
                dphis = [self._floats[d][self._Phi_col] for d in decendents_idx]
                des = [self._floats[d][self._Energy_col] for d in decendents_idx]
                dpts = [1/self._floats[d][self._PT_col]**2 for d in decendents_idx]  # WHY??
                plt.scatter(draps, dphis, dpts, marker='D')
                print(f"Added jet of {len(decendents)} tracks, {self.currently_avalible} pseudojets unfinished")
                plt.pause(0.05)
                input("Press enter for next pseudojet")
                self._remove_pseudojet(row)
            else:
                self._merge_pseudojets(row, column, self._distances[row, column])
        plt.show()

    def idx_from_inpIdx(self, jetInputIdx):
        ids = [p[self._InputIdx_col] for p in self._ints]
        pseudojet_idx = next((idx for idx, inp_idx in enumerate(ids)
                              if inp_idx == jetInputIdx),
                             None)
        if pseudojet_idx is not None:
            return pseudojet_idx
        raise ValueError(f"No pseudojet with ID {jetInputIdx}")

    def get_decendants(self, lastOnly=True, jetInputIdx=None, pseudojet_idx=None):
        if jetInputIdx is None and pseudojet_idx is None:
            raise TypeError("Need to specify a pseudojet")
        elif pseudojet_idx is None:
            pseudojet_idx = self.idx_from_inpIdx(jetInputIdx)
        elif jetInputIdx is None:
            jetInputIdx = self._ints[pseudojet_idx][self._InputIdx_col]
        decendents = []
        if not lastOnly:
            decendents.append(jetInputIdx)
        # make local variables for speed
        local_obs = self.local_obs_idx()
        child1_col = self._Child1_col
        child2_col = self._Child2_col
        # bu this point we have the first pseudojet
        if pseudojet_idx in local_obs:
            # just the one
            return [jetInputIdx]
        to_check = []
        ignore = []
        d1 = self._ints[pseudojet_idx][child1_col]
        d2 = self._ints[pseudojet_idx][child2_col]
        if d1 >= 0:
            to_check.append(d1)
        if d2 >= 0:
            to_check.append(d2)
        while len(to_check) > 0:
            jetInputIdx = to_check.pop()
            pseudojet_idx = self.idx_from_inpIdx(jetInputIdx)
            if (pseudojet_idx in local_obs or not lastOnly):
                decendents.append(jetInputIdx)
            else:
                ignore.append(jetInputIdx)
            d1 = self._ints[pseudojet_idx][child1_col]
            d2 = self._ints[pseudojet_idx][child2_col]
            if d1 >= 0 and d1 not in (decendents + ignore):
                to_check.append(d1)
            if d2 >= 0 and d2 not in (decendents + ignore):
                to_check.append(d2)
        return decendents

    def local_obs_idx(self):
        idx_are_obs = [i for i in range(len(self)) if
                       (self._ints[i][self._Child1_col] < 0 and
                       self._ints[i][self._Child2_col] < 0)]
        return idx_are_obs

    def _combine(self, pseudojet_index1, pseudojet_index2, distance):
        new_id = max([ints[self._InputIdx_col] for ints in self._ints]) + 1
        self._ints[pseudojet_index1][self._Parent_col] = new_id
        self._ints[pseudojet_index2][self._Parent_col] = new_id
        rank = max(self._ints[pseudojet_index1][self._Rank_col],
                   self._ints[pseudojet_index2][self._Rank_col]) + 1
        # inputidx, parent, child1, child2 rank
        # child1 shoul
        ints = [new_id,
                -1,
                self._ints[pseudojet_index1][self._InputIdx_col],
                self._ints[pseudojet_index2][self._InputIdx_col],
                rank]
        # PT px py pz eta phi energy join_distance
        # it's easier conceptually to calculate pt, phi and rapidity afresh than derive them
        # from the exisiting pt, phis and rapidity
        floats = [f1 + f2 for f1, f2 in
                  zip(self._floats[pseudojet_index1],
                      self._floats[pseudojet_index2])]
        px = floats[self._Px_col]
        py = floats[self._Py_col]
        pz = floats[self._Pz_col]
        energy = floats[self._Energy_col]
        phi, pt = Components.pxpy_to_phipt(px, py)
        floats[self._PT_col] = pt
        floats[self._Phi_col] = phi
        if self.from_PseudoRapidity:
            theta = Components.ptpz_to_theta(pt, pz)
            floats[self._Rapidity_col] = Components.theta_to_pseudorapidity(theta)
        else:
            floats[self._Rapidity_col] = Components.ptpze_to_rapidity(pt, pz, energy)
        # fix the distance
        floats[self._JoinDistance_col] = distance
        return ints, floats

    def __len__(self):
        return len(self._ints)

    def __eq__(self, other):
        if len(self) != len(other):
            return False 
        ints_eq = self._ints == other._ints
        floats_eq = np.allclose(self._floats, other._floats)
        return ints_eq and floats_eq


def filter_obs(eventWise, existing_idx_selection):
    assert eventWise.selected_index is not None
    has_track = eventWise.Particle_Track[existing_idx_selection] >= 0
    has_tower = eventWise.Particle_Tower[existing_idx_selection] >= 0
    observable = np.logical_or(has_track, has_tower)
    new_selection = existing_idx_selection[observable]
    return new_selection


def filter_ends(eventWise, existing_idx_selection):
    assert eventWise.selected_index is not None
    is_end = [len(c) == 0 for c in 
              eventWise.Children[existing_idx_selection]]
    new_selection = existing_idx_selection[is_end]
    return new_selection


def filter_pt_eta(eventWise, existing_idx_selection, min_pt=5., max_eta=2.5):
    assert eventWise.selected_index is not None
    # filter PT
    sufficient_pt = eventWise.PT[existing_idx_selection] > min_pt
    updated_selection = existing_idx_selection[sufficient_pt]
    zero_pseudorapidity = eventWise.Pz[updated_selection] == 0
    tan_theta = eventWise.PT[updated_selection]/eventWise.Pz[updated_selection]
    pseudorapidity_choice = np.abs(np.log(np.abs(tan_theta)/2)) < max_eta
    pseudorapidity_choice[zero_pseudorapidity] = 0
    updated_selection = updated_selection[pseudorapidity_choice]
    return updated_selection


def create_jetInputs(eventWise, filter_functions=[filter_obs, filter_pt_eta], batch_length=1000):
    # decide on run range
    eventWise.selected_index = None
    n_events = len(eventWise.Energy)
    start_point = len(getattr(eventWise, "JetInputs_Energy", []))
    if start_point >= n_events:
        print("Finished")
        return True
    end_point = min(n_events, start_point+batch_length)
    print(f" Will stop at {100*end_point/n_events}%")
    # sort out olumn names
    sources = ["PT", "Rapidity", "Phi", "Energy", "Px", "Py", "Pz"]
    for s in sources:
        if not hasattr(eventWise, s):
            print(f"EventWise lacks {s}")
            sources.remove(s)
    columns = ["JetInputs_" + c for c in sources]
    columns.append("JetInputs_SourceIdx")
    # the source column gives indices in the origin
    # construct the observable filter in advance
    contents = {"JetInputs_SourceIdx": list(getattr(eventWise, "JetInputs_SourceIdx", []))}
    for name in columns:
        contents[name] = list(getattr(eventWise, name, []))
    mask = []
    for event_n in range(start_point, end_point):
        if event_n % 100 == 0:
            print(f"{100*event_n/n_events}%", end='\r')
        eventWise.selected_index = event_n
        idx_selection = np.arange(len(eventWise.PT))
        for filter_func in filter_functions:
            idx_selection = filter_func(eventWise, idx_selection)
        contents["JetInputs_SourceIdx"].append(idx_selection)
        mask_here = np.full_like(eventWise.PT, False, dtype=bool)
        mask_here[idx_selection] = True
        mask.append(awkward.fromiter(mask_here))
    mask = awkward.fromiter(mask)
    eventWise.selected_index = None
    try:
        for name, source_name in zip(columns, sources):
            contents[name] += list(getattr(eventWise, source_name)[start_point:end_point][mask])
        contents = {k:awkward.fromiter(v) for k, v in contents.items()}
        eventWise.append(contents)
    except Exception as e:
        return contents, mask, columns, sources, e


def produce_summary(eventWise, event):
    eventWise.selected_index = event
    summary = np.vstack((eventWise.JetInputs_SourceIdx,
                         eventWise.JetInputs_Px,
                         eventWise.JetInputs_Py,
                         eventWise.JetInputs_Pz,
                         eventWise.JetInputs_Energy)).T
    summary = summary.astype(str)
    header = f"# summary file for {eventWise}\n"
    file_name = os.path.join(eventWise.dir_name, f"summary_observables.csv")
    with open(file_name, 'w') as summ_file:
        summ_file.write(header)
        writer = csv.writer(summ_file, delimiter=' ')
        writer.writerows(summary)


def run_FastJet(dir_name, eventWise, deltaR, exponent_multiplyer, jet_name="FastJet", capture_out=False):
    if exponent_multiplyer == -1:
        # antikt algorithm
        algorithm_num = 1
    elif exponent_multiplyer == 0:
        algorithm_num = 2
    elif exponent_multiplyer == 1:
        algorithm_num = 0
    else:
        raise ValueError(f"exponent_multiplyer should be -1, 0 or 1, found {exponent_multiplyer}")
    program_name = "./tree_tagger/applyFastJet"
    if capture_out:
        out = subprocess.run([program_name, dir_name, str(deltaR), str(algorithm_num)],
                             stdout=subprocess.PIPE)
        out = out.stdout.decode("utf-8")
        fastjets = PseudoJet.read_fastjet(dir_name)
        return fastjets, out
    subprocess.run([program_name, dir_name, str(deltaR), str(algorithm_num)])
    fastjets = PseudoJet.read_fastjet(dir_name, eventWise=eventWise, jet_name=jet_name)
    return fastjets


def fastjet_multiapply(eventWise, deltaR, exponent_multiplyer, jet_name=None, batch_length=100):
    if jet_name is None:
        jet_name = "FastJet"
    eventWise.selected_index = None
    dir_name = eventWise.dir_name
    n_events = len(eventWise.JetInputs_Energy)
    start_point = len(getattr(eventWise, jet_name+"_Energy", []))
    if start_point >= n_events:
        print("Finished")
        return True
    end_point = min(n_events, start_point+batch_length)
    print(f" Will stop at {100*end_point/n_events}%")
    for event_n in range(start_point, end_point):
        if event_n % 10 == 0:
            print(f"{100*event_n/n_events}%", end='\r')
        eventWise.selected_index = event_n
        if len(eventWise.JetInputs_PT) == 0:
            continue  # there are no observables
        produce_summary(eventWise, event_n)
        fastjets = run_FastJet(eventWise, dir_name, deltaR, exponent_multiplyer, jet_name=jet_name)
        fastjets = fastjets.split()
        try:
            os.remove(os.path.join(dir_name, 'summary_observables.csv'))
            PseudoJet.write_event(fastjets, jet_name=jet_name, event_num=event_n, eventWise=eventWise)
        except Exception as e:
            print(e)        
            return fastjets
    return False


def homejet_multiappy(eventWise, deltaR=0.4, exponentMulti=0, jet_name=None, batch_length=100):
    if jet_name is None:
        jet_name = "HomeJet"
    n_events = len(eventWise.JetInputs_Energy)
    start_point = len(getattr(eventWise, jet_name+"_Energy", []))
    if start_point >= n_events:
        print("Finished")
        return True
    end_point = min(n_events, start_point+batch_length)
    print(f" Will stop at {100*end_point/n_events}%")
    for event_n in range(start_point, end_point):
        if event_n % 10 == 0:
            print(f"{100*event_n/n_events}%", end='\r')
        eventWise.selected_index = event_n
        if len(eventWise.JetInputs_PT) == 0:
            continue  # there are no observables
        pseudojet = PseudoJet(eventWise, deltaR=deltaR,
                              exponent_multiplyer=exponentMulti,
                              jet_name=jet_name)
        pseudojet.assign_parents()
        jets = pseudojet.split()
        try:
            PseudoJet.write_event(jets, jet_name=jet_name,
                                  event_num=event_n, eventWise=eventWise)
        except:
            return jets, jet_name
    return False


def main():
    ax = plt.gca()
    # colourmap
    colours = plt.get_cmap('gist_rainbow')
    eventWise = Components.EventWise.from_file("/home/henry/lazy/dataset2/h1bBatch2_particles.awkd")
    filter_funcs = [filter_obs, filter_pt_eta]
    if "JetInputs_Energy" not in eventWise.columns:
        create_jetInputs(eventWise, filter_funcs)
    eventWise.selected_index = 0
    deltaR = 0.4
    pseudojet = PseudoJet(eventWise, deltaR, 0., jet_name="HomejetR1KT")
    pseudojet.assign_parents()
    pjets = pseudojet.split()
    # plot the pseudojets
    pseudo_colours = [colours(i) for i in np.linspace(0, 0.4, len(pjets))]
    for c, pjet in zip(pseudo_colours, pjets):
        obs_idx = pjet.local_obs_idx()
        plt.scatter(np.array(pjet._floats)[obs_idx, pjet._Rapidity_col],
                    np.array(pjet._floats)[obs_idx, pjet._Phi_col],
                    c=[c], marker='v', s=30, alpha=0.6)
        plt.scatter([pjet.Rapidity], [pjet.Phi], c='black', marker='o', s=10)
        plt.scatter([pjet.Rapidity], [pjet.Phi], c=[c], marker='o', s=9)
        circle = plt.Circle((pjet.Rapidity, pjet.Phi), radius=deltaR, edgecolor=(0,0,0,0.2), fill=False)
        ax.add_artist(circle)
    plt.scatter([], [], c=[c], marker='v', s=30, alpha=0.6, label="PseudoJets")
    # get soem fast jets
    produce_summary(eventWise, 0)
    fastjets = run_FastJet(eventWise.dir_name, deltaR, 0, eventWise)
    # plot the fastjet
    fjets = fastjets.split()
    pseudo_colours = [colours(i) for i in np.linspace(0.6, 1., len(fjets))]
    for c, fjet in zip(pseudo_colours, fjets):
        obs_idx = fjet.local_obs_idx()
        plt.scatter(np.array(fjet._floats)[obs_idx, fjet._Rapidity_col],
                    np.array(fjet._floats)[obs_idx, fjet._Phi_col],
                    c=[c], marker='^', s=30, alpha=0.6)
        plt.scatter([fjet.Rapidity], [fjet.Phi], c='black', marker='o', s=10)
        plt.scatter([fjet.Rapidity], [fjet.Phi], c=[c], marker='o', s=9)
        circle = plt.Circle((pjet.Rapidity, pjet.Phi), radius=deltaR, edgecolor=(0,0,0,0.2), fill=False)
        ax.add_artist(circle)
    plt.scatter([], [], c=[c], marker='^', s=25, alpha=0.6, label="FastJets")
    plt.legend()
    plt.title("Jets")
    plt.xlabel("rapidity")
    plt.ylabel("phi")
    plt.show()
    return pjets


def plot_jet_spiders(ew, jet_name, event_num, colour=None, ax=None):
    if ax is None:
        ax = plt.gca()
    if colour is None:
        colour = tuple(np.random.rand(3))
    ew.selected_index = event_num
    child1 = getattr(ew, jet_name+"_Child1")
    energy = getattr(ew, jet_name+"_AveEnergy")
    rap= getattr(ew, jet_name+"_AveRapidity")
    phi = getattr(ew, jet_name+"_AvePhi")
    # mark the centers
    #ax.scatter(rap, phi, s=np.sqrt(energy), color=[colour], label=jet_name)
    # make lines to the inputs
    part_Energy = getattr(ew, jet_name+"_Energy")
    part_phi = getattr(ew, jet_name+"_Phi")
    part_rap = getattr(ew, jet_name+"_Rapidity")
    n_jets = len(energy)
    for jet_n in range(n_jets):
        if len(part_Energy) == 1:
            continue
        center_phi = phi[jet_n]
        center_rap = rap[jet_n]
        end_points = np.where([c==-1 for c in child1[jet_n]])[0]
        #for end_idx in end_points:
        #    ax.plot([center_rap, part_rap[jet_n][end_idx]], [center_phi, part_phi[jet_n][end_idx]],
        #            linewidth=np.sqrt(part_Energy[jet_n][end_idx]), alpha=0.5, color=colour)
        if jet_n == 0:
            plt.scatter(part_rap[jet_n][end_points], part_phi[jet_n][end_points], s=part_Energy[jet_n][end_points], c=[colour],label=jet_name)
        else:
            plt.scatter(part_rap[jet_n][end_points], part_phi[jet_n][end_points], s=part_Energy[jet_n][end_points], c=[colour])
    ax.set_xlabel("Rapidity")
    ax.set_ylabel("$\\phi$")
    ax.legend()
    ew.selected_index = None
    
if __name__ == '__main__':
    #main()
    pass
