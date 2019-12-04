import numpy as np
import scipy
import scipy.spatial
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
    def __init__(self, eventWise, selected_index=None, jet_name='PseudoJet', from_PseudoRapidity=False,
                 ints_floats=None, **kwargs):
        # jets can have a varient name
        # this allows multiple saves in the same file
        self.jet_name = jet_name
        self.jet_parameters = {}
        dict_jet_params = kwargs.get('dict_jet_params', {})
        for key in dict_jet_params:  # create the formatting of a eventWise column
            key = key.replace(' ', '')
            key = key[0].upper() + key[1:]
            self.jet_parameters[key] = dict_jet_params[key]
        self.int_columns = [c.replace('Pseudojet', self.jet_name) for c in self.int_columns]
        self.float_columns = [c.replace('Pseudojet', self.jet_name) for c in self.float_columns]
        self.from_PseudoRapidity = from_PseudoRapidity
        if self.from_PseudoRapidity:
            idx = next(i for i, c in enumerate(self.float_columns) if c.endswith("_Rapidity"))
            self.float_columns[idx] = self.float_columns[idx].replace("_Rapidity", "_PseudoRapidity")
        # make a table of ints and a table of floats
        # lists not arrays, becuase they will grow
        self._set_column_numbers()
        if isinstance(eventWise, str):
            assert selected_index is not None, "If loading eventWise form file must specify and index"
            self.eventWise = Components.EventWise.from_file(eventWise)
        else:
            self.eventWise = eventWise
        if selected_index is not None:
            self.eventWise.selected_index = selected_index
        assert self.eventWise.selected_index is not None, "Must specify an index (event number) for the eventWise"
        if ints_floats is not None:
            assert len(ints_floats) == 2
            self._ints = ints_floats[0]
            self._floats = ints_floats[1]
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
        if kwargs.get("assign", False):
            self.assign_parents()

    def _set_hyperparams(self, param_list, dict_jet_params, kwargs):
        if dict_jet_params is None:
            dict_jet_params = {}
        stripped_params = {name.split("_Param")[-1]:name for name in dict_jet_params}
        for name in param_list:
            if name in stripped_params:
                assert name not in kwargs
                setattr(self, name, dict_jet_params[stripped_params[name]])
            else:
                setattr(self, name, kwargs[name])
                dict_jet_params[name] = kwargs[name]
                del kwargs[name]
        kwargs['dict_jet_params'] = dict_jet_params

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
            # floats return the value of the root
            # or list if more than one root
            col_num = getattr(self, self._float_contents[attr_name])
            values = np.array([floats[col_num] for floats, ints in zip(self._floats, self._ints)
                               if ints[self._Parent_col] == -1])
            if attr_name == 'Phi':  # make sure it's -pi to pi
                values = Components.confine_angle(values)
            if len(values) == 0:
                return 0.
            if len(values) == 1:
                return values[0]
            return values
        elif attr_name in self._int_contents:
            # ints return every value
            col_num = getattr(self, self._int_contents[attr_name])
            return np.array([ints[col_num] for ints in self._ints])
        elif attr_name == "Rapidity":
            # if the jet was constructed with pseudorapidity we might still want to know the rapidity
            return Components.ptpze_to_rapidity(self.PT, self.Pz, self.Energy)
        elif attr_name =="Pseudorapidity":
            # vice verca
            return Components.theta_to_pseudorapidity(self.Theta)
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

    @classmethod
    def create_updated_dict(cls, pseudojets, jet_name, event_index, eventWise=None, arrays=None):
        """Make the dictionary to be appended to an eventWise for writing"""
        if arrays is None:
            save_columns = [jet_name + "_RootInputIdx"]
            save_columns += [jet_name + c for c in save_columns]
            save_columns += [jet_name + "_Param" + c for c in pseudojets[0].jet_parameters.keys()]
            int_columns = [c.replace('Pseudojet', jet_name) for c in cls.int_columns]
            float_columns = [c.replace('Pseudojet', jet_name) for c in cls.float_columns]
            save_columns += float_columns
            save_columns += int_columns
            eventWise.selected_index = None
            arrays = {name: list(getattr(eventWise, name, [])) for name in save_columns}
        # check there are enough event rows
        for name in arrays:
            while len(arrays[name]) <= event_index:
                arrays[name].append([])
        for jet in pseudojets:
            assert jet.eventWise == pseudojets[0].eventWise
            for key, item in jet.jet_parameters.items():
                # includes things like DeltaR and exponent multipliers
                name = jet_name + "_Param" + key
                arrays[name][event_index].append(item)
            arrays[jet_name + "_RootInputIdx"][event_index].append(awkward.fromiter(jet.root_jetInputIdxs))
            # if an array is deep it needs converting to an awkward array
            ints = awkward.fromiter(jet._ints)
            for col_num, name in enumerate(jet.int_columns):
                arrays[name][event_index].append(ints[:, col_num])
            floats = awkward.fromiter(jet._floats)
            for col_num, name in enumerate(jet.float_columns):
                arrays[name][event_index].append(floats[:, col_num])
        return arrays

    @classmethod
    def write_event(cls, pseudojets, jet_name="Pseudojet", event_index=None, eventWise=None):
        """Save a handful of jets together """
        if eventWise is None:
            eventWise = pseudojets[0].eventWise
        if event_index is None:
            event_index = eventWise.selected_index
        arrays = cls.create_updated_dict(pseudojets, jet_name, event_index, eventWise)
        arrays = {name: awkward.fromiter(arrays[name]) for name in arrays}
        eventWise.append(arrays)

    @classmethod
    def multi_from_file(cls, file_name, event_idx, jet_name="Pseudojet", batch_start=None, batch_end=None):
        """ read a handful of jets from file """
        int_columns = [c.replace('Pseudojet', jet_name) for c in cls.int_columns]
        float_columns = [c.replace('Pseudojet', jet_name) for c in cls.float_columns]
        # could write a version that just read one jet if needed
        eventWise = Components.EventWise.from_file(file_name)
        eventWise.selected_index = event_idx
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
            param_prefix = jet_name + "_Param"
            param_columns = [c for c in eventWise.columns if c.startswith(param_prefix)]
            param_dict = {name: getattr(eventWise, name) for name in param_columns}
            for key in param_dict:
                assert len(set(param_dict[key])) == 1
                param_dict[key] = param_dict[key][0]
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
            new_jet = cls(eventWise=file_name,
                          selected_index=i,
                          ints_floats=(ints.tolist(), floats.tolist()),
                          root_jetInputIdxs=roots,
                          dict_jet_params=param_dict)
            new_jet.currently_avalible = 0  # assumed since we are reading from file
            jets.append(new_jet)
        return jets

    def _calculate_roots(self):
        self.root_jetInputIdxs = []
        # should only bee needed for reading from file self.currently_avalible == 0, "Assign parents before you calculate roots"
        pseudojet_ids = self.InputIdx
        parent_ids = self.Parent
        for mid, pid in zip(parent_ids, pseudojet_ids):
            if (mid == -1 or
                mid not in pseudojet_ids or
                mid == pid):
                self.root_jetInputIdxs.append(pid)

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
            jet = type(self)(ints_floats=(ints, floats),
                             jet_name=self.jet_name,
                             selected_index=self.eventWise.selected_index,
                             eventWise=self.eventWise,
                             dict_jet_params=self.jet_parameters)
            jet.currently_avalible = 0
            jet.root_jetInputIdxs = [root]
            self.JetList.append(jet)
        return self.JetList
    
    def _calculate_distances(self):
        # this is caluculating all the distances
        raise NotImplementedError

    def _recalculate_one(self, remove_index, replace_index):
        raise NotImplementedError

    def _merge_pseudojets(self, pseudojet_index1, pseudojet_index2, distance):
        replace_index, remove_index = sorted([pseudojet_index1, pseudojet_index2])
        new_pseudojet_ints, new_pseudojet_floats = self._combine(remove_index, replace_index, distance)
        # move the first pseudojet to the back without replacement
        pseudojet1_ints = self._ints.pop(remove_index)
        pseudojet1_floats = self._floats.pop(remove_index)
        self._ints.append(pseudojet1_ints)
        self._floats.append(pseudojet1_floats)
        # move the second pseudojet to the back but replace it with the new pseudojet
        pseudojet2_ints = self._ints[replace_index]
        pseudojet2_floats = self._floats[replace_index]
        self._ints.append(pseudojet2_ints)
        self._floats.append(pseudojet2_floats)
        self._ints[replace_index] = new_pseudojet_ints
        self._floats[replace_index] = new_pseudojet_floats
        # one less pseudojet avalible
        self.currently_avalible -= 1
        # now recalculate for the new pseudojet
        self._recalculate_one(remove_index, replace_index)

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


class Traditional(PseudoJet):
    param_list = {'DeltaR': None, 'ExponentMultiplier': None}
    def __init__(self, eventWise=None, dict_jet_params=None, **kwargs):
        self._set_hyperparams(self.param_list, dict_jet_params, kwargs)
        self.exponent = 2 * self.ExponentMultiplier
        super().__init__(eventWise, **kwargs)

    def _calculate_distances(self):
        # this is caluculating all the distances
        self._distances = np.full((self.currently_avalible, self.currently_avalible), np.inf)
        # for speed, make local variables
        pt_col  = self._PT_col 
        rap_col = self._Rapidity_col
        phi_col = self._Phi_col
        exponent = self.exponent
        DeltaR2 = self.DeltaR**2
        for row in range(self.currently_avalible):
            for column in range(self.currently_avalible):
                if column > row:
                    continue  # we only need a triangular matrix due to symmetry
                elif self._floats[row][pt_col] == 0:
                    distance = 0  # soft radation might as well be at 0 distance
                elif column == row:
                    distance = self._floats[row][pt_col]**exponent * DeltaR2
                else:
                    angular_distance = Components.angular_distance(self._floats[row][phi_col], self._floats[column][phi_col])
                    distance = min(self._floats[row][pt_col]**exponent, self._floats[column][pt_col]**exponent) *\
                               ((self._floats[row][rap_col] - self._floats[column][rap_col])**2 +
                               (angular_distance)**2)
                self._distances[row, column] = distance

    def _recalculate_one(self, remove_index, replace_index):
        # delete the larger index keep the smaller index
        assert remove_index > replace_index
        # delete the first row and column of the merge
        self._distances = np.delete(self._distances, (remove_index), axis=0)
        self._distances = np.delete(self._distances, (remove_index), axis=1)

        # calculate new values into the second column
        for row in range(self.currently_avalible):
            column = replace_index
            if column > row:
                row, column = column, row  # keep the upper triangular form
            if column == row:
                distance = self._floats[row][self._PT_col]**self.exponent * self.DeltaR**2
            else:
                angular_diffrence = abs(self._floats[row][self._Phi_col] - self._floats[column][self._Phi_col]) % (2*np.pi)
                angular_distance = min(angular_diffrence, 2*np.pi - angular_diffrence)
                distance = min(self._floats[row][self._PT_col]**self.exponent, self._floats[column][self._PT_col]**self.exponent) *\
                           ((self._floats[row][self._Rapidity_col] - self._floats[column][self._Rapidity_col])**2 +
                           (angular_distance)**2)
            self._distances[row, column] = distance

    @classmethod
    def read_fastjet(cls, arg, eventWise, jet_name="FastJet", do_checks=False):
        #  fastjet format
        assert eventWise.selected_index is not None
        if isinstance(arg, str):
            ifile_name = os.path.join(arg, f"fastjet_ints.csv")
            ffile_name = os.path.join(arg, f"fastjet_doubles.csv")
            # while it would be nice to filter warnings here it's a high frequency bit of code
            # and I don't want a speed penalty here
            fast_ints = np.genfromtxt(ifile_name, skip_header=1, dtype=int)
            fast_floats = np.genfromtxt(ffile_name, skip_header=1)
            with open(ifile_name, 'r') as ifile:
                header = ifile.readline()[1:]
            with open(ffile_name, 'r') as ffile:
                fcolumns = ffile.readline()[1:].split()
        else:
            header = arg[0].decode()[1:]
            arrays = [[]]
            a_type = int
            for line in arg[1:]:
                line = line.decode().strip()
                if line[0] == '#':  # moves from the ints to the doubles
                    arrays.append([])
                    a_type = float
                    fcolumns = line[1:].split()
                else:
                    arrays[-1].append([a_type(x) for x in line.split()])
            assert len(arrays) == 2, f"Problem wiht input; \n{arg}"
            fast_ints = np.array(arrays[0], dtype=int)
            fast_floats = np.array(arrays[1], dtype=float)
        # first line will be the tech specs and columns
        header = header.split()
        DeltaR = float(header[0].split('=')[1])
        algorithm_name = header[1]
        if algorithm_name == 'kt_algorithm':
            ExponentMultiplier = 1
        elif algorithm_name == 'cambridge_algorithm':
            ExponentMultiplier = 0
        elif algorithm_name == 'antikt_algorithm':
            ExponentMultiplier = -1
        else:
            raise ValueError(f"Algorithm {algorithm_name} not recognised")
        # get the colums for the header
        icolumns = {name: i for i, name in enumerate(header[header.index("Columns;") + 1:])}
        # and from this get the columns
        # the file of fast_ints contains
        n_fastjet_int_cols = len(icolumns)
        if len(fast_ints.shape) == 1:
            fast_ints = fast_ints.reshape((-1, n_fastjet_int_cols))
        else:
            assert fast_ints.shape[1] == n_fastjet_int_cols
        # check that all the input idx have come through
        n_inputs = len(eventWise.JetInputs_SourceIdx)
        assert set(np.arange(n_inputs)).issubset(set(fast_ints[:, icolumns["InputIdx"]])), "Problem with inpu idx"
        next_free = np.max(fast_ints[:, icolumns["InputIdx"]], initial=-1) + 1
        fast_idx_dict = {}
        for line_idx, i in fast_ints[:, [icolumns["pseudojet_id"], icolumns["InputIdx"]]]:
            if i == -1:
                fast_idx_dict[line_idx] = next_free
                next_free += 1
            else:
                fast_idx_dict[line_idx] = i
        fast_idx_dict[-1]=-1
        fast_ints = np.vectorize(fast_idx_dict.__getitem__,
                                 otypes=[np.float])(fast_ints[:, [icolumns["pseudojet_id"],
                                                                  icolumns["parent_id"],
                                                                  icolumns["child1_id"],
                                                                  icolumns["child2_id"]]])
        # now the Inputidx is the first one and the pseudojet_id can be removed
        del icolumns["pseudojet_id"]
        icolumns = {name: i-1 for name, i in icolumns.items()}
        n_fastjet_float_cols = len(fcolumns)
        if do_checks:
            # check that the parent child relationship is reflexive
            for line in fast_ints:
                identifier = f"pseudojet inputIdx={line[0]} "
                if line[icolumns["child1_id"]] == -1:
                    assert line[icolumns["child2_id"]] == -1, identifier + "has only one child"
                else:
                    assert line[icolumns["child1_id"]] != line[icolumns["child2_id"]], identifier + " child1 and child2 are same"
                    child1_line = fast_ints[fast_ints[:, icolumns["InputIdx"]]
                                            == line[icolumns["child1_id"]]][0]
                    assert child1_line[1] == line[0], identifier + " first child dosn't acknowledge parent"
                    child2_line = fast_ints[fast_ints[:, icolumns["InputIdx"]]
                                            == line[icolumns["child2_id"]]][0]
                    assert child2_line[1] == line[0], identifier + " second child dosn't acknowledge parent"
                if line[1] != -1:
                    assert line[icolumns["InputIdx"]] != line[icolumns["parent_id"]], identifier + "is it's own mother"
                    parent_line = fast_ints[fast_ints[:, icolumns["InputIdx"]]
                                            == line[icolumns["parent_id"]]][0]
                    assert line[0] in parent_line[[icolumns["child1_id"],
                                                   icolumns["child2_id"]]], identifier + " parent doesn't acknowledge child"
            for fcol, expected in zip(fcolumns, PseudoJet.float_columns):
                assert expected.endswith(fcol)
            if len(fast_ints) == 0:
                assert len(fast_floats) == 0, "No ints found, but floats are present!"
                print("Warning, no values from fastjet.")
        if len(fast_floats.shape) == 1:
            fast_floats = fast_floats.reshape((-1, n_fastjet_float_cols))
        else:
            assert fast_floats.shape[1] == n_fastjet_float_cols
        if len(fast_ints.shape) > 1:
            num_rows = fast_ints.shape[0]
            assert len(fast_ints) == len(fast_floats), f"len({ifile_name}) != len({ffile_name})"
        elif len(fast_ints) > 0:
            num_rows = 1
        else:
            num_rows = 0
        ints = np.full((num_rows, len(cls.int_columns)), -1, dtype=int)
        floats = np.zeros((num_rows, len(cls.float_columns)), dtype=float)
        if len(fast_ints) > 0:
            ints[:, :4] = fast_ints
            floats[:, :7] = fast_floats
        # make ranks
        rank = -1
        rank_col = len(icolumns)
        ints[ints[:, icolumns["child1_id"]] == -1, rank_col] = rank
        # parents of the lowest rank is the next rank
        this_rank = set(ints[ints[:, icolumns["child1_id"]] == -1, icolumns["parent_id"]])
        this_rank.discard(-1)
        while len(this_rank) > 0:
            rank += 1
            next_rank = []
            for i in this_rank:
                ints[ints[:, icolumns["InputIdx"]] == i, rank_col] = rank
                parent = ints[ints[:, icolumns["InputIdx"]] == i, icolumns["parent_id"]]
                if parent != -1 and parent not in next_rank:
                    next_rank.append(parent)
            this_rank = next_rank
        # create the pseudojet
        new_pseudojet = cls(ints_floats=(ints, floats),
                            eventWise=eventWise,
                            DeltaR=DeltaR,
                            ExponentMultiplier=ExponentMultiplier,
                            jet_name=jet_name)
        new_pseudojet.currently_avalible = 0
        new_pseudojet._calculate_roots()
        return new_pseudojet


class Spectral(PseudoJet):
    # list the params with default values
    param_list = {'DeltaR': None, 'NumEigenvectors': np.inf, 
            'ExponentMultiplier': None, 'AffinityType': 'exponent',
            'AffinityCutoff': None, 'Laplacien': 'unnormalised'}
    def __init__(self, eventWise=None, dict_jet_params=None, **kwargs):
        self._set_hyperparams(self.param_list, dict_jet_params, kwargs)
        self.exponent = 2 * self.ExponentMultiplier
        self.merger_indices = [] # need to track this
        super().__init__(eventWise, **kwargs)

    def _calculate_distances(self):
        if self.currently_avalible < 2:
            self._distances = np.zeros(1).reshape((1,1))
            return np.zeros(self.currently_avalible).reshape((self.currently_avalible, self.currently_avalible))
        # to start with create a 'normal' distance measure
        # this can be based on any of the three algorithms
        physical_distances = np.zeros((self.currently_avalible, self.currently_avalible))
        # for speed, make local variables
        pt_col  = self._PT_col 
        rap_col = self._Rapidity_col
        phi_col = self._Phi_col
        exponent = self.exponent
        for row in range(self.currently_avalible):
            for column in range(self.currently_avalible):
                if column < row:
                    distance = physical_distances[column, row]  # the matrix is symmetric
                elif self._floats[row][pt_col] == 0:
                    distance = 0  # soft radation might as well be at 0 distance
                elif column == row:
                    # not used
                    continue
                else:
                    angular_distance = Components.angular_distance(self._floats[row][phi_col], self._floats[column][phi_col])
                    distance = min(self._floats[row][pt_col]**exponent, self._floats[column][pt_col]**exponent) *\
                               ((self._floats[row][rap_col] - self._floats[column][rap_col])**2 +
                               (angular_distance)**2)
                physical_distances[row, column] = distance
        np.fill_diagonal(physical_distances, 0)
        # now we are in posessio of a standard distance matrix for all points,
        # we can make an affinity calculation
        if self.AffinityCutoff is not None:
            cutoff_type = self.AffinityCutoff[0]
            cutoff_param = self.AffinityCutoff[1]
            if cutoff_type == 'knn':
                if self.AffinityType == 'exponent':
                    def calculate_affinity(distances):
                        affinity = np.exp(-(distances**0.5))
                        affinity[np.argsort(distances, axis=0) < cutoff_param] = 0
                elif self.AffinityType == 'linear':
                    def calculate_affinity(distances):
                        affinity = -distances**0.5
                        affinity[np.argsort(distances, axis=0) < cutoff_param] = 0
            elif cutoff_type == 'distance':
                if self.AffinityType == 'exponent':
                    def calculate_affinity(distances):
                        affinity = np.exp(-(distances**0.5))
                        affinity[distances > cutoff_param] = 0
                elif self.AffinityType == 'linear':
                    def calculate_affinity(distances):
                        affinity = -distances**0.5
                        affinity[distances > cutoff_param] = 0
            else:
                raise ValueError(f"cut off {cutoff_type} unknown")
        affinity = calculate_affinity(physical_distances)
        # this is make into a class fuction becuase it will b needed elsewhere
        self.calculate_affinity = calculate_affinity
        # a graph laplacien can be calculated
        np.fill_diagonal(affinity, 0)
        self.diagonal = np.diag(np.sum(affinity, axis=1))
        if self.Laplacien == 'unnormalised':
            laplacien = self.diagonal - affinity
        elif self.Laplacien == 'symmetric':
            laplacien = self.diagonal - affinity
            alt_diag = self.diagonal**(-0.5)
            laplacien = np.matmul(alt_diag, np.matmul(laplacien, alt_diag))
        # get the eigenvectors (we know the smallest will be identity)
        try:
            eigenvalues, eigenvectors = scipy.linalg.eigh(laplacien, eigvals=(0, self.NumEigenvectors+1))
            opt=True
        except ValueError:
            # sometimes there are fewer eigenvalues avalible
            # just take waht can be found
            eigenvalues, eigenvectors = scipy.linalg.eigh(laplacien)
            opt=False
        self.eigenvectors = eigenvectors  # make publically visible
        # these tests often fall short of tollarance, and they arn't really needed
        #np.testing.assert_allclose(0, eigenvalues[0], atol=0.001)
        #np.testing.assert_allclose(np.ones(self.currently_avalible), eigenvectors[:, 0]/eigenvectors[0, 0])
        # now treating the columns of this matrix as the new points get euclidien distances
        self._distances = scipy.spatial.distance.squareform(
                scipy.spatial.distance.pdist(eigenvectors[:, 1:]))
        # if the clustering is not going to stop at 1 we must put something in the diagonal
        np.fill_diagonal(self._distances, np.inf)

    def _recalculate_one(self, remove_index, replace_index):
        # delete the larger index keep the smaller index
        assert remove_index > replace_index
        # delete the first row and column of the merge
        self._distances = np.delete(self._distances, (remove_index), axis=0)
        self._distances = np.delete(self._distances, (remove_index), axis=1)
        # calculate the physical distance of the new point from all original points
        new_distances = np.zeros(self.diagonal.shape[0])
        for column in range(self.currently_avalible):
            row = replace_index
            if column == row:
                distance = 0.
            else:
                angular_diffrence = abs(self._floats[row][self._Phi_col] - self._floats[column][self._Phi_col]) % (2*np.pi)
                angular_distance = min(angular_diffrence, 2*np.pi - angular_diffrence)
                distance = min(self._floats[row][self._PT_col]**self.exponent, self._floats[column][self._PT_col]**self.exponent) *\
                           ((self._floats[row][self._Rapidity_col] - self._floats[column][self._Rapidity_col])**2 +
                           (angular_distance)**2)
            new_distances[column] = distance
        # from this get a new line of the laplacien
        new_affinity = self.calculate_affinity(new_distances)
        if self.Laplacien == 'unnormalised':
            new_laplacien = new_affinity
            new_laplacien[replace_index] = self.diagonal[replace_index, replace_index]
        elif self.Laplacien == 'symmetric':
            new_laplacien = new_affinity
            new_laplacien[replace_index] = self.diagonal[replace_index, replace_index]
            alt_diag = np.diag(self.diagonal)**(-0.5)
            new_laplacien = alt_diag * (new_laplacien* alt_diag[replace_index])
        # and make its position in vector space
        new_position = np.dot(self.eigenvectors, new_laplacien)
        self._distances[:, replace_index] = new_position
        self._distances[replace_index] = new_position


def filter_obs(eventWise, existing_idx_selection):
    assert eventWise.selected_index is not None
    has_track = eventWise.Particle_Track[existing_idx_selection.tolist()] >= 0
    has_tower = eventWise.Particle_Tower[existing_idx_selection.tolist()] >= 0
    observable = np.logical_or(has_track, has_tower)
    new_selection = existing_idx_selection[observable]
    return new_selection


def filter_ends(eventWise, existing_idx_selection):
    assert eventWise.selected_index is not None
    is_end = [len(c) == 0 for c in 
              eventWise.Children[existing_idx_selection.tolist()]]
    new_selection = existing_idx_selection[is_end]
    return new_selection


def filter_pt_eta(eventWise, existing_idx_selection, min_pt=.5, max_eta=2.5):
    assert eventWise.selected_index is not None
    # filter PT
    sufficient_pt = eventWise.PT[existing_idx_selection.tolist()] > min_pt
    updated_selection = existing_idx_selection[sufficient_pt]
    if "Pseudorapidity" in eventWise.columns:
        pseudorapidity_here = eventWise.Pseudorapidity[updated_selection.tolist()]
    else:
        theta_here = Components.ptpz_to_theta(eventWise.PT[updated_selection.tolist()], eventWise.Pz[updated_selection.tolist()])
        pseudorapidity_here = Components.theta_to_pseudorapidity(theta_here)
    pseudorapidity_choice = np.abs(pseudorapidity_here) < max_eta
    updated_selection = updated_selection[pseudorapidity_choice.tolist()]
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


def produce_summary(eventWise, to_file=True):
    assert eventWise.selected_index is not None
    n_inputs = len(eventWise.JetInputs_SourceIdx)
    summary = np.vstack((np.arange(n_inputs),
                         eventWise.JetInputs_Px,
                         eventWise.JetInputs_Py,
                         eventWise.JetInputs_Pz,
                         eventWise.JetInputs_Energy)).T
    summary = summary.astype(str)
    if to_file:
        header = f"# summary file for {eventWise}, event {eventWise.selected_index}\n"
        file_name = os.path.join(eventWise.dir_name, f"summary_observables.csv")
        with open(file_name, 'w') as summ_file:
            summ_file.write(header)
            writer = csv.writer(summ_file, delimiter=' ')
            writer.writerows(summary)
    else:
        rows = [' '.join(row) for row in summary]
        return '\n'.join(rows).encode()


def run_FastJet(eventWise, DeltaR, ExponentMultiplier, jet_name="FastJet", use_pipe=True):
    assert eventWise.selected_index is not None
    if ExponentMultiplier == -1:
        # antikt algorithm
        algorithm_num = 1
    elif ExponentMultiplier == 0:
        algorithm_num = 2
    elif ExponentMultiplier == 1:
        algorithm_num = 0
    else:
        raise ValueError(f"ExponentMultiplier should be -1, 0 or 1, found {ExponentMultiplier}")
    program_name = "./tree_tagger/applyFastJet"
    if use_pipe:
        summary_lines = produce_summary(eventWise, False)
        out = run_applyfastjet(summary_lines, str(DeltaR).encode(), 
                                  str(algorithm_num).encode())
        fastjets = Traditional.read_fastjet(out, eventWise, jet_name=jet_name)
        return fastjets
    produce_summary(eventWise)
    subprocess.run([program_name, str(DeltaR), str(algorithm_num), eventWise.dir_name])
    fastjets = Traditional.read_fastjet(eventWise.dir_name, eventWise=eventWise, jet_name=jet_name)
    return fastjets


def run_applyfastjet(input_lines, DeltaR, algorithm_num, program_path="./tree_tagger/applyFastJet", tries=0):
    '''
    Run applyfastjet, sending the provided input lines to stdin
    

    Parameters
    ----------
    input_lines: list of byte array
         contents of the input as byte arrays

    Returns
    -------
    output_lines: list of byte array
         the data that applyfastjet prints to stdout

    '''
    # input liens should eb one long byte string
    assert isinstance(input_lines, bytes)
    process = subprocess.Popen([program_path, DeltaR, algorithm_num],
                               stdout=subprocess.PIPE,
                               stdin=subprocess.PIPE)
    while process.poll() is None:
        output_lines = None
        process_output = process.stdout.readline()
        if process_output[:2] == b' *': # all system prompts start with *
            # note that SusHi reads the input file several times
            if b'**send input file to stdin' in process_output:
                process.stdin.write(input_lines)
                process.stdin.flush()
                process.stdin.close()
            elif b'**output file starts here' in process_output:
                process.wait()  # ok let it complete
                output_lines = process.stdout.readlines()
    if output_lines is None:
        print("Error! No output, retrying that input")
        tries += 1
        if tries > 5:
            print("Tried this 5 times... already")
            st()
        # recursive call
        output_lines = run_applyfastjet(input_lines, DeltaR, algorithm_num, program_path, tries)
    return output_lines


def cluster_multiapply(eventWise, cluster_algorithm, cluster_parameters={}, jet_name=None, batch_length=100, silent=False):
    if jet_name is None and 'jet_name' in cluster_parameters:
        jet_name = cluster_parameters['jet_name']
    elif jet_name is None:
        jet_name = {Traditional: "HomeJet",
                    run_FastJet: "FastJet",
                    Spectral: "SpectralJet"}.get(cluster_algorithm)
    cluster_parameters["jet_name"] = jet_name  # enforce consistancy
    if cluster_algorithm == run_FastJet:
        # make sure fast jet uses the pipe
        cluster_parameters["use_pipe"] = True
        jet_class = Traditional
    else:
        # often the cluster algorithm is the jet class
        jet_class = cluster_algorithm
        # make sure the assignment is done on creation
        cluster_parameters["assign"] = True
    eventWise.selected_index = None
    dir_name = eventWise.dir_name
    n_events = len(eventWise.JetInputs_Energy)
    start_point = len(getattr(eventWise, jet_name+"_Energy", []))
    if start_point >= n_events:
        if not silent:
            print("Finished")
        return True
    end_point = min(n_events, start_point+batch_length)
    if not silent:
        print(f" Starting at {100*start_point/n_events}%")
        print(f" Will stop at {100*end_point/n_events}%")
    # updated_dict will be replaced in the first batch
    updated_dict = None
    for event_n in range(start_point, end_point):
        if event_n % 10 == 0 and not silent:
            print(f"{100*event_n/n_events}%", end='\r', flush=True)
        eventWise.selected_index = event_n
        if len(eventWise.JetInputs_PT) == 0:
            continue  # there are no observables
        jets = cluster_algorithm(eventWise, **cluster_parameters)
        jets = jets.split()
        updated_dict = jet_class.create_updated_dict(jets, jet_name, event_n, eventWise, updated_dict)
    updated_dict = {name: awkward.fromiter(updated_dict[name]) for name in updated_dict}
    eventWise.append(updated_dict)
    return end_point == n_events


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
        for end_idx in end_points:
            ax.plot([center_rap, part_rap[jet_n][end_idx]], [center_phi, part_phi[jet_n][end_idx]],
                    linewidth=np.sqrt(part_Energy[jet_n][end_idx]), alpha=0.5, color=colour)
        if jet_n == 0:
            plt.scatter(part_rap[jet_n][end_points], part_phi[jet_n][end_points], s=part_Energy[jet_n][end_points], c=[colour],label=jet_name)
        else:
            plt.scatter(part_rap[jet_n][end_points], part_phi[jet_n][end_points], s=part_Energy[jet_n][end_points], c=[colour])
    ax.set_xlabel("Rapidity")
    ax.set_ylabel("$\\phi$")
    ax.legend()
    ew.selected_index = None


def plot_spider(ax, colour, body, body_size, leg_ends, leg_size):
    alpha=0.4
    leg_size = np.sqrt(leg_size)
    for end, size in zip(leg_ends, leg_size):
        line = np.vstack((body, end))
        # work out if this leg crossed the edge
        if np.abs(line[0, 1] - line[1, 1]) > np.pi:
            # work out the x coord of the axis cross
            top = np.argmax(line[:, 1])
            bottom = (top+1)%2
            distance_ratio = (np.pi - line[top, 1])/(np.pi + line[bottom, 1])
            x_top = distance_ratio * (line[bottom, 0] - line[top, 0])
            x_bottom = (line[top, 0] - line[bottom, 0])/distance_ratio
            plt.plot([line[top, 0], x_top], [line[top, 1], np.pi], 
                     c=colour, linewidth=size, alpha=alpha)
            plt.plot([line[bottom, 0], x_bottom], [line[bottom, 1], -np.pi], 
                     c=colour, linewidth=size, alpha=alpha)
                     
        else:
            plt.plot(line[:, 0], line[:, 1],
                     c=colour, linewidth=size, alpha=alpha)
    plt.scatter([body[0]], [body[1]], c='black', marker='o', s=body_size-1)
    plt.scatter([body[0]], [body[1]], c=[colour], marker='o', s=body_size+1)
    

def main():
    ax = plt.gca()
    # colourmap
    colours = plt.get_cmap('gist_rainbow')
    eventWise = Components.EventWise.from_file("megaIgnore/DeltaRp4_akt_arthur.awkd")
    # create inputs if needed
    if "JetInputs_Energy" not in eventWise.columns:
        filter_funcs = [filter_ends, filter_pt_eta]
        if "JetInputs_Energy" not in eventWise.columns:
            create_jetInputs(eventWise, filter_funcs)
    eventWise.selected_index = 0
    DeltaR = 0.4
    alpha=0.4
    pseudojet_traditional = Traditional(eventWise, DeltaR=DeltaR, ExponentMultiplier=0., jet_name="HomeJet")
    pseudojet_traditional.assign_parents()
    pjets_traditional = pseudojet_traditional.split()
    # plot the pseudojets
    # traditional_colours = [colours(i) for i in np.linspace(0, 0.4, len(pjets_traditional))]
    traditional_colours = ['red' for _ in pjets_traditional]
    for c, pjet in zip(traditional_colours, pjets_traditional):
        obs_idx = [i for i, child1 in enumerate(pjet.Child1) if child1==-1]
        input_rap = np.array(pjet._floats)[obs_idx, pjet._Rapidity_col]
        input_phi = np.array(pjet._floats)[obs_idx, pjet._Phi_col]
        leg_ends = np.vstack((input_rap, input_phi)).T
        input_energy = np.array(pjet._floats)[obs_idx, pjet._Energy_col]
        plot_spider(ax, c, [pjet.Rapidity, pjet.Phi], pjet.Energy, leg_ends, input_energy)
        #circle = plt.Circle((pjet.Rapidity, pjet.Phi), radius=DeltaR, edgecolor=c, fill=False)
        #ax.add_artist(circle)
    plt.plot([], [], c=c, alpha=alpha, label="HomeJets")
    pseudojet_spectral = Spectral(eventWise, DeltaR=DeltaR, ExponentMultiplier=0., NumEigenvectors=5, jet_name="SpectralJet")
    pseudojet_spectral.assign_parents()
    pjets_spectral = pseudojet_spectral.split()
    # plot the pseudojets
    #spectral_colours = [colours(i) for i in np.linspace(0.6, 1.0, len(pjets_spectral))]
    spectral_colours = ['blue' for _ in pjets_spectral]
    for c, pjet in zip(spectral_colours, pjets_spectral):
        obs_idx = [i for i, child1 in enumerate(pjet.Child1) if child1==-1]
        input_rap = np.array(pjet._floats)[obs_idx, pjet._Rapidity_col]
        input_phi = np.array(pjet._floats)[obs_idx, pjet._Phi_col]
        leg_ends = np.vstack((input_rap, input_phi)).T
        input_energy = np.array(pjet._floats)[obs_idx, pjet._Energy_col]
        plot_spider(ax, c, [pjet.Rapidity, pjet.Phi], pjet.Energy, leg_ends, input_energy)
        #circle = plt.Circle((pjet.Rapidity, pjet.Phi), radius=DeltaR, edgecolor=c, fill=False)
        #ax.add_artist(circle)
    plt.plot([], [], c=c, alpha=alpha, label="SpectralJet")
    plt.legend()
    plt.title("Jets")
    plt.xlabel("rapidity")
    plt.ylim(-np.pi, np.pi)
    plt.ylabel("phi")
    plt.show()
    return pjets_spectral


if __name__ == '__main__':
    main()
