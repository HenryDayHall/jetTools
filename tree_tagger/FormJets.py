""" Module for tools to create and handle jets """
import subprocess
import os
import csv
import scipy
import scipy.spatial
import awkward
from matplotlib import pyplot as plt
import matplotlib
from ipdb import set_trace as st
import numpy as np
from tree_tagger import Components, TrueTag, InputTools, Constants, FormShower, PlottingTools

TRUTH_COLOUR = 'limegreen'
SPECTRAL_COLOUR = 'dodgerblue'
FAST_COLOUR = 'tomato'
TRUTH_LINEWIDTH = 1.0
TRUTH_SIZE = 25.
JET_ALPHA = 0.5


class PseudoJet:
    """ Base class for jets, needs to be extended to be usable """
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
    def __init__(self, eventWise, selected_index=None,
                 jet_name='PseudoJet', from_PseudoRapidity=False,
                 ints_floats=None, **kwargs):
        """
        Class constructor

        Parameters
        ----------
        eventWise : EventWise
            data file for inputs
        selected_index : int
            event number to use, can already be set in the eventWise
            (Default; None)
        jet_name : string
            name to prefix the jet properties with when saving
            (Default; "PseudoJet")
        from_PseudoRapidity : bool
            Use pseudorapidity instead or rapidity
            (Default; False)
        int_floats : tuple of 2d array likes
            Predefined int and float tables.
            If not given they will be constructed from the eventWise.
            (Default; None)
        dict_jet_params : dict (optional)
            Settings for jet clustering. If not given defaults are used.
        root_jetInputIdxs : list of ints (optional)
            List of the indices of the roots of each jet.
        assign : bool (optional)
            Should the jets eb clustered immediatly?
            (Default; False)
        """
        # jets can have a varient name
        # this allows multiple saves in the same file
        self.jet_name = jet_name
        self.jet_parameters = {}
        dict_jet_params = kwargs.get('dict_jet_params', {})
        for key in dict_jet_params:  # create the formatting of a eventWise column
            formatted_key = key.replace(' ', '')
            # if letters after the first one are uppercase leave them up!
            formatted_key = formatted_key[0].capitalize() + formatted_key[1:]
            self.jet_parameters[formatted_key] = dict_jet_params[key]
        self.int_columns = [c.replace('Pseudojet', self.jet_name) for c in self.int_columns]
        self.float_columns = [c.replace('Pseudojet', self.jet_name) for c in self.float_columns]
        self.from_PseudoRapidity = from_PseudoRapidity
        if self.from_PseudoRapidity:
            idx = next(i for i, c in enumerate(self.float_columns) if c.endswith("_Rapidity"))
            self.float_columns[idx] = self.float_columns[idx].replace("_Rapidity",
                                                                      "_PseudoRapidity")
        # make a table of ints and a table of floats
        # lists not arrays, becuase they will grow
        self._set_column_numbers()
        if isinstance(eventWise, str):
            assert selected_index is not None, \
                    "If loading eventWise form file must specify and index"
            self.eventWise = Components.EventWise.from_file(eventWise)
        else:
            self.eventWise = eventWise
        if selected_index is not None:
            self.eventWise.selected_index = selected_index
        assert self.eventWise.selected_index is not None, \
                "Must specify an index (event number) for the eventWise"
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
            self._floats = np.vstack((eventWise.JetInputs_PT,
                                      rapidity_var,
                                      eventWise.JetInputs_Phi,
                                      eventWise.JetInputs_Energy,
                                      eventWise.JetInputs_Px,
                                      eventWise.JetInputs_Py,
                                      eventWise.JetInputs_Pz,
                                      np.zeros(self.n_inputs))).T.tolist()
            # as we go note the root notes of the pseudojets
            self.root_jetInputIdxs = []
        # define the physical distance measure
        # this requires that the class has had the attribute self.PhyDistance
        # set which should be done by the class that inherits
        # from this one before calling this constructor
        self._define_physical_distance()
        # keep track of how many clusters don't yet have a parent
        self._calculate_currently_avalible()
        self._calculate_distances()
        if kwargs.get("assign", False):
            self.assign_parents()

    def assign_parents(self):
        """ Join pseudojets until all avalible psseudojets are taken """
        while self.currently_avalible > 0:
            self._step_assign_parents()

    def plt_assign_parents(self):
        """
        Join pseudojets until all avalible psseudojets are taken.
        Also plot the process
        """
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
            removed = self._step_assign_parents()
            if removed is not None:
                decendents = self.get_decendants(lastOnly=True, pseudojet_idx=removed)
                decendents_idx = [self.idx_from_inpIdx(d) for d in decendents]
                draps = [self._floats[d][self._Rapidity_col] for d in decendents_idx]
                dphis = [self._floats[d][self._Phi_col] for d in decendents_idx]
                dpts = [1/self._floats[d][self._PT_col]**2 for d in decendents_idx]  # WHY??
                plt.scatter(draps, dphis, dpts, marker='D')
                print(f"Added jet of {len(decendents)} tracks," +
                      f"{self.currently_avalible} pseudojets unfinished")
                plt.pause(0.05)
                input("Press enter for next pseudojet")
        plt.show()

    def _step_assign_parents(self):
        """ Take a single step to join pseudojets """
        # now find the smallest distance
        row, column = np.unravel_index(np.argmin(self._distances2), self._distances2.shape)
        if row == column:
            self._remove_pseudojet(row)
            return row
        self._merge_pseudojets(row, column, self._distances2[row, column])
        return None

    def _define_physical_distance(self):
        """
        Define a function that measures distance in physical space,
        and also a function that measures distance of a particle to the beam.
        """
        pt_col = self._PT_col
        # caluclate invarint mass here!
        leaves = np.where(self.Child1 == -1)[0]
        sum_energies = sum(self._floats[row][self._Energy_col] for row in leaves)
        sum_px = sum(self._floats[row][self._Px_col] for row in leaves)
        sum_py = sum(self._floats[row][self._Py_col] for row in leaves)
        sum_pz = sum(self._floats[row][self._Pz_col] for row in leaves)
        invarient_mass2 = sum_energies**2 - sum_px**2 - sum_py**2 - sum_pz**2
        if invarient_mass2 > 0:
            invarient_mass = np.sqrt(invarient_mass2)
        elif invarient_mass2 > -1e-10:  # massless events are a weird specal case
            invarient_mass = 1.  # give them mass 1. to avoid nan issues.
        else:
            written_sum = "e**2 - px**2 - py**2 - pz**2 =\n" +\
                    f"{sum_energies}**2 - {sum_px}**2 - {sum_py}**2 - {sum_pz}**2" +\
                    f"={invarient_mass2}"
            raise ValueError(f"PhyDistance mass of event is tachyonic\n{written_sum}")
        # set up a few more variables
        exponent_now = self.ExpofPTPosition == 'input'
        exponent = self.ExpofPTMultiplier * 2
        inv_invar_exp = invarient_mass**-exponent
        deltaR2 = self.DeltaR**2
        # same for everything but Luclus
        if exponent_now:
            def beam_distance2(row):
                """
                Calculate the distance of this particle to the beam squared

                Parameters
                ----------
                row : list of floats
                    data about particle, in the order specified by the column
                    attributes

                Returns
                -------
                : float
                    the distance squared to the beam

                """
                return deltaR2 * row[pt_col]**exponent * inv_invar_exp
        else:
            def beam_distance2(row):
                """
                Calculate the distance of this particle to the beam squared

                Parameters
                ----------
                row : list of floats
                    data about particle, in the order specified by the column attributes

                Returns
                -------
                : float
                    the distance squared to the beam

                """
                return deltaR2
        if self.PhyDistance == "Luclus":
            rap_col = self._Rapidity_col
            phi_col = self._Phi_col
            def physical_distance2(row, column):
                """
                Calculate the physical distance between 2 particles.

                Parameters
                ----------
                row : list of ints
                    data about particle, in the order specified by the column attributes
                column : list of ints
                    data about particle, in the order specified by the column attributes

                Returns
                -------
                : float
                    the distance squared between the two particles

                """
                angular_distance = Components.angular_distance(row[phi_col], column[phi_col])
                distance2 = (row[rap_col] - column[rap_col])**2 + angular_distance**2
                if exponent_now:
                    distance2 *= inv_invar_exp*(row[pt_col]**exponent)*\
                                 (column[pt_col]**exponent) *\
                                 (row[pt_col] + column[pt_col])**-exponent
                return distance2
            def beam_distance2(row):
                """
                Calculate the distance of this particle to the beam squared

                Parameters
                ----------
                row : list of floats
                    data about particle, in the order specified by the column
                    attributes

                Returns
                -------
                : float
                    the distance squared to the beam

                """
                return deltaR2
        elif self.PhyDistance == 'invarient':
            px_col = self._Px_col
            py_col = self._Py_col
            pz_col = self._Pz_col
            e_col = self._Energy_col
            inv_invar2 = invarient_mass**-2
            def physical_distance2(row, column):
                """
                Calculate the physical distance between 2 particles.

                Parameters
                ----------
                row : list of ints
                    data about particle, in the order specified by the column attributes
                column : list of ints
                    data about particle, in the order specified by the column attributes

                Returns
                -------
                : float
                    the distance squared between the two particles

                """
                # this is proper length, but it may be -ve..... so should be using invarient mass?
                #distance2 = ((row[e_col] - column[e_col])**2
                #             - (row[px_col] - column[px_col])**2
                #             - (row[py_col] - column[py_col])**2
                #             - (row[pz_col] - column[pz_col])**2)
                # invarient mass
                distance2 = (row[e_col]*column[e_col]
                             - row[px_col]*column[px_col]
                             - row[py_col]*column[py_col]
                             - row[pz_col]*column[pz_col])*inv_invar2
                if exponent_now:
                    distance2 *= min(row[pt_col]**exponent, column[pt_col]**exponent) *\
                                 inv_invar_exp
                return distance2
        elif self.PhyDistance == 'normed':
            px_col = self._Px_col
            py_col = self._Py_col
            pz_col = self._Pz_col
            e_col = self._Energy_col
            small_num = 1e-10
            def physical_distance2(row, column):
                """
                Calculate the physical distance between 2 particles.

                Parameters
                ----------
                row : list of ints
                    data about particle, in the order specified by the column attributes
                column : list of ints
                    data about particle, in the order specified by the column attributes

                Returns
                -------
                : float
                    the distance squared between the two particles

                """
                energies = row[e_col] * column[e_col]
                if energies == 0:
                    energies = small_num
                row_3vec = np.array([row[px_col], row[py_col], row[pz_col]])
                column_3vec = np.array([column[px_col], column[py_col], column[pz_col]])
                distance2 = 1. - np.sum(row_3vec*column_3vec)/energies
                if exponent_now:
                    distance2 *= min(row[pt_col]**exponent, column[pt_col]**exponent) *\
                                 inv_invar_exp
                return distance2
        elif self.PhyDistance == 'angular':
            rap_col = self._Rapidity_col
            phi_col = self._Phi_col
            def physical_distance2(row, column):
                """
                Calculate the physical distance between 2 particles.

                Parameters
                ----------
                row : list of ints
                    data about particle, in the order specified by the column attributes
                column : list of ints
                    data about particle, in the order specified by the column attributes

                Returns
                -------
                : float
                    the distance squared between the two particles

                """
                angular_distance = Components.angular_distance(row[phi_col], column[phi_col])
                distance2 = (row[rap_col] - column[rap_col])**2 + angular_distance**2
                if exponent_now:
                    distance2 *= min(row[pt_col]**exponent, column[pt_col]**exponent) *\
                                 inv_invar_exp
                return distance2
        else:
            raise ValueError(f"Don't recognise {self.PhyDistance} as an Invarient")
        self.physical_distance2 = physical_distance2
        self.beam_distance2 = beam_distance2

    def _set_hyperparams(self, param_list, dict_jet_params, kwargs):
        """
        Using the default parameters and the chosen parameters set the attributes
        of the Pseudojet to contain the parameters used for clustering.
        Harmless to call multiple times.

        Parameters
        ----------
        param_list : dict of params
            dictionary of default settings, to be applied
            when parameters are not set elsewhere
            key is parameter name, value is parameter value
        kwargs : dict
            Parameters may be specified in the kwargs dict
            alongside other things
        dict_jet_params : dict of params
            parameters may be supplied together as a dictionary
            key is parameter name, value is parameter value
        """
        if dict_jet_params is None:
            dict_jet_params = {}
        stripped_params = {name.split("_")[-1]:name for name in dict_jet_params}
        for name in param_list:
            if name in stripped_params:
                assert name not in kwargs
                setattr(self, name, dict_jet_params[stripped_params[name]])
            elif name in kwargs:
                setattr(self, name, kwargs[name])
                dict_jet_params[name] = kwargs[name]
                del kwargs[name]
            else:
                setattr(self, name, param_list[name])
                dict_jet_params[name] = param_list[name]
        kwargs['dict_jet_params'] = dict_jet_params

    def _set_column_numbers(self):
        """
        Using the list of column names make the index of each
        column accesable as an attribute with the form self._col_<Varname>
        """
        #print('coln_psu', end='\r', flush=True)
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
            if name[prefix_len:] == "PseudoRapidity" :
                name = "Rapidity"
            attr_name = '_' + name[prefix_len:] + "_col"
            self._float_contents[name[prefix_len:]] = attr_name
            setattr(self, attr_name, i)

    def __dir__(self):
        """ Ensure the attributes are displayed in consistant order """
        new_attrs = set(super().__dir__())
        return sorted(new_attrs)

    def _calculate_currently_avalible(self):
        """ Update the cound of how many pseudojets could pottentially be combined """
        # keep track of how many clusters don't yet have a parent
        self.currently_avalible = sum([p[self._Parent_col] == -1 for p in self._ints])

    def __getattr__(self, attr_name):
        """
        Make the attributes for the floats and ints used to construct jets.
        The integer columns are simply returned as numpy arrays.
        The float columns are used to calculate overall flat properties of each formed jet.
        """
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
        if attr_name in self._int_contents:
            # ints return every value
            col_num = getattr(self, self._int_contents[attr_name])
            return np.fromiter((ints[col_num] for ints in self._ints), dtype=int)
        elif attr_name == "Rapidity":
            # if the jet was constructed with pseudorapidity
            # we might still want to know the rapidity
            return Components.ptpze_to_rapidity(self.PT, self.Pz, self.Energy)
        elif attr_name == "Pseudorapidity":
            # vice verca
            return Components.theta_to_pseudorapidity(self.Theta)
        raise AttributeError(f"{self.__class__.__name__} does not have {attr_name}")

    @property
    def P(self):
        """ Total momentum of each jet """
        if len(self) == 0:
            return np.nan
        roots = self.Parent == -1
        if len(roots) == 1:
            birr = np.linalg.norm([self.Px, self.Py, self.Pz], axis=0)
        else:
            birr = np.linalg.norm([self.Px[roots], self.Py[roots], self.Pz[roots]],
                                  axis=0)
        return birr

    @property
    def Theta(self):
        """ Theta of the jets"""
        if len(self) == 0:
            return np.nan
        roots = self.Parent == -1
        if len(roots) == 1:
            theta = Components.ptpz_to_theta(self.PT, self.Pz)
        else:
            theta = Components.ptpz_to_theta(self.PT[roots], self.Pz[roots])
        return theta

    @classmethod
    def create_updated_dict(cls, pseudojets, jet_name, event_index, eventWise=None, arrays=None):
        """
        Adds an event to the dictionary of columns to be appended to an eventWise for writing

        Parameters
        ----------
        pseudojets : list of PseudoJet
            pseudojet objects created in a chosen event.
        jet_name: string
            name of the jet for prefixes in the eventWise
        event_index : int
            zero index event number that these pseudojets belong to
        eventWise : EventWise
            eventWise object which can be used to find jets form other events.
            If not given, an arrays must be given.
            Will not be written to.
            (Default value = None)
        arrays : dictionary of lists
            The data from other events
            If not given, an eventWise must be given.
            (Default value = None)

        Returns
        -------
        arrays : dictionary of lists
            The data from existing events plus this one.
        """
        if arrays is None:
            save_columns = [jet_name + "_RootInputIdx"]
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
            arrays[jet_name + "_RootInputIdx"][event_index].append(awkward.fromiter(jet.root_jetInputIdxs))
            # if an array is deep it needs converting to an awkward array
            ints = awkward.fromiter(jet._ints)
            for col_num, name in enumerate(jet.int_columns):
                arrays[name.replace('PseudoJet', jet_name)][event_index].append(ints[:, col_num])
            floats = awkward.fromiter(jet._floats)
            for col_num, name in enumerate(jet.float_columns):
                arrays[name.replace('PseudoJet', jet_name)][event_index].append(floats[:, col_num])
        return arrays

    def create_param_dict(self):
        """
        Create a dictionary of parameters discribing the settings used for this jet.

        Returns
        -------
        params : dict
            keys are name sof parameters and values are parameter values
        """
        # add any default values
        defaults = {name:value for name, value in self.param_list.items()
                    if name not in self.jet_parameters}
        params = {**self.jet_parameters, **defaults}
        return params

    @classmethod
    def write_event(cls, pseudojets, jet_name="Pseudojet", event_index=None, eventWise=None):
        """
        Save a jets from a single event

        Parameters
        ----------
        pseudojets : list of PseudoJet
            pseudojet objects created in a chosen event.
        jet_name: string
            name of the jet for prefixes in the eventWise
        event_index : int
            zero index event number that these pseudojets belong to
            can also be specified as a selected_index of the eventWise
            (Default value = None)
        eventWise : EventWise
            eventWise object in which to store these pseudojets
            If not given, the eventWise of the first pseudojet will be used
            (Default value = None)
        """
        if eventWise is None:
            eventWise = pseudojets[0].eventWise
        # only need to check the parameters of one jet (adds the hyperparameters)
        pseudojets[0].check_params(eventWise)
        if event_index is None:
            event_index = eventWise.selected_index
        arrays = cls.create_updated_dict(pseudojets, jet_name, event_index, eventWise)
        arrays = {name: awkward.fromiter(arrays[name]) for name in arrays}
        eventWise.append(**arrays)

    def check_params(self, eventWise):
        """
        If the eventWise contains params, verify they are the same as in this Pseudojet.
        If no parameters are found in the eventWise add them.

        Parameters
        ----------
        eventWise : EventWise
            eventWise object to look for jet parameters in

        Returns
        -------
        : bool
            The parameters in the eventWise match those of this jet.

        """
        my_params = self.create_param_dict()
        written_params = get_jet_params(eventWise, self.jet_name)
        if written_params:  # if written params exist check they match the jets params
            # returning false imediatly if not
            if set(written_params.keys()) != set(my_params.keys()):
                return False
            for name in written_params:
                try:
                    same = np.allclose(written_params[name], my_params[name])
                    if not same:
                        return False
                except TypeError:
                    if written_params[name] != my_params[name]:
                        return False
        else:  # save the jets params
            new_hyper = {self.jet_name + '_' + name: my_params[name] for name in my_params}
            eventWise.append_hyperparameters(**new_hyper)
        # if we get here everything went well
        return True

    @classmethod
    def multi_from_file(cls, file_name, event_idx, jet_name="Pseudojet",
                        batch_start=None, batch_end=None):
        """
        Read jets form a given event from file.

        Parameters
        ----------
        file_name : string
            path of an eventWise file with jets in
        event_idx : int
            zero index event number that to read jets from
        jet_name: string
            name of the jet for prefixes in the eventWise
            (Default; "Pseudojet")
        batch_start: int
            inclusive jet to start reading
        batch_end : int
            exclusive jet to end on

        Returns
        -------
        jets: list of PseudoJet
            the read jets

        """
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
        param_columns = [c for c in eventWise.hyperparameter_columns if c.startswith(jet_name)]
        param_dict = {name: getattr(eventWise, name) for name in param_columns}
        for i in range(batch_start, batch_end):
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
        """ Set the root_jetInputIdxs by looking for parentless pseudojets """
        self.root_jetInputIdxs = []
        # should only be needed for reading from file
        assert self.currently_avalible == 0, "Assign parents before you calculate roots"
        pseudojet_ids = self.InputIdx
        parent_ids = self.Parent
        for mid, pid in zip(parent_ids, pseudojet_ids):
            if (mid == -1 or
                mid not in pseudojet_ids or
                mid == pid):
                self.root_jetInputIdxs.append(pid)

    def split(self):
        """
        Split this PseudoJet into as many unconnected jets as it contains

        Returns
        -------
        JetList : list of PseudoJet
            the indervidual jets found in here
        """
        assert self.currently_avalible == 0, "Need to assign_parents before splitting"
        if len(self) == 0:  # nothing else to do if the jet is empty
            return []
        assert len(self.root_jetInputIdxs), \
                "A fully merged cluster should have at least one jet root"
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
        """ Calculate all distances between avalible pseudojets """
        # this is caluculating all the distances
        raise NotImplementedError

    def _recalculate_one(self, remove_index, replace_index):
        """
        Recalculate all the distances involving one pseudojet

        Parameters
        ----------
        remove_index : int
            index of the jet that will be removed (moved to the back)
            after joining
        replace_index : int
            index of the jet that will become the index of the combined jet
            (current jet will be moved to the back)

        """
        raise NotImplementedError

    def _merge_pseudojets(self, pseudojet_index1, pseudojet_index2, distance2):
        """
        Merge two pseudojets to form a new pseudojet, moving the
        exising pseudojets to the back of the ints/floats lists

        Parameters
        ----------
        pseudojet_index1 : int
            index of the first input pseudojet
        pseudojet_index2 : int
            index of the second input pseudojet
        distance2 : float
            distance squared between them.
        """
        replace_index, remove_index = sorted([pseudojet_index1, pseudojet_index2])
        new_pseudojet_ints, new_pseudojet_floats = \
                self._combine(remove_index, replace_index, distance2)
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
        """
        Remove a pseudojet from the currently_avalible and move it to the back
        of the ints/floats lists.

        Parameters
        ----------
        pseudojet_index : int
            index to remove

        """
        # move the first pseudojet to the back without replacement
        pseudojet_ints = self._ints.pop(pseudojet_index)
        pseudojet_floats = self._floats.pop(pseudojet_index)
        self._ints.append(pseudojet_ints)
        self._floats.append(pseudojet_floats)
        self.root_jetInputIdxs.append(pseudojet_ints[self._InputIdx_col])
        # delete the row and column
        self._distances2 = np.delete(self._distances2, (pseudojet_index), axis=0)
        self._distances2 = np.delete(self._distances2, (pseudojet_index), axis=1)
        # one less pseudojet avalible
        self.currently_avalible -= 1

    def idx_from_inpIdx(self, jetInputIdx):
        """
        Given a JetInputIdx, which may be an idx of one of the jet inputs,
        of an id ceated subsequently during clustering, find the idx
        of the corrisponding row in the ints/floats

        Parameters
        ----------
        jetInputIdx : int
            id of type JetInputIdx

        Returns
        -------
        pseudojet_idx : int
            the row number of this jetInputIdx
        """
        col = self._InputIdx_col
        try:
            pseudojet_idx = next((idx for idx, row in enumerate(self._ints)
                                  if row[col] == jetInputIdx))
            return pseudojet_idx
        except StopIteration:
            raise ValueError(f"No pseudojet with ID {jetInputIdx}")

    def get_decendants(self, lastOnly=True, jetInputIdx=None, pseudojet_idx=None):
        """
        Get all decendants of a chosen particle within the structure of the jet.

        Parameters
        ----------
        lastOnly : bool
            Only return the end point decendants
            (Default value = True)
        jetInputIdx : int
            jetInputIdx used to identify the starting particle
            if not given pseudojet_idx required
            (Default value = None)
        pseudojet_idx : int
            Internal index to identify the particle
            if not given jetInputIdx required
            (Default value = None)

        Returns
        -------
        decendants : list of ints
            decendants identified by jetInputIdx

        """
        if jetInputIdx is None and pseudojet_idx is None:
            raise TypeError("Need to specify a pseudojet")
        if pseudojet_idx is None:
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
        child1 = self._ints[pseudojet_idx][child1_col]
        child2 = self._ints[pseudojet_idx][child2_col]
        if child1 >= 0:
            to_check.append(child1)
        if child2 >= 0:
            to_check.append(child2)
        while len(to_check) > 0:
            jetInputIdx = to_check.pop()
            pseudojet_idx = self.idx_from_inpIdx(jetInputIdx)
            if (pseudojet_idx in local_obs or not lastOnly):
                decendents.append(jetInputIdx)
            else:
                ignore.append(jetInputIdx)
            child1 = self._ints[pseudojet_idx][child1_col]
            child2 = self._ints[pseudojet_idx][child2_col]
            if child1 >= 0 and child1 not in (decendents + ignore):
                to_check.append(child1)
            if child2 >= 0 and child2 not in (decendents + ignore):
                to_check.append(child2)
        return decendents

    def local_obs_idx(self):
        """
        Local indices of the pseudojets corrisponding to inputs (which would be observabel)

        Returns
        -------
        idx_are_obs : list of ints
            local idx of the observable pseudojets
        """
        idx_are_obs = [i for i in range(len(self)) if
                       (self._ints[i][self._Child1_col] < 0 and
                        self._ints[i][self._Child2_col] < 0)]
        return idx_are_obs

    def _combine(self, pseudojet_index1, pseudojet_index2, distance2):
        """
        Caluclate the floats and ints created by combining two pseudojets.

        Parameters
        ----------
        pseudojet_index1 : int
            index of the first pseudojet to input
        pseudojet_index2 : int
            index of the second pseudojet to input
        distance2 : float
            distanc esquared between the pseudojets

        Returns
        -------
        ints : list of ints
            int columns of the combined pseudojet, order as per the column attributes
        floats : list of floats
            float columns of the combined pseudojet, order as per the column attributes
        """
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
        # check for tachyonic behavior and fix - makes no diference
        #if energy**2 < px**2 + py**2 + pz**2:
        #    energy = np.sqrt(px**2 + py**2 + pz**2)
        #    floats[self._Energy_col] = energy
        phi, pt = Components.pxpy_to_phipt(px, py)
        floats[self._PT_col] = pt
        floats[self._Phi_col] = phi
        if self.from_PseudoRapidity:
            theta = Components.ptpz_to_theta(pt, pz)
            floats[self._Rapidity_col] = Components.theta_to_pseudorapidity(theta)
        else:
            floats[self._Rapidity_col] = Components.ptpze_to_rapidity(pt, pz, energy)
        # fix the distance
        floats[self._JoinDistance_col] = np.sqrt(distance2)
        return ints, floats

    def __len__(self):
        """ consider the length to be equal to the number of pseudojets """
        return len(self._ints)

    def __eq__(self, other):
        """ consider pseudojets to eb equal if their ints and flaots are equal """
        if len(self) != len(other):
            return False
        ints_eq = self._ints == other._ints
        floats_eq = np.allclose(self._floats, other._floats)
        return ints_eq and floats_eq


class Traditional(PseudoJet):
    """ Jet clustering in the style of kt/anti-kt/cambridge-acchen """
    param_list = {'DeltaR': .8, 'ExpofPTMultiplier': 0, 'PhyDistance': 'angular'}
    permited_values = {'DeltaR': Constants.numeric_classes['pdn'],
                       'ExpofPTMultiplier': Constants.numeric_classes['rn'],
                       'PhyDistance': ['angular', 'normed', 'Luclus', 'invarient']}
    def __init__(self, eventWise=None, dict_jet_params=None, **kwargs):
        """
        Class constructor

        Parameters
        ----------
        eventWise : EventWise
            data file for inputs
        selected_index : int
            event number to use, can already be set in the eventWise
            (Default; None)
        jet_name : string
            name to prefix the jet properties with when saving
            (Default; "PseudoJet")
        from_PseudoRapidity : bool
            Use pseudorapidity instead or rapidity
            (Default; False)
        int_floats : tuple of 2d array likes
            Predefined int and float tables.
            If not given they will be constructed from the eventWise.
            (Default; None)
        dict_jet_params : dict (optional)
            Settings for jet clustering. If not given defaults are used.
        root_jetInputIdxs : list of ints (optional)
            List of the indices of the roots of each jet.
        assign : bool (optional)
            Should the jets eb clustered immediatly?
            (Default; False)
        """
        self._set_hyperparams(self.param_list, dict_jet_params, kwargs)
        # check for nonsense in the kwargs or dict_jet_params
        pos = kwargs.get('ExpofPTPosition', 'input')
        if dict_jet_params is not None:
            pos = dict_jet_params.get('ExpofPTPosition', pos)
        assert pos == 'input'  # don't just silently swallow nonsense
        # then set the right thing anyway
        self.ExpofPTPosition = 'input'
        if dict_jet_params is not None:
            kwargs['dict_jet_params'] = dict_jet_params
        super().__init__(eventWise, **kwargs)

    def _calculate_distances(self):
        """ Calculate all distances between avalible pseudojets """
        # this is caluculating all the distances
        self._distances2 = np.full((self.currently_avalible, self.currently_avalible), np.inf)
        # for speed, make local variables
        pt_col = self._PT_col
        for row in range(self.currently_avalible):
            for column in range(self.currently_avalible):
                if column > row:
                    continue  # we only need a triangular matrix due to symmetry
                if self._floats[row][pt_col] == 0:
                    distance2 = 0  # soft radation might as well be at 0 distance
                elif column == row:
                    distance2 = self.beam_distance2(self._floats[row])
                else:
                    distance2 = self.physical_distance2(self._floats[row], self._floats[column])
                self._distances2[row, column] = distance2

    def _recalculate_one(self, remove_index, replace_index):
        """
        Recalculate all the distances involving one pseudojet

        Parameters
        ----------
        remove_index : int
            index of the jet that will be removed (moved to the back)
            after joining
        replace_index : int
            index of the jet that will become the index of the combined jet
            (current jet will be moved to the back)

        """
        # delete the larger index keep the smaller index
        assert remove_index > replace_index
        # delete the first row and column of the merge
        self._distances2 = np.delete(self._distances2, (remove_index), axis=0)
        self._distances2 = np.delete(self._distances2, (remove_index), axis=1)

        # calculate new values into the second column
        for row in range(self.currently_avalible):
            column = replace_index
            if column > row:
                distance2 = self._distances2[column, row]
            if column == row:
                distance2 = self.beam_distance2(self._floats[row])
            else:
                distance2 = self.physical_distance2(self._floats[row], self._floats[column])
            self._distances2[row, column] = distance2

    @classmethod
    def read_fastjet(cls, arg, eventWise, jet_name="FastJet", do_checks=False):
        """
        Read the outputs of the fastjet program into a PseudoJet

        Parameters
        ----------
        arg : string or list of strings
            if the argument is a sngle string it is the path
            to a directory in which text fiels containing output are stored
            If the argument is a list of strings it is the byte output of the
            fastjet program
        eventWise : EventWise
            data file to assign these jets to
        jet_name : string
            Name of the jet to be prefixed in the eventWise
            (Default value = "FastJet")
        do_checks : bool
            If checks ont he form of the fastjet output should be done (slow)
            (Default value = False)


        Returns
        -------
        new_pseudojet : PseudoJet
            the peseudojets read from the program

        """
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
            ExpofPTMultiplier = 1
        elif algorithm_name == 'cambridge_algorithm':
            ExpofPTMultiplier = 0
        elif algorithm_name == 'antikt_algorithm':
            ExpofPTMultiplier = -1
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
        assert set(np.arange(n_inputs)).issubset(set(fast_ints[:, icolumns["InputIdx"]])),\
                "Problem with inpu idx"
        next_free = np.max(fast_ints[:, icolumns["InputIdx"]], initial=-1) + 1
        fast_idx_dict = {}
        for line_idx, i in fast_ints[:, [icolumns["pseudojet_id"], icolumns["InputIdx"]]]:
            if i == -1:
                fast_idx_dict[line_idx] = next_free
                next_free += 1
            else:
                fast_idx_dict[line_idx] = i
        fast_idx_dict[-1] = -1
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
                    assert line[icolumns["child2_id"]] == -1, \
                            identifier + "has only one child"
                else:
                    assert line[icolumns["child1_id"]] != line[icolumns["child2_id"]], \
                            identifier + " child1 and child2 are same"
                    child1_line = fast_ints[fast_ints[:, icolumns["InputIdx"]]
                                            == line[icolumns["child1_id"]]][0]
                    assert child1_line[1] == line[0], \
                            identifier + " first child dosn't acknowledge parent"
                    child2_line = fast_ints[fast_ints[:, icolumns["InputIdx"]]
                                            == line[icolumns["child2_id"]]][0]
                    assert child2_line[1] == line[0], \
                            identifier + " second child dosn't acknowledge parent"
                if line[1] != -1:
                    assert line[icolumns["InputIdx"]] != line[icolumns["parent_id"]], \
                            identifier + "is it's own mother"
                    parent_line = fast_ints[fast_ints[:, icolumns["InputIdx"]]
                                            == line[icolumns["parent_id"]]][0]
                    assert line[0] in parent_line[[icolumns["child1_id"],
                                                   icolumns["child2_id"]]], \
                            identifier + " parent doesn't acknowledge child"
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
                            ExpofPTMultiplier=ExpofPTMultiplier,
                            jet_name=jet_name)
        new_pseudojet.currently_avalible = 0
        new_pseudojet._calculate_roots()
        return new_pseudojet


class Spectral(PseudoJet):
    """ Clustering algorithm that embeds the jet inputs into spectral space to cluster """
    # list the params with default values
    param_list = {'DeltaR': .2, 'NumEigenvectors': np.inf,
                  'ExpofPTPosition': 'input', 'ExpofPTMultiplier': 0,
                  'AffinityType': 'exponent', 'AffinityCutoff': None,
                  'Laplacien': 'unnormalised',
                  'PhyDistance': 'angular', 'StoppingCondition': 'standard'}
    permited_values = {'DeltaR': Constants.numeric_classes['pdn'],
                       'NumEigenvectors': [Constants.numeric_classes['nn'], np.inf],
                       'ExpofPTPosition': ['input', 'eigenspace'],
                       'ExpofPTMultiplier': Constants.numeric_classes['rn'],
                       'AffinityType': ['linear', 'exponent', 'exponent2', 'inverse'],
                       'AffinityCuttoff': [None, ('knn', Constants.numeric_classes['nn']), ('distance', Constants.numeric_classes['pdn'])],
                       'Laplacien': ['unnormalised', 'symmetric'],
                       'PhyDistance': ['angular', 'normed', 'Luclus', 'invarient'],
                       'StoppingCondition': ['standard', 'beamparticle']}
    def __init__(self, eventWise=None, dict_jet_params=None, **kwargs):
        """
        Class constructor

        Parameters
        ----------
        eventWise : EventWise
            data file for inputs
        selected_index : int
            event number to use, can already be set in the eventWise
            (Default; None)
        jet_name : string
            name to prefix the jet properties with when saving
            (Default; "PseudoJet")
        from_PseudoRapidity : bool
            Use pseudorapidity instead or rapidity
            (Default; False)
        int_floats : tuple of 2d array likes
            Predefined int and float tables.
            If not given they will be constructed from the eventWise.
            (Default; None)
        dict_jet_params : dict (optional)
            Settings for jet clustering. If not given defaults are used.
        root_jetInputIdxs : list of ints (optional)
            List of the indices of the roots of each jet.
        assign : bool (optional)
            Should the jets eb clustered immediatly?
            (Default; False)
        """
        self._set_hyperparams(self.param_list, dict_jet_params, kwargs)
        self._define_calculate_affinity()
        self.eigenvalues = []  # create a list to track the eigenvalues
        self.beam_particle = self.StoppingCondition == 'beamparticle'
        assign = kwargs.get('assign', False)
        kwargs['assign'] = False  # don't let the super constructor assign
        if dict_jet_params is not None:
            kwargs['dict_jet_params'] = dict_jet_params
        super().__init__(eventWise, **kwargs)
        self._calculate_eigenspace()  # we need to make the eigenspace first
        if assign:
            self.assign_parents()

    def _calculate_distances(self):
        """ Calculate all distances between avalible pseudojets """
        # if there is a beam particle need to get the distance to the beam particle too
        n_distances = self.currently_avalible + self.beam_particle
        if n_distances < 2:
            self._distances2 = np.zeros((1, 1))
            self._affinity = np.array([[]])
            return
        # to start with create a 'normal' distance measure
        # this can be based on any of the three algorithms
        physical_distances2 = np.zeros((n_distances, n_distances))
        # for speed, make local variables
        pt_col = self._PT_col
        rap_col = self._Rapidity_col
        phi_col = self._Phi_col
        # future calculatins will depend on the starting positions
        self._starting_position = np.array([self._floats[row][:] for row
                                      in range(self.currently_avalible)])
        if self.beam_particle:
            # the beam particles dosn't have a real location,
            # but to preserve the dimensions of future calculations, add it in
            self._starting_position = np.vstack((self._starting_position,
                                                 np.ones(len(self.float_columns))))
            # it is added to the end so as to maintain the indices
        for row in range(self.currently_avalible):
            for column in range(self.currently_avalible):
                if column < row:
                    distance2 = physical_distances2[column, row]  # the matrix is symmetric
                elif self._floats[row][pt_col] == 0:
                    distance2 = 0  # soft radation might as well be at 0 distance
                elif column == row:
                    # not used
                    continue
                else:
                    distance2 = self.physical_distance2(self._floats[row], self._floats[column])
                physical_distances2[row, column] = distance2
        if self.beam_particle:
            # the last row and column should give the distance of each particle to the beam
            physical_distances2[-1, :] = [self.beam_distance2(row) for row in self._starting_position]
            physical_distances2[:, -1] = physical_distances2[-1, :]
        np.fill_diagonal(physical_distances2, 0.)
        # now we are in posessio of a standard distance matrix for all points,
        # we can make an affinity calculation
        self._affinity = self.calculate_affinity(physical_distances2)
        # a graph laplacien can be calculated
        np.fill_diagonal(self._affinity, 0.)  # the self._affinity may have problems on the diagonal

    def _calculate_eigenspace(self):
        """
        Calculate the embedding of the currently_avalible pseudojets in eignspace
        Also find the distances in eigenspace and the eigenvalues.
        """
        if np.sum(np.abs(self._affinity), initial=0) == 0.:
            # everything is seperated
            self.root_jetInputIdxs = [row[self._InputIdx_col] for row in
                                      self._ints[:self.currently_avalible]]
            self.currently_avalible = 0
            return
        diagonal = np.diag(np.sum(self._affinity, axis=1))
        if self.Laplacien == 'unnormalised':
            laplacien = diagonal - self._affinity
        elif self.Laplacien == 'symmetric':
            laplacien = diagonal - self._affinity
            self.alt_diag = np.diag(diagonal)**(-0.5)
            self.alt_diag[np.diag(diagonal) == 0] = 0.
            diag_alt_diag = np.diag(self.alt_diag)
            laplacien = np.matmul(diag_alt_diag, np.matmul(laplacien, diag_alt_diag))
        else:
            raise NotImplementedError(f"Don't have a laplacien {self.Laplacien}")
        # get the eigenvectors (we know the smallest will be identity)
        try:
            eigenvalues, eigenvectors = \
                    scipy.linalg.eigh(laplacien, eigvals=(1, self.NumEigenvectors+1))
        except (ValueError, TypeError):
            # sometimes there are fewer eigenvalues avalible
            # just take waht can be found
            eigenvalues, eigenvectors = scipy.linalg.eigh(laplacien)
            eigenvalues = eigenvalues[1:]
            eigenvectors = eigenvectors[:, 1:]
        except Exception as e:
            # display whatever caused this
            print(f"Exception while processing event {self.eventwise.selected_index}")
            print(f"With jet params; {self.jet_parameters}")
            print(e)
            self.root_jetInputIdxs = [row[self._InputIdx_col] for row in
                                      self._ints[:self.currently_avalible]]
            self.currently_avalible = 0
            return
        self.eigenvalues.append(eigenvalues.tolist())
        # at the start the eigenspace positions are the eigenvectors
        self._eigenspace = np.copy(eigenvectors)
        # now treating the rows of this matrix as the new points get euclidien distances
        self._distances2 = scipy.spatial.distance.squareform(
                scipy.spatial.distance.pdist(eigenvectors,
                metric='sqeuclidean'))
        if self.ExpofPTPosition == 'eigenspace':
            exponent = 2 * self.ExpofPTMultiplier
            # if beamparticle the last entry will be nonsense, but we wont touch it anyway
            pt_fractions = np.fromiter((row[self._PT_col]**exponent for
                                        row in self._starting_position),
                                       dtype=float)
            for row in range(self.currently_avalible):
                for column in range(self.currently_avalible):
                    if column < row:
                        distance2 = self._distances2[column, row]  # the matrix is symmetric
                    elif column == row:
                        # not used
                        continue
                    else:
                        # at this point apply the pt factors
                        distance2 = min(pt_fractions[row], pt_fractions[column]) *\
                                    self._distances2[row, column]
                    self._distances2[row, column] = distance2
            if self.beam_particle:
                self._distances2[-1, :-1] *= pt_fractions[:-1]
                self._distances2[:-1, -1] *= pt_fractions[:-1]
        # if the clustering is not going to stop at 1 we must put something in the diagonal
        if self.beam_particle:  # in he case of a beam particle we stop the clustering when
            # our particle reaches the beam particle
            # so the diagonal should never be grouped with
            np.fill_diagonal(self._distances2, np.inf)
        else:
            # the diagonal is the stopping condition
            np.fill_diagonal(self._distances2, self.DeltaR**2)

    def _define_calculate_affinity(self):
        """ Define functions to caluclate affinity from real distance """
        if self.AffinityCutoff is not None:
            cutoff_type = self.AffinityCutoff[0]
            cutoff_param = self.AffinityCutoff[1]
            if cutoff_type == 'knn':
                if self.AffinityType == 'exponent':
                    def calculate_affinity(distances2):
                        """
                        Given physical distance squared, find an affinity.

                        Parameters
                        ----------
                        distances2 : array like of floats
                            the distances squared

                        Returns
                        -------
                        affinity : array like of floats
                            the affinities

                        """
                        affinity = np.exp(-(distances2**0.5))
                        affinity[np.argsort(distances2, axis=0) < cutoff_param] = 0
                        return affinity
                elif self.AffinityType == 'exponent2':
                    def calculate_affinity(distances2):
                        """
                        Given physical distance squared, find an affinity.

                        Parameters
                        ----------
                        distances2 : array like of floats
                            the distances squared

                        Returns
                        -------
                        affinity : array like of floats
                            the affinities

                        """
                        affinity = np.exp(-(distances2))
                        affinity[np.argsort(distances2, axis=0) < cutoff_param] = 0
                        return affinity
                elif self.AffinityType == 'linear':
                    def calculate_affinity(distances2):
                        """
                        Given physical distance squared, find an affinity.

                        Parameters
                        ----------
                        distances2 : array like of floats
                            the distances squared

                        Returns
                        -------
                        affinity : array like of floats
                            the affinities

                        """
                        affinity = -distances2**0.5
                        affinity[np.argsort(distances2, axis=0) < cutoff_param] = 0
                        affinity -= np.min(affinity)
                        return affinity
                elif self.AffinityType == 'inverse':
                    def calculate_affinity(distances2):
                        """
                        Given physical distance squared, find an affinity.

                        Parameters
                        ----------
                        distances2 : array like of floats
                            the distances squared

                        Returns
                        -------
                        affinity : array like of floats
                            the affinities

                        """
                        affinity = distances2**-0.5
                        affinity[np.argsort(distances2, axis=0) < cutoff_param] = 0
                        return affinity
                else:
                    raise ValueError(f"affinity type {self.AffinityType} unknown")
            elif cutoff_type == 'distance':
                cutoff_param2 = cutoff_param**2
                if self.AffinityType == 'exponent':
                    def calculate_affinity(distances2):
                        """
                        Given physical distance squared, find an affinity.

                        Parameters
                        ----------
                        distances2 : array like of floats
                            the distances squared

                        Returns
                        -------
                        affinity : array like of floats
                            the affinities

                        """
                        affinity = np.exp(-(distances2**0.5))
                        affinity[distances2 > cutoff_param2] = 0
                        return affinity
                elif self.AffinityType == 'exponent2':
                    def calculate_affinity(distances2):
                        """
                        Given physical distance squared, find an affinity.

                        Parameters
                        ----------
                        distances2 : array like of floats
                            the distances squared

                        Returns
                        -------
                        affinity : array like of floats
                            the affinities

                        """
                        affinity = np.exp(-(distances2))
                        affinity[distances2 > cutoff_param2] = 0
                        return affinity
                elif self.AffinityType == 'linear':
                    def calculate_affinity(distances2):
                        """
                        Given physical distance squared, find an affinity.

                        Parameters
                        ----------
                        distances2 : array like of floats
                            the distances squared

                        Returns
                        -------
                        affinity : array like of floats
                            the affinities

                        """
                        affinity = -distances2**0.5
                        affinity[distances2 > cutoff_param2] = 0
                        affinity -= np.min(affinity)
                        return affinity
                elif self.AffinityType == 'inverse':
                    def calculate_affinity(distances2):
                        """
                        Given physical distance squared, find an affinity.

                        Parameters
                        ----------
                        distances2 : array like of floats
                            the distances squared

                        Returns
                        -------
                        affinity : array like of floats
                            the affinities

                        """
                        affinity = distances2**-0.5
                        affinity[distances2 > cutoff_param2] = 0
                        return affinity
                else:
                    raise ValueError(f"affinity type {self.AffinityType} unknown")
            else:
                raise ValueError(f"cut off {cutoff_type} unknown")
        else:
            if self.AffinityType == 'exponent':
                def calculate_affinity(distances2):
                    """
                    Given physical distance squared, find an affinity.

                    Parameters
                    ----------
                    distances2 : array like of floats
                        the distances squared

                    Returns
                    -------
                    affinity : array like of floats
                        the affinities

                    """
                    affinity = np.exp(-(distances2**0.5))
                    return affinity
            elif self.AffinityType == 'exponent2':
                def calculate_affinity(distances2):
                    """
                    Given physical distance squared, find an affinity.

                    Parameters
                    ----------
                    distances2 : array like of floats
                        the distances squared

                    Returns
                    -------
                    affinity : array like of floats
                        the affinities

                    """
                    affinity = np.exp(-(distances2))
                    return affinity
            elif self.AffinityType == 'linear':
                def calculate_affinity(distances2):
                    """
                    Given physical distance squared, find an affinity.

                    Parameters
                    ----------
                    distances2 : array like of floats
                        the distances squared

                    Returns
                    -------
                    affinity : array like of floats
                        the affinities

                    """
                    affinity = -distances2**0.5
                    affinity -= np.min(affinity)
                    return affinity
            elif self.AffinityType == 'inverse':
                def calculate_affinity(distances2):
                    """
                    Given physical distance squared, find an affinity.

                    Parameters
                    ----------
                    distances2 : array like of floats
                        the distances squared

                    Returns
                    -------
                    affinity : array like of floats
                        the affinities

                    """
                    affinity = distances2**-0.5
                    return affinity
            else:
                raise ValueError(f"affinity type {self.AffinityType} unknown")
        # this is make into a class fuction becuase it will b needed elsewhere
        self.calculate_affinity = calculate_affinity

    def _merge_pseudojets(self, pseudojet_index1, pseudojet_index2, distance2):
        """
        Merge two pseudojets to form a new pseudojet, moving the
        exising pseudojets to the back of the ints/floats lists
        Also remove the redundant rows from the embedding space

        Parameters
        ----------
        pseudojet_index1 : int
            index of the first input pseudojet
        pseudojet_index2 : int
            index of the second input pseudojet
        distance2 : float
            distance squared between them.
        """
        replace_index, remove_index = sorted([pseudojet_index1, pseudojet_index2])
        new_pseudojet_ints, new_pseudojet_floats = self._combine(remove_index, replace_index,
                                                                 distance2)
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
        # remove from the affinity and eiegnspace
        self._eigenspace = np.delete(self._eigenspace, remove_index, axis=0)
        self._affinity = np.delete(self._affinity, remove_index, axis=0)
        self._affinity = np.delete(self._affinity, remove_index, axis=1)
        self._distances2 = np.delete(self._distances2, remove_index, axis=0)
        self._distances2 = np.delete(self._distances2, remove_index, axis=1)

    def _remove_pseudojet(self, pseudojet_index):
        """
        Remove a pseudojet from the currently_avalible and move it to the back
        of the ints/floats lists.
        Also remove the redundant rows from the embedding space

        Parameters
        ----------
        pseudojet_index : int
            index to remove

        """
        # move the first pseudojet to the back without replacement
        pseudojet_ints = self._ints.pop(pseudojet_index)
        pseudojet_floats = self._floats.pop(pseudojet_index)
        self._ints.append(pseudojet_ints)
        self._floats.append(pseudojet_floats)
        # remove from the affinity and eiegnspace
        self._eigenspace = np.delete(self._eigenspace, pseudojet_index, axis=0)
        self._affinity = np.delete(self._affinity, pseudojet_index, axis=0)
        self._affinity = np.delete(self._affinity, pseudojet_index, axis=1)
        # delete the row and column
        self._distances2 = np.delete(self._distances2, pseudojet_index, axis=0)
        self._distances2 = np.delete(self._distances2, pseudojet_index, axis=1)
        # one less pseudojet avalible
        self.currently_avalible -= 1
        self.root_jetInputIdxs.append(pseudojet_ints[self._InputIdx_col])

    def _recalculate_one(self, remove_index, replace_index):
        """
        Recalculate all the distances involving one pseudojet
        Also update the embedding space

        Parameters
        ----------
        remove_index : int
            index of the jet that will be removed (moved to the back)
            after joining
        replace_index : int
            index of the jet that will become the index of the combined jet
            (current jet will be moved to the back)

        """
        # delete the larger index keep the smaller index
        assert remove_index > replace_index
        # delete the first row and column of the merge
        self._distances2 = np.delete(self._distances2, (remove_index), axis=0)
        self._distances2 = np.delete(self._distances2, (remove_index), axis=1)
        # calculate the physical distance of the new point from all original points
        # floats and ints will have been updated already in _mearge_pseudojets
        new_position = self._floats[replace_index]
        # since we take rows out of the eigenspace the
        # laplacien also needs to get corrispondingly smaller
        new_distances2 = np.fromiter((self.physical_distance2(self._floats[row], new_position)
                                      for row in range(self.currently_avalible)),
                                     dtype=float)
        if self.beam_particle:
            # then add in one more index for the beam partical
            new_distances2 = np.append(new_distances2, self.beam_distance2(new_position))
        # from this get a new line of the laplacien
        new_laplacien = -self.calculate_affinity(new_distances2)
        new_laplacien[replace_index] = 0.
        new_laplacien[replace_index] = -np.sum(new_laplacien)
        if self.Laplacien == 'symmetric':
            self.alt_diag = np.delete(self.alt_diag, remove_index)
            new_alt_diag = np.sum(new_laplacien)**(-0.5)
            self.alt_diag[replace_index] = new_alt_diag
            new_laplacien = self.alt_diag * (new_laplacien * new_alt_diag)
        # remove from the eigenspace and the affinity
        self._eigenspace = np.delete(self._eigenspace, (remove_index), axis=0)
        self._affinity = np.delete(self._affinity, (remove_index), axis=0)
        self._affinity = np.delete(self._affinity, (remove_index), axis=1)
        # and make its position in vector space
        new_position = np.dot(self._eigenspace.T, new_laplacien)
        self._eigenspace[replace_index] = new_position
        # get the new disntance in eigenspace
        new_distances2 = np.sum((self._eigenspace - new_position)**2, axis=1)
        if self.ExpofPTPosition == 'eigenspace':
            exponent = 2 * self.ExpofPTMultiplier
            pt_here = self._floats[replace_index][self._PT_col]**exponent
            pt_factor = np.fromiter((min(row[self._PT_col]**exponent, pt_here)
                                     for row in self._floats[:self.currently_avalible]),
                                    dtype=float)
            new_distances2[:self.currently_avalible] *= pt_factor
        if self.beam_particle:
            new_distances2[replace_index] = np.inf
        else:
            new_distances2[replace_index] = self.DeltaR**2
        self._distances2[replace_index] = new_distances2

    def _step_assign_parents(self):
        """
        Take a single step to join pseudojets

        Returns
        -------
        removed : int
            index of the pseudojet that is now at the back
        """
        beam_index = self.currently_avalible
        # now find the smallest distance
        row, column = np.unravel_index(np.argmin(self._distances2), self._distances2.shape)
        removed = None
        if row == column:
            if self.beam_particle:
                raise RuntimeError("A jet with a beam particle should" +
                                   " never have a minimal diagonal")
            self._remove_pseudojet(row)
            removed = row
        elif self.beam_particle and row == beam_index:
            # the column merged with the beam
            self._remove_pseudojet(column)
            removed = column
        elif self.beam_particle and column == beam_index:
            # the row merged with the beam
            self._remove_pseudojet(row)
            removed = row
        else:
            self._merge_pseudojets(row, column, self._distances2[row, column])
        return removed


class Splitting(Spectral):
    """ An extention of Spectral jets to cluster the jets in a divisive, simpler manner """
    # list the params with default values
    param_list = {'NumEigenvectors': np.inf,
                  'ExpofPTPosition': 'input', 'ExpofPTMultiplier': 0,
                  'AffinityType': 'exponent', 'AffinityCutoff': None,
                  'Laplacien': 'unnormalised',
                  'PhyDistance': 'angular'}
    permited_values = {'NumEigenvectors': [Constants.numeric_classes['nn'], np.inf],
                       'ExpofPTPosition': ['input', 'eigenspace'],
                       'ExpofPTMultiplier': Constants.numeric_classes['rn'],
                       'AffinityType': ['linear', 'exponent', 'exponent2', 'inverse'],
                       'AffinityCuttoff': [None, ('knn', Constants.numeric_classes['nn']),
                                           ('distance', Constants.numeric_classes['pdn'])],
                       'Laplacien': ['unnormalised', 'symmetric'],
                       'PhyDistance': ['angular', 'normed', 'Luclus', 'invarient']}
    def __init__(self, eventWise=None, dict_jet_params=None, **kwargs):
        """
        Class constructor

        Parameters
        ----------
        eventWise : EventWise
            data file for inputs
        selected_index : int
            event number to use, can already be set in the eventWise
            (Default; None)
        jet_name : string
            name to prefix the jet properties with when saving
            (Default; "PseudoJet")
        from_PseudoRapidity : bool
            Use pseudorapidity instead or rapidity
            (Default; False)
        int_floats : tuple of 2d array likes
            Predefined int and float tables.
            If not given they will be constructed from the eventWise.
            (Default; None)
        dict_jet_params : dict (optional)
            Settings for jet clustering. If not given defaults are used.
        root_jetInputIdxs : list of ints (optional)
            List of the indices of the roots of each jet.
        assign : bool (optional)
            Should the jets eb clustered immediatly?
            (Default; False)
        """
        self.StoppingCondition = 'standard'  # this is a property used by Spectral Jets
        self.DeltaR = 1.  # thsi si required for defining the unused beam distances in
        # _define_physical_distance
        if dict_jet_params is not None:
            kwargs['dict_jet_params'] = dict_jet_params
        super().__init__(eventWise, **kwargs)

    def _calculate_distances(self):
        """ Calculate all distances between avalible pseudojets """
        n_distances = self.currently_avalible
        # this clusterign mechanism doesn't actually use distances,
        # but other functions expect thier presence
        self._distances2 = np.empty((n_distances, n_distances))
        if n_distances < 2:
            self._affinity = np.empty((0, 0))
            return
        # to start with create a 'normal' distance measure
        # this can be based on any of the three algorithms
        physical_distances2 = np.zeros((n_distances, n_distances))
        # for speed, make local variables
        pt_col = self._PT_col
        rap_col = self._Rapidity_col
        phi_col = self._Phi_col
        for row in range(self.currently_avalible):
            for column in range(self.currently_avalible):
                if column < row:
                    distance2 = physical_distances2[column, row]  # the matrix is symmetric
                elif self._floats[row][pt_col] == 0:
                    distance2 = 0  # soft radation might as well be at 0 distance
                elif column == row:
                    # not used
                    continue
                else:
                    distance2 = self.physical_distance2(self._floats[row], self._floats[column])
                physical_distances2[row, column] = distance2
        np.fill_diagonal(physical_distances2, 0.)
        # now we are in posessio of a standard distance matrix for all points,
        # we can make an affinity calculation, which we will need
        self._affinity = self.calculate_affinity(physical_distances2)
        # a graph laplacien can be calculated
        np.fill_diagonal(self._affinity, 0.)  # the affinity may have problems on the diagonal

    def _calculate_eigenspace(self):
        """
        Calculate the embedding of the currently_avalible pseudojets in eignspace
        """
        if np.sum(np.abs(self._affinity)) == 0.:
            # everything is seperated
            self.root_jetInputIdxs += [row[self._InputIdx_col] for row in
                                       self._ints[:self.currently_avalible]]
            self.currently_avalible = 0
            return
        diagonal = np.diag(np.sum(self._affinity, axis=1))
        if self.Laplacien == 'unnormalised':
            laplacien = diagonal - self._affinity
        elif self.Laplacien == 'symmetric':
            laplacien = diagonal - self._affinity
            self.alt_diag = np.diag(diagonal)**(-0.5)
            self.alt_diag[np.diag(diagonal) == 0] = 0.
            diag_alt_diag = np.diag(self.alt_diag)
            laplacien = np.matmul(diag_alt_diag, np.matmul(laplacien, diag_alt_diag))
        else:
            raise NotImplementedError(f"Don't have a laplacien {self.Laplacien}")
        # get the eigenvectors (we know the smallest will be identity)
        try:
            eigenvalues, eigenvectors = scipy.linalg.eigh(laplacien,
                                                          eigvals=(1, self.NumEigenvectors))
        except (ValueError, TypeError):
            # sometimes there are fewer eigenvalues avalible
            # just take waht can be found
            eigenvalues, eigenvectors = scipy.linalg.eigh(laplacien)
            eigenvalues = eigenvalues[1:]
            eigenvectors = eigenvectors[:, 1:]
        except Exception as e:
            # display whatever caused this
            print(f"Exception while processing event {self.eventwise.selected_index}")
            print(f"With jet params; {self.jet_parameters}")
            print(e)
            self.root_jetInputIdxs += [row[self._InputIdx_col] for row in
                                       self._ints[:self.currently_avalible]]
            self.currently_avalible = 0
            return
        self.eigenvalues.append(eigenvalues.tolist())
        # at the start the eigenspace positions are the eigenvectors
        self._eigenspace = eigenvectors

    def _recalculate_one(self, remove_index, replace_index):
        """
        Redundant function in this method, present for interface consistancy.
        """
        pass  # do nothing on a single merge

    def _merge_complete_jets(self, list_input_indices):
        """
        Merge multple pseudojets to form many new pseudojets, moving the
        formed pseudojets to the back of the ints/floats lists

        Parameters
        ----------
        list_input_indices : list of list of int
            indices of the pseudojets to merge, sorted by pseudojet to be formed
        """
        array_input_indices = awkward.fromiter(list_input_indices)
        for input_indices in array_input_indices:
            self._merge_complete_jet(input_indices)
            # now fix the remaining indices to merge
            for idx in sorted(input_indices, reverse=True):
                # going through each index that was removed,
                # shift everythign infront of it back by one
                mask = array_input_indices > idx
                array_input_indices[mask] = array_input_indices[mask] -1

    def _merge_complete_jet(self, input_indices):
        """
        Merge multple pseudojets to form a new pseudojet, moving the
        formed pseudojet to the back of the ints/floats lists

        Parameters
        ----------
        input_indices : list of list of int
            indices of the pseudojets to merge

        """
        # for now just merge in pairs
        input_indices = sorted(input_indices)
        replace = input_indices[0]
        for remove in input_indices[:0:-1]:
            self._merge_pseudojets(replace, remove, 0)
        self._remove_pseudojet(replace)

    def _step_assign_parents(self, eigenvector_num=0):
        """
        Take a single step to join pseudojets

        Parameters
        ----------
        eigenvector_num : int
            The eigenvector to use while taking this step
             (Default value = 0)

        Returns
        -------
        order : list of ints
            The order the particles take when ordred by the eigenvector
        outcomes : array of floats
            The score for splitting at each step in the order
        flip_point : int
            The point at which the inside of the cluster becomes the end of the list
            rather than the begining
        splits : list of ints
            the outcomes at which the group should eb divided into seperate groups
        """
        # get the order of elements
        order = np.argsort(self._eigenspace[:, eigenvector_num])
        n_trials = self.currently_avalible - 1
        # vector to store the results of calculating the ratiocut/ncut score
        outcomes = np.empty(n_trials, dtype=float)
        # if the laplacien is symmetric this is needed to find the flip point
        sum_affinity = np.sum(self._affinity)
        # flip point is the element from which the 'inside' group is the trailing elements
        post_flip = False  # switch for having found the flip point
        flip_point = None  # index of the flip point
        for trial in range(1, self.currently_avalible):  # start at 1, as in after the 0th element
            in_group = order[:trial]  # in the jet
            out_group = order[trial:]  # yet to be sorted
            if post_flip:  # in group and out group get switched after flip point has been reached
                in_group, out_group = out_group, in_group
            # numerator is the affinities that cross groups
            numerator = np.sum(self._affinity[out_group][:, in_group])
            if self.Laplacien == 'unnormalised':  # ratiocut style
                post_flip = trial > 0.5*n_trials  # check if we are post flip
                denominator = abs(trial - post_flip*n_trials)
                if flip_point is None and post_flip:
                    flip_point = trial
            elif self.Laplacien == 'symmetric':  # ncut style
                denominator = np.sum(self._affinity[in_group])
                if not post_flip:  # calculation not easly generalised
                    if denominator > 0.5*sum_affinity:
                        # if reached we have found the flip point
                        flip_point = trial
                        post_flip = True
                        # fix the denominator
                        denominator = sum_affinity - denominator
            # store the result at this step
            outcomes[trial-1] = numerator/denominator
        if not post_flip:  # we never found a flip point
            # (can happen when the last affinity is v large)
            flip_point = trial + 1
        outcomes[np.isinf(outcomes)] = np.nan  # treat infinities as nan
        if np.all(np.isnan(outcomes)):
            # everything that's left is BG
            self.currently_avalible = 0
            return order, outcomes, flip_point, np.nan
        # find the minima, the groups should be formed like
        # minima_a <= indices_in_group < minima_b
        # 2 <= 2,3,4 < 5
        # the element at the minima goes in the group with largest endpoint
        splits = [i+1 for i, value in enumerate(outcomes[1:-1])
                  if outcomes[i] > value and outcomes[i+2] > value]
        splits = [0] + splits + [self.currently_avalible]
        self._merge_complete_jets([order[start:end] for start, end in zip(splits, splits[1:])])
        self._calculate_eigenspace()  # now the eigenspace needs recalculating
        return order, outcomes, flip_point, splits

    def plt_assign_parents(self, save_prefix=None, eigenvector_num=0, steps_required=np.inf):
        """
        Join pseudojets until all avalible psseudojets are taken.
        Also plot the process

        Parameters
        ----------
        save_prefix : string
            If the plots should be saved to file use this vaiable to supply the start
            of the save path. If None the plot is shown on the screen
             (Default value = None)
        eigenvector_num : int
            The eigenvector to use for clustering.
             (Default value = 0)
        steps_required : int
            Max number of iterations to plot
             (Default value = np.inf)
        """
        step_no = 0
        while self.currently_avalible:
            if eigenvector_num >= self._eigenspace.shape[1]:
                return # cant get this eigenvector
            # precalculate some quantities, becuase the step will change them
            prior_eigenvectors = np.copy(self._eigenspace)
            prior_eigenvector = prior_eigenvectors[:, eigenvector_num]
            previously_avalible = self.currently_avalible
            # work out which points are b decendants
            b_decendants = np.fromiter(FormShower.descendant_idxs(self.eventWise,
                                                                  *self.eventWise.BQuarkIdx),
                                       dtype=int)
            is_decendant = np.fromiter((idx in b_decendants for
                                        idx in self.eventWise.JetInputs_SourceIdx),
                                       dtype=bool)
            b_mask = np.fromiter((is_decendant[idx] for
                                  idx in self.InputIdx[:self.currently_avalible]), dtype=bool)
            pts = np.fromiter((row[self._PT_col] for
                               row in self._floats[:self.currently_avalible]), dtype=float)
            rap = np.array([self._floats[row][self._Rapidity_col] for row
                            in range(self.currently_avalible)])
            phi = np.array([self._floats[row][self._Phi_col] for row
                            in range(self.currently_avalible)])
            # take a step
            order, outcomes, flip_point, splits = self._step_assign_parents(eigenvector_num)
            n_trials = len(outcomes)
            outcomes[np.isinf(outcomes)] = np.nan  # inifinities are hard for plotting
            # use the split that minimises the outcome
            fig = plt.figure(figsize=(10, 7))
            ax0 = plt.subplot(221)
            ax1 = plt.subplot(222)
            ax2 = plt.subplot(223)
            ax3 = plt.subplot(224, sharex=ax1)
            # some useful ranges for plotting
            max_out = np.nanmax(outcomes)
            #ax0, ax1, ax2, ax3 = ax_ar))
            # ax1 and ax3 are the points in order
            ax1.set_xlabel("Split position")
            ax1.set_ylabel("Cut score")
            ordered_pts = pts[order]
            colour_args = {'vmax':np.max(pts), 'vmin':np.min(pts), 'cmap': 'plasma'}
            # scatter plots show disagreement here between thsi and plot_tags
            ordered_b_mask = b_mask[order]
            # the -1 is for zero indexing, the -0.5 is for ebing between second to last and last
            split_colour = np.array([1., 0., 0., 0.5])
            flip_colour = np.array([0., 0., 1., 0.5])
            ax1.scatter(np.linspace(0.5, previously_avalible-1-0.5, n_trials),
                        outcomes, c='black', marker='|')
            # problem may be that outccomes can be nan,
            # create a list of particle psotions between the split conditions
            # that is never nan
            filled_outcomes = np.zeros(previously_avalible)
            filled_outcomes[1:] = outcomes
            filled_outcomes[:-1] += outcomes
            filled_outcomes[1:-1] *= 0.5
            for i, out in enumerate(filled_outcomes):
                if np.isnan(out):
                    out = next(np.nanmean(outcomes[i-width:i+width]) for
                               width in range(previously_avalible)
                               if not np.all(np.isnan(outcomes[i-width:i+width])))
                    filled_outcomes[i] = out
            ax1.scatter(np.arange(previously_avalible), filled_outcomes,
                        c=ordered_pts, **colour_args)
            ax1.scatter(np.where(ordered_b_mask)[0], filled_outcomes[ordered_b_mask],
                        c=np.zeros((1, 4)), edgecolors=TRUTH_COLOUR, marker='o')
            for split in splits[1:-1]:
                split += 0.5  # splits happen at test points and start at 0
                ax1.vlines(split, 0, max_out, colors=split_colour)
                ax1.text(split+0.1, 0.5*max_out, "Split location",
                         rotation='vertical', c=split_colour)
            ax1.vlines(flip_point-0.5, 0, max_out, colors=flip_colour, ls='--')
            ax1.text(flip_point-0.4, 0.5*max_out, "Flip location",
                     rotation='vertical', c=flip_colour)
            ax3.set_xlabel("Eigenvector element")
            # plot all the other eigenvector values in grey
            for i in range(prior_eigenvectors.shape[1]):
                if i == eigenvector_num:
                    continue
                alpha = 1/(i+2)
                shape = ['v', 's', '*', 'D', 'P', 'X'][i%6]
                ax3.scatter(range(previously_avalible), prior_eigenvectors[order, i],
                            c='black', alpha=alpha, marker=shape, label=i)
            # plot the focal ones in colour
            points = ax3.scatter(range(previously_avalible), prior_eigenvector[order],
                                 c=ordered_pts, **colour_args, label=eigenvector_num)
            ax3.scatter(np.where(ordered_b_mask)[0], prior_eigenvector[order][ordered_b_mask],
                        c=np.zeros((1, 4)), edgecolors=TRUTH_COLOUR, marker='o')
            ax3.legend()
            min_eig, max_eig = np.min(prior_eigenvectors), np.max(prior_eigenvector)
            for split in splits[1:-1]:
                split += 0.5
                ax3.vlines(split, min_eig, max_eig, colors=split_colour)
                ax3.text(split+0.1, 0.5*(max_eig + min_eig), "Split location",
                         rotation='vertical', c=split_colour)
            ax3.vlines(flip_point-0.5, min_eig, max_eig, colors=flip_colour, ls='--')
            ax3.text(flip_point-0.4, 0.5*(max_eig + min_eig), "Flip location",
                     rotation='vertical', c=flip_colour)
            # ax2 is the point sin the pt eta plain
            ax2.set_xlabel("Rapidity")
            ax2.set_ylabel("$\\phi$")
            ax2.scatter(rap, phi, c=pts, **colour_args)
            arrow_style = '<-'
            jet_map = matplotlib.cm.get_cmap('jet')
            colours = [jet_map(i/len(splits)) for i in range(len(splits))]
            switch_colour = (0.5, 0.5, 0.5, 1.)
            split_num, split_end = 0, splits[1]
            for i in range(n_trials):
                start_idx = order[i]
                end_idx = order[i+1]
                # split ends may start at 1,
                # which represents a group that splits at the second cut
                # which would take particles 0, and 1,
                # this should swtich when i=1
                switch = i >= split_end
                if switch:
                    split_num += 1
                    split_end = splits[split_num+1]
                    colour = switch_colour
                else:
                    colour = colours[split_num]
                rap_cross, sign = PlottingTools.find_crossing_point(rap[start_idx], phi[start_idx],
                                                                    rap[end_idx], phi[end_idx])
                if rap_cross is not None:
                    ax2.plot([rap[start_idx], rap_cross], [phi[start_idx], sign*np.pi],
                             c=colour)
                    ax2.annotate('', (rap_cross, -sign*np.pi), (rap[end_idx], phi[end_idx]),
                                 arrowprops={'arrowstyle':arrow_style, 'color':colour})
                else:
                    ax2.annotate('', (rap[start_idx], phi[start_idx]), (rap[end_idx], phi[end_idx]),
                                 arrowprops={'arrowstyle':arrow_style, 'color':colour})
            plot_tags(self.eventWise, ax=ax2)
            # discribe the jet and stick the colour bar on ax 0
            PlottingTools.discribe_jet(properties_dict=self.jet_parameters, ax=ax0,
                                       additional_text=f"Step {step_no}\n" +
                                                       f"Eigenvector {eigenvector_num}")
            colorbar = plt.colorbar(points, ax=ax0)
            colorbar.set_label("Particle $p_T$")
            fig.tight_layout()
            if save_prefix is None:
                plt.show()
            else:
                fig_name = f"images/splitting/{save_prefix}_{self.Laplacien}" +\
                           f"_step{step_no}_ev{eigenvector_num}.png"
                plt.savefig(fig_name)
            plt.close('all')
            step_no += 1
            if step_no > steps_required:
                return


class SpectralMean(Spectral):
    """ A slightly simplified varient of Spectral where new positions in the embedding space
    are simply geometric means"""
    def _recalculate_one(self, remove_index, replace_index):
        """
        Recalculate all the distances involving one pseudojet
        Also update the embedding space

        Parameters
        ----------
        remove_index : int
            index of the jet that will be removed (moved to the back)
            after joining
        replace_index : int
            index of the jet that will become the index of the combined jet
            (current jet will be moved to the back)

        """
        # delete the larger index keep the smaller index
        assert remove_index > replace_index
        # delete the first row and column of the merge
        self._distances2 = np.delete(self._distances2, (remove_index), axis=0)
        self._distances2 = np.delete(self._distances2, (remove_index), axis=1)
        # and make its position in eigenspace
        new_position = (self._eigenspace[[remove_index]] + self._eigenspace[[replace_index]])*0.5
        # CHanged -> simply delete the eigenspace line
        self._eigenspace = np.delete(self._eigenspace, remove_index, axis=0)
        self._eigenspace[replace_index] = new_position
        # get the new disntance in eigenspace
        new_distances2 = np.sum((self._eigenspace - new_position)**2, axis=1)
        if self.ExpofPTPosition == 'eigenspace':
            exponent = 2 * self.ExpofPTMultiplier
            pt_here = self._floats[replace_index][self._PT_col]**exponent
            pt_factor = np.fromiter((min(row[self._PT_col]**exponent, pt_here)
                                     for row in self._floats[:self.currently_avalible]),
                                    dtype=float)
            new_distances2[:self.currently_avalible] *= pt_factor
        if self.beam_particle:
            new_distances2[replace_index] = np.inf
        else:
            new_distances2[replace_index] = self.DeltaR**2
        self._distances2[replace_index] = new_distances2


class SpectralFull(Spectral):
    """ A computationally exspensive varient of Spectral where the whole embedding space
    is recalculated at every step """
    def _recalculate_one(self, remove_index, replace_index):
        """
        Recalculate all the distances involving one pseudojet
        Also update the embedding space

        Parameters
        ----------
        remove_index : int
            index of the jet that will be removed (moved to the back)
            after joining
        replace_index : int
            index of the jet that will become the index of the combined jet
            (current jet will be moved to the back)

        """
        self._calculate_eigenspace()


def filter_obs(eventWise, existing_idx_selection):
    """
    Filter particle in an eventWise data structure to select only the observable particles
    those being the ones that left a trace in the detector.

    Parameters
    ----------
    eventWise : EventWise
        data structure
    existing_idx_selection : list of ints
        any preexisting selection results

    Returns
    -------
    new_selection : list of ints
        the indices of the selected particles
    """
    assert eventWise.selected_index is not None
    has_track = eventWise.Particle_Track[existing_idx_selection.tolist()] >= 0
    has_tower = eventWise.Particle_Tower[existing_idx_selection.tolist()] >= 0
    observable = np.logical_or(has_track, has_tower)
    new_selection = existing_idx_selection[observable]
    return new_selection


def filter_ends(eventWise, existing_idx_selection):
    """
    Filter particle in an eventWise data structure to select only the particles
    that have not decayed in the end state.

    Parameters
    ----------
    eventWise : EventWise
        data structure
    existing_idx_selection : list of ints
        any preexisting selection results

    Returns
    -------
    new_selection : list of ints
        the indices of the selected particles
    """
    assert eventWise.selected_index is not None
    is_end = [len(c) == 0 for c in
              eventWise.Children[existing_idx_selection.tolist()]]
    new_selection = existing_idx_selection[is_end]
    return new_selection


def filter_pt_eta(eventWise, existing_idx_selection, min_pt=.5, max_eta=2.5):
    """
    Filter particle in an eventWise data structure that have enough pt
    and a small enough barrel angle.

    Parameters
    ----------
    eventWise : EventWise
        data structure
    existing_idx_selection : list of ints
        any preexisting selection results
    min_pt : float
        smallest pt permissable in a particle
        (Default value = .5)
    max_eta : float
        larges abs value of pseudorapidity in a particle
        (Default value = 2.5)

    Returns
    -------
    new_selection : list of ints
        the indices of the selected particles
    """
    assert eventWise.selected_index is not None
    # filter PT
    sufficient_pt = eventWise.PT[existing_idx_selection.tolist()] > min_pt
    updated_selection = existing_idx_selection[sufficient_pt]
    if "Pseudorapidity" in eventWise.columns:
        pseudorapidity_here = eventWise.Pseudorapidity[updated_selection.tolist()]
    else:
        theta_here = Components.ptpz_to_theta(eventWise.PT[updated_selection.tolist()],
                                              eventWise.Pz[updated_selection.tolist()])
        pseudorapidity_here = Components.theta_to_pseudorapidity(theta_here)
    pseudorapidity_choice = np.abs(pseudorapidity_here) < max_eta
    updated_selection = updated_selection[pseudorapidity_choice.tolist()]
    return updated_selection


def create_jetInputs(eventWise, filter_functions=[filter_obs, filter_pt_eta], batch_length=1000):
    """
    Add to the eventWise a set of particles prefixed by JetInputs
    which pass all criteria required to be used in jet clustering.

    Parameters
    ----------
    eventWise : EventWise
        data structure
    filter_functions : list of callabels
        callabels with the same signature as filter_pt_eta
        that should reduce down the list of list of particle indices
        to those that are sutable as jet inputs
        (Default value = [filter_obs, filter_pt_eta])
    batch_length : int
        how many particles to checks
        (Default value = 1000)

    """
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
            print(f"{100*event_n/n_events}%", end='\r', flush=True)
        eventWise.selected_index = event_n
        idx_selection = np.arange(len(eventWise.PT))
        for filter_func in filter_functions:
            idx_selection = filter_func(eventWise, idx_selection)
        contents["JetInputs_SourceIdx"].append(idx_selection)
        mask_here = np.full_like(eventWise.PT, False, dtype=bool)
        mask_here[idx_selection] = True
        mask.append(awkward.fromiter(mask_here))
    mask = awkward.fromiter(mask).astype(bool)
    eventWise.selected_index = None
    for name, source_name in zip(columns, sources):
        contents[name] += list(getattr(eventWise, source_name)[start_point:end_point][mask])
    contents = {k:awkward.fromiter(v) for k, v in contents.items()}
    eventWise.append(**contents)
    return end_point == n_events


def produce_summary(eventWise, to_file=True):
    """
    Create a csv of the jet inputs for one event.
    Can be used to sent to other programs or collaborators.

    Parameters
    ----------
    eventWise : EventWise
        file containing data
    to_file : bool
        should this be written straight to disk
        instead of returing as a string
        (Default value = True)

    Returns
    -------


    """
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


def run_FastJet(eventWise, DeltaR, ExpofPTMultiplier, jet_name="FastJet", use_pipe=True):
    """
    Run fastjet on one event. Data not written to eventWise.

    Parameters
    ----------
    eventWise : EventWise
        Input data file
    DeltaR: float
        stoppign parameter for clustering
    ExpofPTMultiplier : int
        should be -1, 0, or 1 depending if anti-kt, cambridge aachen or
        kt clustering is required
    jet_name: string
        Prefix name for the jet in eventWise
        (Default value = "FastJet")
    use_pipe : bool
        Should the data be piped to fastjet, rather than reading and writing from disk?
        (Default value = True)

    Returns
    -------
    fastjets : Traditional
        Traditional jet objects created

    """
    assert eventWise.selected_index is not None
    if ExpofPTMultiplier == -1:
        # antikt algorithm
        algorithm_num = 1
    elif ExpofPTMultiplier == 0:
        algorithm_num = 2
    elif ExpofPTMultiplier == 1:
        algorithm_num = 0
    else:
        raise ValueError(f"ExpofPTMultiplier should be -1, 0 or 1, found {ExpofPTMultiplier}")
    program_name = "./tree_tagger/applyFastJet"
    if use_pipe:
        summary_lines = produce_summary(eventWise, False)
        out = _run_applyfastjet(summary_lines, str(DeltaR).encode(),
                                str(algorithm_num).encode())
        fastjets = Traditional.read_fastjet(out, eventWise, jet_name=jet_name)
        return fastjets
    produce_summary(eventWise)
    subprocess.run([program_name, str(DeltaR), str(algorithm_num), eventWise.dir_name])
    fastjets = Traditional.read_fastjet(eventWise.dir_name, eventWise=eventWise, jet_name=jet_name)
    return fastjets


def _run_applyfastjet(input_lines, DeltaR, algorithm_num, program_path="./tree_tagger/applyFastJet", tries=0):
    """
    Run applyfastjet, sending the provided input lines to stdin
    Helper function for run_FastJet

    Parameters
    ----------
    input_lines : list of byte array
        contents of the input as byte arrays
    DeltaR : float
        stopping parameter for clustering
    algorithm_num : int
        number indicating the algorithm to use
    program_path : string
        path to call the program at
        (Default value = "./tree_tagger/applyFastJet")
    tries : int
        number of tries with this input
        (Default value = 0)


    Returns
    -------
    output_lines : list of bytes
        returned from fastjet

    """
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
            #st()
            raise RuntimeError("Subrocess problems")
        # recursive call
        output_lines = run_applyfastjet(input_lines, DeltaR, algorithm_num, program_path, tries)
    return output_lines


def cluster_multiapply(eventWise, cluster_algorithm, cluster_parameters={},
                       jet_name=None, batch_length=100, silent=False):
    """
    Apply a clustering algorithm to many events.

    Parameters
    ----------
    eventWise : EventWise
        data file with inputs, results are also written here
    cluster_algorithm: callable
        function or class that will create the jets
    cluster_parameters : dict
        dictionary of input parameters for clustering settings
        (Default value = {})
    jet_name : string
        Prefix name for the jet in eventWise
        (Default value = None)
    batch_length : int
        numebr of events to process
        (Default value = 100)
    silent : bool
        should print statments indicating progrss be suppressed?
        useful for running in parallel
        (Default value = False)

    Returns
    -------
    : bool
        All events in the eventWise have been clustered

    """
    check_hyperparameters(cluster_algorithm, cluster_parameters)
    if jet_name is None and 'jet_name' in cluster_parameters:
        jet_name = cluster_parameters['jet_name']
    elif jet_name is None:
        for name, algorithm in cluster_classes.items():
            if algorithm == cluster_algorithm:
                jet_name = name
                break
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
    checked = False
    has_eigenvalues = 'NumEigenvectors' in cluster_parameters
    if has_eigenvalues:
        eigenvalues = []
    for event_n in range(start_point, end_point):
        if event_n % 100 == 0 and not silent:
            print(f"{100*event_n/n_events}%", end='\r', flush=True)
        eventWise.selected_index = event_n
        if len(eventWise.JetInputs_PT) == 0:
            continue  # there are no observables
        jets = cluster_algorithm(eventWise, **cluster_parameters)
        if has_eigenvalues:
            eigenvalues.append(awkward.fromiter(jets.eigenvalues))
        jets = jets.split()
        if not checked and len(jets) > 0:
            assert jets[0].check_params(eventWise), f"Jet parameters don't match recorded parameters for {jet_name}"
            checked = True
        updated_dict = jet_class.create_updated_dict(jets, jet_name, event_n, eventWise, updated_dict)
    updated_dict = {name: awkward.fromiter(updated_dict[name]) for name in updated_dict}
    if has_eigenvalues:
        updated_dict[jet_name + "_Eigenvalues"] = awkward.fromiter(eigenvalues)
    eventWise.append(**updated_dict)
    return end_point == n_events


# track which classes in this module are cluster classes
cluster_classes = ["Traditional", "Spectral", "Splitting", "SpectralFull", "SpectralMean"]
# track which things are valid inputs to multiapply
multiapply_input = {"Fast": run_FastJet, "Home": Traditional}
for name in cluster_classes:
    multiapply_input[name] = locals()[name]


def check_hyperparameters(cluster_class, params):
    """
    Check the clustering parameters chosen are valid for the clustering class to be used.
    Raises an error if there is a problem.

    Parameters
    ----------
    cluster_class : class
        clusteirng class defining requirements
    params : dict
        parameters to be checked

    """
    if isinstance(cluster_class, str):
        cluster_class = cluster_classes[cluster_class]
    permitted = cluster_class.permited_values
    error_str = f"In {cluster_class.__name__} {{}} is not a permitted value for {{}}. Permitted value are {{}}"
    for name, opts in permitted.items():
        try:
            value = params[name]
        except KeyError:
            continue  # the default will be used
        try:
            if value in opts:
                continue  # no problem
        except TypeError:
            pass
        try:
            if Constants.is_numeric_class(value, opts):
                continue  # no problem
        except (ValueError, TypeError):
            pass
        if name == 'AffinityCutoff':
            primary, secondary = value
            try:
                secondary_options = next(s for p, s in opts if p == primary)
            except StopIteration:
                raise ValueError(error_str.format(value, name, opts))
            if Constants.is_numeric_class(secondary, secondary_options):
                continue  # no problem
        if isinstance(opts, list):
            found_correct = False
            for opt in opts:
                try:
                    if Constants.is_numeric_class(value, opt):
                        found_correct = True
                        break
                except (ValueError, TypeError):
                    pass
            if found_correct:
                continue  # no problem
        # if we have yet ot hit a continue statment then this option is not valid
        raise ValueError(error_str.format(value, name, opts))


def get_jet_params(eventWise, jet_name, add_defaults=False):
    """
    Given an eventwise in which a jet was written return it's settings.

    Parameters
    ----------
    eventWise : EventWise
        data structure with jets
    jet_name : string
        Prefix name of jet we are interested in
    add_defaults : bool
        should class default be put inplace on unspecified settings?
        (Default value = False)

    Returns
    -------
    columns : dict
        dictionary with keys being parameter names and values being parameter values

    """
    prefix = jet_name + "_"
    trim = len(prefix)
    columns = {name[trim:]: getattr(eventWise, name) for name in eventWise.hyperparameter_columns
               if name.startswith(prefix)}
    if add_defaults:
        if jet_name.startswith("SpectralMean"):
            defaults = SpectralMean.param_list
        elif jet_name.startswith("Spectral"):
            defaults = Spectral.param_list
        else:
            defaults = Traditional.param_list
        not_found = {name: defaults[name] for name in defaults
                     if name not in columns}
        columns = {**columns, **not_found}
    return columns


def get_jet_names(eventWise):
    """
    Given and eventwise get the Prefix names of all jets inside it.

    Parameters
    ----------
    eventWise : EventWise
        data structure with jets

    Returns
    -------
    possibles : list of strings
        names of jets inside this eventWise

    """
    # grab an ending fromt he pseudojet int columns
    ending = next(PseudoJet.int_columns).split('_', 1)[1]
    possibles = [name.split('_', 1)[0] for name in eventWise.columns if name.endswith(ending)]
    # the beggining could be any of the multiapply inputs
    # with the word Jet put on the end
    possibles = [name  for name in possibles if name.split('Jet', 1)[0] in multiapply_input]
    return possibles


def plot_spider(ax, colour, body, body_size, leg_ends, leg_size):
    """
    Given an axis and the locaton of a body and it's legs plot a 'spider'

    Parameters
    ----------
    ax : pyplot axis
        where to plot the spider
    body : list like of floats
        [x, y] coordinates of the spider's body
    body_size : float
        how large to maek the body
    legs : list like of floats
        [x, y] coordinates of the spider's legs
    legs_size : float
        how thick to maek the legs
    colour : matplotlib colour specifier
        colour of the spider
    """
    leg_size = np.clip(np.sqrt(leg_size), 0.2, None)
    for end, size in zip(leg_ends, leg_size):
        line = np.vstack((body, end))
        # line[body/end, x/y]
        # work out if this leg crossed the edge
        #         body y - leg y
        x_cross, sign = PlottingTools.find_crossing_point(*line[0], *line[1])
        if x_cross is not None:
            ax.plot([line[0, 0],  x_cross], [line[0, 1], sign*np.pi],
                     c=colour, linewidth=size, alpha=JET_ALPHA)
            ax.plot([line[1, 0],  x_cross], [line[1, 1], -sign*np.pi],
                     c=colour, linewidth=size, alpha=JET_ALPHA)

        else:
            ax.plot(line[:, 0], line[:, 1],
                     c=colour, linewidth=size, alpha=JET_ALPHA)
    ax.scatter(leg_ends[:, 0], leg_ends[:, 1], c=colour, s=leg_size)
    #ax.scatter([body[0]], [body[1]], c='black', marker='o', s=body_size+1)
    #ax.scatter([body[0]], [body[1]], c=[colour], marker='o', s=body_size)

def plot_tags(eventWise, b_decendants=True, ax=None):
    """
    Plot the location of taggin particles and their decendants

    Parameters
    ----------
    eventWise : EventWise
        data file with informaton about tags
    b_decendants : bool
        show location of decendants
         (Default value = True)
    ax : pyplot axis
        axis to plot on
         (Default value = None)

    """
    assert eventWise.selected_index is not None
    if ax is None:
        ax = plt.gca()
    tag_phis = eventWise.Phi[eventWise.TagIndex]
    tag_rapidity = eventWise.Rapidity[eventWise.TagIndex]
    ax.scatter(tag_rapidity, tag_phis, marker='d', c=TRUTH_COLOUR)
    if b_decendants:
        b_decendants = np.fromiter(FormShower.descendant_idxs(eventWise, *eventWise.BQuarkIdx),
                                   dtype=int)
        included = np.fromiter((idx in eventWise.JetInputs_SourceIdx for idx in b_decendants),
                               dtype=bool)
        ax.scatter(eventWise.Rapidity[b_decendants[included].tolist()],
                   eventWise.Phi[b_decendants[included].tolist()],
                   c=[(0., 0., 0., 0.),], edgecolors=TRUTH_COLOUR,
                   linewidths=TRUTH_LINEWIDTH, s=TRUTH_SIZE, marker='o')
        # plot the others invisibly so as to set the rapidity axis with them
        ax.scatter(eventWise.Rapidity[b_decendants[~included]],
                   eventWise.Phi[b_decendants[~included]],
                   c='black', alpha=0.,#TRUTH_COLOUR,
                   s=TRUTH_LINEWIDTH, marker='x')


def plot_cluster(pseudojet, colours, ax=None, spiders=True, pt_text=False, circles=False):
    """
    Plot the results of a jet clustering algorithm in real space

    Parameters
    ----------
    pseudojet : PseudoJet
        the clusters to plot
    colours : list of matplotlib colour objects
        a colour for each jet, or a single string discribing the colour for all jets
    ax : pytplot axis
        axis to plot on
         (Default value = None)
    spiders : bool
        should the gathing of the jets from the inputs be plotted as a spider?
         (Default value = True)
    pt_text : bool
        Should the pt of th ejet be written by it's center?
         (Default value = False)
    circles : bool
        Should a circle be drawn round the center with the stoping parameter?
         (Default value = False)

    """
    if ax is None:
        ax = plt.gca()
    pseudojet.assign_parents()
    pjets = pseudojet.split()
    if isinstance(colours, str):
        colours = [colours for _ in pjets]
    # plot the pseudojets
    for c, pjet in zip(colours, pjets):
        obs_idx = [i for i, child1 in enumerate(pjet.Child1) if child1==-1]
        input_rap = np.array(pjet._floats)[obs_idx, pjet._Rapidity_col]
        input_phi = np.array(pjet._floats)[obs_idx, pjet._Phi_col]
        input_energy = np.array(pjet._floats)[obs_idx, pjet._Energy_col]
        if pt_text:
            ax.text(pjet.Rapidity, pjet.Phi-.1, str(pjet.PT)[:7], c=c)
        if spiders:
            leg_ends = np.vstack((input_rap, input_phi)).T
            plot_spider(ax, c, [pjet.Rapidity, pjet.Phi], pjet.Energy, leg_ends, input_energy)
        else:
            ax.scatter(input_rap, input_phi, s=input_energy, c=c)
        if circles:
            circle = plt.Circle((pjet.Rapidity, pjet.Phi), radius=pjet.DeltaR, edgecolor=c, fill=False)
            ax.add_artist(circle)


def plot_eigenspace(eventWise, event_n, spectral_jet_params, eigendim1, eigendim2, ax=None, spiders=True, spectral_class=SpectralFull):
    """
    Plot the clustering of a spectral jet in the eigenspace embedding

    Parameters
    ----------
    eventWise : EventWise
        data structure from whcih to cluster jets
    event_n : int
        the event number in which to cluster jets
    spectral_jet_params : dict
        dictionary of jet inputs for Spectral
    eigendim1 : int
        index of the first eigendimension to plot
    eigendim2 : int
        index of the second eigendimension to plot
    ax : pyplot axis
        axis to plot on
         (Default value = None)
    spiders : bool
        draw th ejet clustering with spiders?
         (Default value = True)
    spectral_class : class
        Spectral class used to cluster
         (Default value = SpectralFull)
    """
    eventWise.selected_index = event_n
    if ax is None:
        ax = plt.gca()
    #ax.set_facecolor('black')
    ax.grid(c='dimgrey', ls='--')
    # create inputs if needed
    if "JetInputs_Energy" not in eventWise.columns:
        filter_funcs = [filter_ends, filter_pt_eta]
        create_jetInputs(eventWise, filter_funcs)
    # get the b_decendants
    b_decendants = FormShower.descendant_idxs(eventWise, *eventWise.BQuarkIdx)
    # make the spectral jet and get transformed coordinates
    pseudojet_spectral = spectral_class(eventWise, **spectral_jet_params)
    eigenspace = pseudojet_spectral._eigenspace
    input_idx = list(pseudojet_spectral.InputIdx)
    energies = np.fromiter((row[pseudojet_spectral._Energy_col] for row in pseudojet_spectral._floats),
                           dtype=float)
    # find out which are the b_decendants
    source_idx = eventWise.JetInputs_SourceIdx[input_idx]
    is_decendant = np.fromiter((idx in b_decendants for idx in source_idx), dtype=bool)
    if spiders:
        # cluster the psudojets and get the end loactions
        pseudojet_spectral.assign_parents()
        pjets = pseudojet_spectral.split()
        for pjet in pjets:
            new_input_idx = list(pjet.InputIdx)
            root_new_idx = new_input_idx.index(pjet.root_jetInputIdxs[0])
            leg_idxs = [input_idx.index(idx) for idx, child1 in zip(pjet.InputIdx, pjet.Child1) if child1 == -1]
            leg_xs = eigenspace[leg_idxs, eigendim1]
            leg_ys = eigenspace[leg_idxs, eigendim2]
            legs = np.vstack((leg_xs, leg_ys)).T
            leg_energies = energies[leg_idxs]
            root_pos = np.mean(legs, axis=0)
            plot_spider(ax, SPECTRAL_COLOUR, root_pos, pjet.Energy, legs, leg_energies)
    else:
        input_x = eigenspace[:, eigendim1]
        input_y = eigenspace[:, eigendim2]
        input_energies = np.array(pseudojet_spectral._floats)[:, pseudojet_spectral._Energy_col]
        ax.scatter(input_x, input_y, s=input_energies, c=SPECTRAL_COLOUR)
    # on top of this plot the truth
    ax.scatter(eigenspace[is_decendant, eigendim1], eigenspace[is_decendant, eigendim2],
               c=(0., 0., 0., 0.), edgecolors=TRUTH_COLOUR,
               linewidths=TRUTH_LINEWIDTH, s=TRUTH_SIZE, marker='o')
    ax.set_xlabel(f"Eigenvector {eigendim1}")
    ax.set_ylabel(f"Eigenvector {eigendim2}")


def plot_realspace(eventWise, event_n, fast_jet_params, comparitor_jet_params, ax=None, comparitor_class=SpectralFull):
    """
    Plot the full situation in realspace after clustering

    Parameters
    ----------
    eventWise : EventWise
        data structure wiht input data for clustering
    event_n: int
        index of the event to cluster
    fast_jet_params : dict
        Input parameters for traditional clustering
    comparitor_jet_params: dict
        Input parameters for spectral clustering
    ax : pyplot axis
        axis on which to plot
         (Default value = None)
    comparitor_class : class
        Spectral class used for clustering
        Default value = SpectralFull)

    """
    if ax is None:
        ax = plt.gca()
    ax.set_facecolor('black')
    ax.grid(c='dimgrey', ls='--')
    # create inputs if needed
    if "JetInputs_Energy" not in eventWise.columns:
        filter_funcs = [filter_ends, filter_pt_eta]
        create_jetInputs(eventWise, filter_funcs)
    # add tags if needed
    if "TagIndex" not in eventWise.columns:
        TrueTag.add_tag_particles(eventWise)
    # plot the location of the tag particles
    eventWise.selected_index = event_n
    pseudojet_traditional = run_FastJet(eventWise, **fast_jet_params)
    # plot the pseudojets
    # traditional_colours = [colours(i) for i in np.linspace(0, 0.4, len(pjets_traditional))]
    plot_cluster(pseudojet_traditional, FAST_COLOUR, ax=ax, pt_text=False)
    pseudojet_comparitor = comparitor_class(eventWise, assign=False, **comparitor_jet_params)
    # plot the pseudojets
    #comparitor_colours = [colours(i) for i in np.linspace(0.6, 1.0, len(pjets_comparitor))]
    plot_cluster(pseudojet_comparitor, SPECTRAL_COLOUR, ax=ax, pt_text=False)
    plot_tags(eventWise, ax=ax)
    ax.set_title("Jets")
    ax.set_xlabel("rapidity")
    ax.set_ylim(-np.pi, np.pi)
    ax.set_ylabel("phi")


def eigengrid(eventWise, event_num, fast_jet_params, spectral_jet_params, c_class):
    """
    Grid of plots discribing the state of the eigenspace embedding,
    plus the real space state and the parameters of the jet

    Parameters
    ----------
    eventWise : EventWise
        data structure wiht input data for clustering
    event_num: int
        index of the event to cluster
    fast_jet_params : dict
        Input parameters for traditional clustering
    spectral_jet_params: dict
        Input parameters for spectral clustering
    c_class : class
        Spectral class used for clustering
        Default value = SpectralFull)
    """
    num_eig = spectral_jet_params['NumEigenvectors']
    unit_size=5
    n_rows, n_cols = 2, num_eig-1
    fig, axarry = plt.subplots(n_rows, n_cols, figsize=(n_rows*unit_size, n_cols*unit_size))
    axarry = axarry.reshape((2, -1))
    # top row ~~~
    plot_realspace(eventWise, event_num, fast_jet_params, spectral_jet_params, ax=axarry[0, 0], comparitor_class=c_class)
    # add details in the spare axis
    if n_cols > 1:
        PlottingTools.discribe_jet(properties_dict=fast_jet_params, ax=axarry[0, 1])
        axarry[0, 1].plot([], [], c=SPECTRAL_COLOUR, alpha=JET_ALPHA, label=spectral_jet_params['jet_name'])
        axarry[0, 1].plot([], [], c=FAST_COLOUR, alpha=JET_ALPHA, label=fast_jet_params['jet_name'])
        axarry[0, 1].scatter([], [], marker='d', c=TRUTH_COLOUR, label="Tags")
        axarry[0, 1].scatter([], [], label="b decendant",
                             c=(0., 0., 0., 0.), edgecolors=TRUTH_COLOUR,
                             linewidths=TRUTH_LINEWIDTH, s=TRUTH_SIZE, marker='o')
        #axarry[0, 1].scatter([], [], label="unseen b decendant",
        #                     c=TRUTH_COLOUR, s=TRUTH_LINEWIDTH, marker='x')
        axarry[0, 1].legend()
    if n_cols > 2:
        PlottingTools.discribe_jet(properties_dict=spectral_jet_params, ax=axarry[0, 2])
    if n_cols > 3:
        for i in range(3, n_cols):
            axarry[0, i].axis('off')
    # bottom row
    for i in range(1, num_eig):
        plot_eigenspace(eventWise, event_num, spectral_jet_params, i, 0, ax=axarry[1, i-1], spectral_class=c_class)
        if i > 1:
            axarry[1, i-1].set_ylabel(None)
    plt.show()


if __name__ == '__main__':
    #InputTools.pre_selections = InputTools.PreSelections(
    eventWise_path = InputTools.get_file_name("Where is the eventwise of collection fo eventWise? ", '.awkd')
    eventWise = Components.EventWise.from_file(eventWise_path)
    event_num = InputTools.get_literal("Event number? ", int)
    fast_jet_params = dict(DeltaR=.8, ExpofPTMultiplier=0, jet_name="CambridgeAachenp8")
    #spectral_jet_params = dict(DeltaR=0.4, ExpofPTMultiplier=0.2,
    #                           ExpofPTPosition='eigenspace',
    #                           NumEigenvectors=6,
    #                           Laplacien='symmetric',
    #                           AffinityType='exponent2',
    #                           AffinityCutoff=None,
    #                           #AffinityCutoff=('knn', 4),
    #                           StoppingCondition='standard',
    #                           PhyDistance='Luclus',
    #                           jet_name="SpectralFull")

    #c_class = SpectralFull
    spectral_jet_params = dict(ExpofPTMultiplier=0,
                               ExpofPTPosition='input',
                               NumEigenvectors=3,
                               Laplacien='symmetric',
                               AffinityType='exponent',
                               AffinityCutoff=('distance', 1),
                               PhyDistance='invarient',
                               jet_name="Splitting")

    c_class = Splitting
    check_hyperparameters(c_class, spectral_jet_params)
    eventWise.selected_index = event_num
    jets = c_class(eventWise, assign=False, dict_jet_params=spectral_jet_params)
    jets.plt_assign_parents(save_prefix=None, eigenvector_num=0)
    #for event_num in range(30):
    #    eventWise.selected_index = event_num
    #    jets = c_class(eventWise, assign=False, dict_jet_params=spectral_jet_params)
    #    jets.plt_assign_parents(save_prefix=f"evt{event_num}", eigenvector_num=0)
    #    jets = c_class(eventWise, assign=False, dict_jet_params=spectral_jet_params)
    #    jets.plt_assign_parents(save_prefix=f"evt{event_num}", eigenvector_num=1, steps_required=1)
    #    jets = c_class(eventWise, assign=False, dict_jet_params=spectral_jet_params)
    #    jets.plt_assign_parents(save_prefix=f"evt{event_num}", eigenvector_num=2, steps_required=1)
    plot_realspace(eventWise, event_num, fast_jet_params, spectral_jet_params, comparitor_class=c_class)
    plt.show()
    #eigengrid(eventWise, event_num, fast_jet_params, spectral_jet_params, c_class)

