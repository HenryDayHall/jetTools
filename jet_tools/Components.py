"""Low level components, format apes that of root """
import pickle
import warnings
import os
##from ipdb import set_trace as st
import awkward
import uproot
import numpy as np
#from jet_tools import Constants, InputTools
from jet_tools import Constants, InputTools
import pygit2

def flatten(nested):
    """
    Create a flat iterable of an object with any depth.

    Parameters
    ----------
    nested : iterable
        the object to be flattened

    Returns
    -------
    : iterable
        a flat iterable

    """
    for part in nested:
        if hasattr(part, '__iter__'):
            yield from flatten(part)
        else:
            yield part


def detect_depth(nested):
    """
    Check the maximum depth of an iterable, not counting strings or byte strings.

    Parameters
    ----------
    nested : iterable
        thing to be checked

    Returns
    -------
    found_end : bool
        True if the max depth is determined by something
        that isn't itself iterable, rather than an empty iterable
    max_depth : int
        The greatest number of layers of iterable
    
    """
    # assumed arrays are wider than deep
    max_depth = 0
    for x in nested:
        # some thigs have ittr methods, but are the end goal
        if isinstance(x, (str, bytes)) or not hasattr(x, '__iter__'):
            return True, 0  # absolute end found
        if len(x) > 0:
            abs_end, x_depth = detect_depth(x)
            if abs_end:
                return abs_end, x_depth+1
            max_depth = max(max_depth, x_depth+1)
        else:
            # if it contains an empty list keep looking for somethign not empty
            # but also count it as at least one deep
            max_depth = max(max_depth, 1)
    return False, max_depth


def apply_array_func(func, *nested, depth=None):
    """
    Apply a function to an object with arbitary nested depth,
    such that the function is always applyed at the lowest n layers.

    Parameters
    ----------
    func : callable
        function to be applied, should accept as many
        arguments as there are nested objects provided
    *nested: nested iterables
        objects to apply the function to
    depth : int
        number of layers down to apply the function
        None is equivalent to applying at full depth
        (Default value = None)

    Returns
    -------
    : nested iterables
        objects after function application
    
    """
    if depth is None:
        abs_end, depth = detect_depth(nested[0])
        if not abs_end:  # no objects in array
            print("Warning, no object found in array!")
    return _apply_array_func(func, depth, *nested)


def _apply_array_func(func, depth, *nested):
    """
    Recursive helper function for apply_array_func

    Parameters
    ----------
    func : callable
        function to be applied, should accept as many
        arguments as there are nested objects provided
    depth : int
        number of layers down to apply the function
    *nested: nested iterables
        objects to apply the function to

    Returns
    -------
    : nested iterables
        objects after function application
    
    """
    # all nested must have the same shape
    out = []
    if depth == 0:  # flat lists
        #if len(nested[0]) != 0:
        out = func(*nested)
    else:
        for parts in zip(*nested):
            out.append(_apply_array_func(func, depth-1, *parts))
    try:
        return awkward.fromiter(out)
    except TypeError:
        return out


def confine_angle(angle):
    """
    Confine angle x, s.t. -pi <= x < pi

    Parameters
    ----------
    angle : float or arraylike of floats
        angle to be confined

    Returns
    -------
    angle : float or arraylike of floats
        confined angle
    
    """
    return ((angle + np.pi)%(2*np.pi)) - np.pi


def angular_distance(a, b):
    """
    Get the shortest distance between a and b

    Parameters
    ----------
    a : float or arraylike of floats
        angle
        
    b : float or arraylike of floats
        angle
        

    Returns
    -------
    : float or arraylike of floats
        absolute distance between angles

    """
    raw = a - b
    return np.min((raw%(2*np.pi), np.abs(-raw%(2*np.pi))), axis=0)


def raw_to_angular_distance(raw):
    """
    Convert a angular measurment to the smallest distance in cyclic coordinates.

    Parameters
    ----------
    raw : float or arraylike of floats
        angular measurements

    Returns
    -------
    : float or arraylike of floats
        absolute distance between angles

    """
    return np.min((raw%(2*np.pi), np.abs(-raw%(2*np.pi))), axis=0)


def safe_convert(cls, string):
    """
    Safe conversion out of strings
    designed to deal with None gracefully, and identify sensible bools

    Parameters
    ----------
    cls : class
        the class to convert to
    string : string: str
        the string that needs converting

    Returns
    -------
    : object
        the properly typed object
    
    """
    if string == "None":
        return None
    if cls == bool:
        return Constants.lowerCase_truthies.get(string.lower(), True)
    else: return cls(string)


class git_properties:
    def __init__(self, gitdict=None):
        # start with them being None, try to fix
        self.latest_branch = self.latest_time = self.latest_message = None
        if gitdict is None:  # this is assumed to be brand new
            self.update_latest()
            self.initial_branch = self.latest_branch
            self.initial_time = self.latest_time
            self.outdated = False
        elif 'initial_branch' in gitdict:
            self.initial_branch = gitdict['initial_branch']
            self.initial_time = gitdict.get('initial_time', None)
            self.latest_branch = gitdict.get('latest_branch', None)
            self.latest_time = gitdict.get('latest_time', None)
            self.latest_message = gitdict.get('latest_message', None)
            self.outdated = gitdict.get('outdated', True)
        elif gitdict == 'old':
            # if the initial time is not known, this is a reasonable estimate
            self.initial_branch = "master"
            self.initial_time = 1598960722
            self.outdated = True
        else:  # gitdict has no recognised form
            raise KeyError

    def _get_current(self):
        repo = pygit2.Repository('.')
        branch = repo.head.shorthand
        walker = repo.walk(repo.head.target, pygit2.GIT_SORT_TOPOLOGICAL)
        commit = next(walker)
        time = commit.commit_time
        message = commit.message
        return branch, time, message

    def update_latest(self):
        try:
            self.latest_branch, self.latest_time, self.latest_message = self._get_current()
            self.outdated = False
        except (pygit2.GitError, KeyError):
            self.outdated = True
            #print("Couldn't get git commit, update manually" + 
            #      " with eventWise.write(update_git_properties=True)")

    @property
    def gitdict(self):
        gitdict = dict(initial_branch=self.initial_branch,
                       initial_time=self.initial_time,
                       latest_branch=self.latest_branch,
                       latest_time=self.latest_time,
                       latest_message=self.latest_message,
                       outdated=self.outdated)
        return gitdict

    def __str__(self):
        string = '\n'.join([f"{key}={value}" for key, value in self.gitdict.items()])
        return string


class EventWise:
    """The most basic properties of collections that exist in an eventwise sense"""
    selected_index = None
    EVENT_DEPTH = 1 # events, objects in events
    JET_DEPTH = 2 # events, jets, objects in jets

    def __init__(self, path_name, columns=None, contents=None, hyperparameter_columns=None, gitdict=None):
        """
        Default constructor.
        If loading from file see 'from_file' method as alterative constructor.

        Parameters
        ----------
        path_name : string
            file name for saving with
        columns : list of strings
            names for the contents that should be accessable as attributes
        contents : dict of iterable
            the keys are names (strings), the values are iterables
            the iterabels whoes names are found in the columns list
            should all have the same length int he 0th axis,
            this length beign the number of events
        hyperparameter_columns : list
            members of the contents that should be accessable as attributes
            but may not be iterable and will not be equal to the number of events

        """
        self._loaded_contents = {}  # keep columns that have been accessed in ram
        # the init method must generate some table of items,
        # nomally a jagged array
        self.dir_name, self.save_name = os.path.split(path_name)
        # by using None as the default value it os possible to detect non entry
        if columns is not None:
            self.columns = columns
        else:
            self.columns = []
        if hyperparameter_columns is not None:
            self.hyperparameter_columns = hyperparameter_columns
        else:
            self.hyperparameter_columns = []
        if contents is not None:
            # maintain lazy loading
            self._column_contents = contents
        else:
            self._column_contents = {}
        assert len(set(self.columns)) == len(self.columns), f"Duplicates in columns; {self.columns}"
        self._alias_dict = self._gen_alias()
        self.hyperparameters = {}
        self.git_properties = git_properties(gitdict)

    def _gen_alias(self):
        """
        Some columns may be refered to by more than one name.
        This method checks the _column_contents dict for a list of know alias
        and implements them

        Returns
        -------
        alias_dict : dict
            keys are the alias names values are the name to which the alias refers
        """
        alias_dict = {}
        if 'alias' in self._column_contents:
            for row in self._column_contents['alias']:
                alias_dict[row[0]] = row[1]
                self.columns.append(row[0])
        else:
            try:
                self._column_contents['alias'] = awkward.fromiter([])
            except TypeError:
                self._column_contents = {k:v for k, v in self._column_contents.items()}
                self._column_contents['alias'] = awkward.fromiter([])
        return alias_dict

    def _remove_alias(self, to_remove):
        """
        Remove an existing alias, no data is removed as the data belongs to
        the actual naem not the alias.

        Parameters
        ----------
        to_remove : string
            alisas name to remove

        """
        alias_keys = list(self._column_contents['alias'][:, 0])
        alias_list = list(self._column_contents['alias'])
        alias_idx = alias_keys.index(to_remove)
        # if to_remove is not infact an alias the line above will throw an error
        del self._alias_dict[to_remove]
        del alias_list[alias_idx]
        self._column_contents = {**self._column_contents, 'alias': awkward.fromiter(alias_list)}
        self.columns.remove(to_remove)

    def add_alias(self, name, target):
        """
        Create a new alias.

        Parameters
        ----------
        name : string
            alias name
        target : string
            column the alias refers to
        """
        assert target in self.columns
        assert name not in self.columns
        alias_list = self._column_contents['alias'].tolist()
        alias_list.append([name, target])
        self._column_contents = {**self._column_contents, 'alias': awkward.fromiter(alias_list)}
        self._alias_dict[name] = target
        self.columns.append(name)

    def __getattr__(self, attr_name):
        """
        The columns and hyperparameter_columns are all avalible attrs
        
        Parameters
        ----------
        attr_name : string
            name of the column or hyperparameter_column being access
            case insensative
            can be an alias
        """
        # if _alias_dict gets here then it doent exist yet
        if attr_name == '_alias_dict':
            # we can safely return an empty dict
            return {}
        # capitalise raises the case of the first letter
        attr_name = attr_name[0].upper() + attr_name[1:]
        # if it is an alias get the true name
        attr_name = self._alias_dict.get(attr_name, attr_name)
        if attr_name in self.columns:
            try:  # start by assuming it has been loaded
                if self.selected_index is not None:
                    return self._loaded_contents[attr_name][self.selected_index]
                return self._loaded_contents[attr_name]
            except KeyError:  # it hasn't been loaded
                try:
                    self._loaded_contents[attr_name] = self._column_contents[attr_name][:]
                except TypeError:  # cannot be indexed
                    self._loaded_contents[attr_name] = self._column_contents[attr_name]
                return getattr(self, attr_name)
            except AttributeError: # we dont have a loaded dict yet
                self._loaded_contents = {}
                return getattr(self, attr_name)
        if attr_name in self.hyperparameter_columns:
            return self._column_contents[attr_name]
        raise AttributeError(f"{self.__class__.__name__} does not have {attr_name}")

    def match_indices(self, attr_name, match_from, match_to=None, event_n=None):
        """
        Applies to a single event.
        Given a column and one or two lists of indices,
        select items from those columns.
        If only one list of indices are given, then
        the list of indices selects an item from each row in the specified column.
        If two lists of indices are given then the second
        list is taken to be the ordering of the column,
        it should have the same shape as the column,
        and items from the column are taken from each row
        where the two lists of indices match.

        The second form is useful when a column has id codes for each row
        and you wish to pick out the value corrisponding to a given id from each row.
        Then the first list of indices is the desired id
        and the second list is the assigned ids.

        Parameters
        ----------
        attr_name : string
            column to return values from
        match_from: string or arraylike
            column name or list of desired indices 
        match_to : string or arraylike
            column name or list of indices that indicate
            the order of the attribute
            (Default value = None)
        event_n : int
            Required if this eventWise does not already have a selected_index
            (Default value = None)

        Returns
        -------
        out : awkward array
            the selected objects from this event
        
        """
        if event_n is None:
            assert self.selected_index is not None
        else:
            self.selected_index = event_n
        attr = getattr(self, attr_name)
        if isinstance(match_from, str):
            match_from = getattr(self, match_from)
        if isinstance(match_to, str):
            match_to = getattr(self, match_to)
        if match_to is not None:
            try:
                out = [row[f == t] for f, t, row in zip(match_from, match_to, attr)]
            except TypeError:
                mask = int(match_from) == match_to
                out = [row[m] for m, row in zip(mask, attr)]
        else:
            out = [row[m] for m, row in zip(match_from, attr)]
        return awkward.fromiter(out)

    def __dir__(self):
        """Overiding the __dir__ to add the columns and hyperparameter_columns """
        new_attrs = set(super().__dir__() + self.columns + self.hyperparameter_columns)
        return sorted(new_attrs)

    def __str__(self):
        """ Overrridign the string conversion with a simple description """
        msg = f"<EventWise with {len(self.columns)} columns;" +\
              f" {os.path.join(self.dir_name, self.save_name)}>"
        return msg

    def __eq__(self, other):
        """ Assumeing two eventwise objects saved in the same place are the same object """
        return self.save_name == other.save_name and self.dir_name == other.dir_name

    def write(self, update_git_properties=False):
        """Write to disk"""  # TODO, could append mode speed this up?
        path = os.path.join(self.dir_name, self.save_name)
        assert len(self.columns) == len(set(self.columns)), "Columns contains duplicates"
        non_alias_cols = [c for c in self.columns if c not in self._alias_dict]
        column_order = awkward.fromiter(non_alias_cols)
        #try:
        #    del self._column_contents['column_order']
        #except KeyError:
        #    pass
        non_alias_hcols = [c for c in self.hyperparameter_columns if c not in self._alias_dict]
        hyperparameter_column_order = awkward.fromiter(non_alias_hcols)
        #try:
        #    del self._column_contents['hyperparameter_column_order']
        #except KeyError:
        #    pass
        #all_content = {'column_order': column_order,
        #               'hyperparameter_column_order': hyperparameter_column_order,
        #               **self._column_contents}
        all_content = {}
        # must happen in this order so the new column order overwrites the old
        all_content.update(self._column_contents)
        all_content['column_order'] = column_order
        all_content['hyperparameter_column_order'] = hyperparameter_column_order
        if update_git_properties:
            self.git_properties.update_latest()
        # turn the gitdict into a list of tuples
        # this prevents clashes with other parts of the code that assume everhting is
        # basically an awkward array
        gittuples = awkward.fromiter([(key, value) for key, value
                                      in self.git_properties.gitdict.items()])
        all_content['gitdict'] = gittuples
        awkward.save(path, all_content, mode='w')

    @classmethod
    def from_file(cls, path):
        """
        Alternative constructor.

        Parameters
        ----------
        path : string
            full or relative file path to the saved eventWise

        Returns
        -------
        new_eventWise : EventWise
            loaded eventWise object
        
        """
        contents = awkward.load(path)
        columns = list(contents['column_order'])
        hyperparameter_columns = list(contents['hyperparameter_column_order'])
        if 'gitdict' in contents:  # it will appear as a list of tuples
            gitdict = {key: value for key, value in contents['gitdict']}
        else:  # the file format is outdated
            gitdict = 'old'
        new_eventWise = cls(path, columns=columns,
                            hyperparameter_columns=hyperparameter_columns,
                            contents=contents, gitdict=gitdict)
        return new_eventWise

    def append(self, **new_content):
        """
        Append a new column to the eventwise.
        Will remove any existing column with the same name.
        Will write the results to disk.

        Parameters
        ----------
        **new_content : iterables
            the parameter names are the names for the columns
            the parameter values are the column content
        """
        if new_content:
            new_columns = sorted(new_content.keys())
            # enforce the first letter of each attrbute to be capital
            New_columns = [c[0].upper() + c[1:] for c in new_columns]
            new_content = {C: new_content[c] for C, c in zip(New_columns, new_columns)}
            # check it's not in hyperparameters
            for name in New_columns:
                if name in self.hyperparameter_columns:
                    raise KeyError(f"Already have {name} as a hyperparameter column")
            # delete existing duplicates
            for name in New_columns:
                if name in self.columns:
                    self.remove(name)
            self.columns += New_columns
            self._column_contents = {**self._column_contents, **new_content}
            self.write(update_git_properties=True)

    def append_hyperparameters(self, **new_content):
        """
        Append a new hyperparameter to the eventwise.
        Will remove any existing hyperparameters with the same name.
        Will write the results to disk.

        Parameters
        ----------
        **new_content : objects
            the parameter names are the names for the hyperparameters
            the parameter values are the hyperparameters
        """
        if new_content:
            new_columns = sorted(new_content.keys())
            # enforce the first letter of each attrbute to be capital
            New_columns = [c[0].upper() + c[1:] for c in new_columns]
            new_content = {C: new_content[c] for C, c in zip(New_columns, new_columns)}
            # check it's not in columns
            for name in New_columns:
                if name in self.columns:
                    raise KeyError(f"Already have {name} as a column")
            # delet duplicates
            for name in New_columns:
                if name in self.hyperparameter_columns:
                    self.remove(name)
            self.hyperparameter_columns += New_columns
            self._column_contents = {**self._column_contents, **new_content}
            self.write(update_git_properties=True)

    def remove(self, col_name):
        """
        Delete a column or a hyperparameter_column from the eventwise.
        Does not write.

        Parameters
        ----------
        col_name : string
            name of column or hyperparameter_column to be deleted
        """
        if col_name in self._alias_dict:
            self._remove_alias(col_name)
        else:
            if col_name in self.columns:
                self.columns.remove(col_name)
            elif col_name in self.hyperparameter_columns:
                self.hyperparameter_columns.remove(col_name)
            else:
                raise KeyError(f"Don't have a column called {col_name}")
            if not isinstance(self._column_contents, dict):
                # it's not yet been loaded
                # gotta load it or we cannot delete elements
                self._column_contents = dict(self._column_contents)
            del self._column_contents[col_name]
            if col_name in self._loaded_contents:
                del self._loaded_contents[col_name]

    def remove_prefix(self, col_prefix):
        """
        Delete all columns and hyperparameter_columns that begin with the 
        given prefix. Does not write.

        Parameters
        ----------
        col_prefix : string
            Prefix or everythign to be deleted.
        """
        to_remove = [c for c in self.columns + self.hyperparameter_columns
                     if c.startswith(col_prefix)]
        for col in to_remove:
            self.remove(col)

    def rename(self, old_name, new_name):
        """
        Rename a column or hyperparameter.
        This is not creating an alias, this is changing the actual name.
        Does not write.

        Parameters
        ----------
        old_name : string
            name that should be changed
        new_name : string
            what the name should be changed to
        """
        if old_name in self._alias_dict:
            target = self._alias_dict[old_name]
            self._remove_alias(old_name)
            self.add_alias(new_name, target)
        else:
            if old_name in self.columns:
                self.columns[self.columns.index(old_name)] = new_name
            elif old_name in self.hyperparameter_columns:
                self.hyperparameter_columns[self.hyperparameter_columns.index(old_name)] = new_name
            else:
                raise KeyError(f"Don't have a column called {old_name}")
            if not isinstance(self._column_contents, dict):
                self._column_contents = {k:v for k, v in self._column_contents.items()}
            self._column_contents[new_name] = self._column_contents[old_name]
            del self._column_contents[old_name]

    def rename_prefix(self, old_prefix, new_prefix):
        """
        Change all instances of a prefix in columns and hyperparameter_columns.
        This is not creating an alias, this is changing the actual names.
        Does not write.

        Parameters
        ----------
        old_prefix : string
            prefix that should be changed
        new_prefix : string
            what the prefix should be changed to
        """
        for name in self.columns+self.hyperparameter_columns:
            if name.startswith(old_prefix):
                new_name = name.replace(old_prefix, new_prefix, 1)
                self.rename(name, new_name)

    def fragment(self, per_event_component, **kwargs):
        """
        Split an eventWise into reguar length parts, each one containing
        the same columns but an exclusive subset of the events.
        Writes the new events in a subfolder of the save_dir.

        Parameters
        ----------
        per_event_component : string or list like
            The name of at least one column (or the column vaues itself)
            that has the same axis 0 length as the number of events.
        n_fragments : int
            number of fragments to break into
            (optional, may supply this or fragment_length)
        fragment_length : int
            number of events to put in each fragment
            (optional, may supply this or n_fragments,
             if both supplied, n_fragments is used)
        part_name : string
            component to append to the save name,
            before the filetype extention
            (Default value = "part")
        dir_name : string
            Name of directory to save the new eventWises
            (Optional, if not given a name will be generated from the
            name of the eventWise being split)
        no_dups : bool
            Should content that is not per-event should be stored
            only in the first split created?
            (Default; True)

        Returns
        -------
        all_paths : list of strings
            file paths to the new eventWise objects

        
        """
        if not isinstance(per_event_component, str):
            n_events = len(getattr(self, per_event_component[0]))
        else:
            n_events = len(getattr(self, per_event_component))
        if "n_fragments" in kwargs:
            n_fragments = kwargs["n_fragments"]
            fragment_length = int(n_events/n_fragments)
        elif "fragment_length" in kwargs:
            fragment_length = kwargs['fragment_length']
            # must round up in the case of uneven length fragments
            n_fragments = int(np.ceil(n_events/fragment_length))
        lower_bounds = [i*fragment_length for i in range(n_fragments)]
        upper_bounds = lower_bounds[1:] + [n_events]
        return self.split(lower_bounds, upper_bounds,
                          per_event_component, part_name="fragment",
                          **kwargs)

    def split_unfinished(self, per_event_component, unfinished_component, **kwargs):
        """
        Split the eventWise object into an eventWise where all columns have the
        same length (finished) and an eventWise that lacks content in one or 
        more columns (unfinished)
        Writes the new eventWises in a subfolder of the save_dir.

        Parameters
        ----------
        per_event_component : string or list of strings
            the name of (or names) a component that
            has one row per event
        unfinished_component : string or list of strings
            the name of (or names) a component that is shorter than
            one row per event
        part_name : string
            component to append to the save name,
            before the filetype extention
            (Default value = "part")
        dir_name : string
            Name of directory to save the new eventWises
            (Optional, if not given a name will be generated from the
            name of the eventWise being split)
        no_dups : bool
            Should content that is not per-event should be stored
            only in the first split created?
            (Default; True)

        Returns
        -------
        all_paths : list of strings
            file paths to the new eventWise objects

        
        """
        self.selected_index = None
        if not isinstance(per_event_component, str):
            n_events = len(getattr(self, per_event_component[0]))
        else:
            n_events = len(getattr(self, per_event_component))
        if not isinstance(unfinished_component, str):
            if not unfinished_component:
                # there are no unfinished components
                # the whole thing should be considered infinnished
                point_reached = 0
            else:
                # check they all have the same length
                lengths = {len(getattr(self, name)) for name in unfinished_component}
                assert len(lengths) == 1, "Error not all unfinished components have the same length"
                point_reached = list(lengths)[0]
        else:
            point_reached = len(getattr(self, unfinished_component))
        assert point_reached <= n_events
        if point_reached == 0:
            # the whole thing is unfinished
            return None, os.path.join(self.dir_name, self.save_name)
        if point_reached == n_events:
            # the event is complete
            return os.path.join(self.dir_name, self.save_name), None
        lower_bounds = [0, point_reached]
        upper_bounds = [point_reached, n_events]
        return self.split(lower_bounds, upper_bounds,
                          per_event_component, part_name="progress",
                          **kwargs)

    def split(self, lower_bounds, upper_bounds=None, per_event_component="Energy", part_name="part", **kwargs):
        """
        Split an eventWise into specified parts, each one containing
        the same columns but a subset of the events.
        Writes the new eventWises in a subfolder of the save_dir.

        Parameters
        ----------
        lower_bounds : array like of int
            first event in each section, inclusive
        upper_bounds : array like of int
            last event in each section, exclusive
        per_event_component : string or list like
            the name of (or content of) a component that
            has one row per event
            (Default value = "Energy")
        part_name : string
            component to append to the save name,
            before the filetype extention
            (Default value = "part")
        dir_name : string
            Name of directory to save the new eventWises
            (Optional, if not given a name will be generated from the
            name of the eventWise being split)
        no_dups : bool
            Should content that is not per-event should be stored
            only in the first split created?
            (Default; True)

        Returns
        -------
        all_paths : list of strings
            file paths to the new eventWise objects
        
        """
        # not thread safe....
        self.selected_index = None
        # decide where the fragments will go
        name_format = self.save_name.split('.', 1)[0] + "_" + part_name + "{}.awkd"
        if 'dir_name' in kwargs:
            save_dir = kwargs['dir_name']
        else:
            save_dir = os.path.join(self.dir_name, name_format.replace('{}.awkd', ''))
        try:
            os.mkdir(save_dir)
        except FileExistsError:
            pass
        # not all the components are obliged to have the same number of items,
        # a component is identified as being the correct length, and all components
        # of that length are split between the fragments
        if not isinstance(per_event_component, str):
            # if a list of per event compoenents are given
            # it should eb verified that they are all the same length
            to_check = set([c.capitalize() for c in per_event_component[1:]])
            per_event_component = per_event_component[0]
        else:
            to_check = set()
        n_events = len(getattr(self, per_event_component))
        # work out which lists have this property
        per_event_cols = [c for c in self.columns
                          if len(self._column_contents[c]) == n_events]
        assert to_check.issubset(per_event_cols)
        new_contents = []
        if upper_bounds is None:  # treat lower bounds as a list of indices
            for index_list in lower_bounds:
                if len(index_list) == 0:
                    # append none as a placeholder
                    new_contents.append(None)
                else:
                    new_content = {k: self._column_contents[k][index_list] for k in per_event_cols}
                    new_contents.append(new_content)
        else:
            for lower, upper in zip(lower_bounds, upper_bounds):
                if lower > upper:
                    raise ValueError(f"lower bound {lower} greater than upper bound {upper}")
                if lower == upper:
                    # append none as a placeholder
                    new_contents.append(None)
                else:
                    new_content = {k: self._column_contents[k][lower:upper] for k in per_event_cols}
                    new_contents.append(new_content)
        # if no dupes only put the unchanged content in the first event
        no_dups = kwargs.get('no_dups', True)
        all_paths = []
        i = 0
        add_unsplit = True
        # add the hyperparameters to all things...
        hyper_param_dict = {name: self._column_contents[name] for
                            name in self.hyperparameter_columns}
        unchanged_parts = {k: self._column_contents[k][:] for k in self._column_contents.keys()
                           if k not in per_event_cols and k not in hyper_param_dict}
        for new_content in new_contents:
            if new_content is None:
                all_paths.append(None)
                continue
            new_content = {**new_content, **hyper_param_dict}
            # this bit isn't thread safe and could create a race condition
            while name_format.format(i) in os.listdir(save_dir):
                i += 1
            name = name_format.format(i)
            path = os.path.join(save_dir, name)
            if add_unsplit:
                new_content = {**new_content, **unchanged_parts}
                eventWise = type(self)(path,
                                columns=self.columns,
                                hyperparameter_columns=self.hyperparameter_columns,
                                contents=new_content)
                if no_dups:
                    # don't do this again
                    add_unsplit = False
            else:
                eventWise = type(self)(path,
                                columns=per_event_cols,
                                hyperparameter_columns=self.hyperparameter_columns,
                                contents=new_content)
            all_paths.append(path)
            eventWise.write()
        return all_paths

    @classmethod
    def recursive_combine(cls, dir_name, check_for_dups=False, del_fragments=True):
        """
        Combine every eventWise found in or under the given directory.

        Parameters
        ----------
        dir_name : string
            directory naem to find the eventWise files in
        check_for_dups : bool
            If True prevents adding columns or hyperparameter_columns with the same name and 
            content multiple times.
            Shouldn't be needed if the split was done with no_dups.
            (Default value = False)
        del_fragments : bool
            Should the fragments that have been combined be deleted
            once the combined file is written.
            (Default value = True)

        Returns
        -------
        joined_name : string
            name of disk of the joined eventWise
        
        """
        if dir_name.endswith('/'):
            dir_name = dir_name[:-1]
        root_dir = '/'.join(dir_name.split('/')[:-1])
        save_base = dir_name.split('/')[-1].split('_', 1)[0]
        for name in os.listdir(dir_name):
            if not name.endswith('.awkd'):
                subdir_name = os.path.join(dir_name, name)
                merged_name = cls.recursive_combine(subdir_name, check_for_dups, del_fragments)
                os.rename(merged_name, subdir_name + ".awkd")
        fragments = [n for n in os.listdir(dir_name) if n.endswith('.awkd')]
        combined_eventWise = cls.combine(dir_name, save_base, fragments, check_for_dups, del_fragments=True)
        joined_name = os.path.join(root_dir, save_base+".awkd")
        os.rename(os.path.join(combined_eventWise.dir_name, combined_eventWise.save_name),
                  joined_name)
        if del_fragments:
            os.rmdir(dir_name)
        return joined_name

    @classmethod
    def combine(cls, dir_name, save_base, fragments=None, check_for_dups=False,
                del_fragments=False, weighted_average=None):
        """
        Join multiple eventWise objects so that all events are contaiend in a single eventWise.
        Inverts the split funciton.
        Writes to disk.
        If coulmns don't have event length requires Event_n to sort them

        Parameters
        ----------
        dir_name : string
            directory naem to find the eventWise files in
        save_base: string
            the prefix of the file names to be combined
        fragments : list of strings
            names of the files to be combined
            if None then all files that start with the save_base
            are combined
            (Default value = None)
        check_for_dups : bool
            If True prevents adding columns or hyperparameter_columns with the same name and 
            content multiple times.
            Shouldn't be needed if the split was done with no_dups.
            (Default value = False)
        del_fragments : bool
            Should the fragments that have been combined be deleted
            once the combined file is written.
            (Default value = False)
        weighted_average : list of str
            Names of hyperparameter columns for which a weighted average should be taken
            instead of checking for a match

        Returns
        -------
        new_eventWise : EventWise
            the combined eventWise object
        
        """
        if weighted_average is None:
            weighted_average = []
        in_dir = os.listdir(dir_name)
        if fragments is None:
            fragments = [name for name in in_dir
                         if name.startswith(save_base)
                         and name.endswith(".awkd")]
        columns = []
        hyperparameter_columns = set()
        contents = {}
        pickle_strs = {}  # when checking for dups
        # to create the averaged hyperparameters
        hyperparameter_to_average = {}
        for fragment_n, fragment in enumerate(fragments):
            path = os.path.join(dir_name, fragment)
            try:
                content_here = awkward.load(path)
            except Exception:
                print(f"Problem in {path}, skipping")
                continue
            # check hyperparameters match and add as needed
            found_hcols = set(content_here.get("hyperparameter_column_order", []))
            # if we cannot find a column order try Event_n
            column_name = getattr(content_here, "column_order", ["Event_n"])[0]
            # if we cannot fnd the column return 1
            segment_length = len(content_here.get(column_name, [None]))
            if found_hcols:  # check they are the same here
                for name in found_hcols:
                    if name not in hyperparameter_columns:
                        # it is a new hyperparameter
                        hyperparameter_columns.add(name)
                        if name in weighted_average:
                            hyperparameter_to_average[name] = [content_here[name]], [segment_length]
                        else:
                            contents[name] = content_here[name]
                    elif name in weighted_average:
                        # it is not a new hyper parameter and take an average
                        hyperparameter_to_average[name][0].append(content_here[name])
                        hyperparameter_to_average[name][1].append(segment_length)
                    else:  # we have seen it before, check for match
                        error_msg = f"Missmatch in hyperparameter {name}"
                        try:
                            np.testing.assert_allclose(content_here[name], contents[name],
                                                       err_msg=error_msg)
                        except TypeError:
                            assert content_here[name] == contents[name], error_msg
            # update columns
            for name in content_here['column_order']:
                if name not in columns:
                    columns.append(name)
                    # add padding
                    contents[name] = [[] for _ in range(fragment_n)]
            # then go through the contents adding what is foud, but do not flatten
            for key in columns:
                if key not in content_here:
                    contents[key].append([]) # placeholder
                elif check_for_dups:
                    if key not in pickle_strs:
                        pickle_strs[key] = pickle.dumps(content_here[key])
                        contents[key].append(content_here[key])
                    elif pickle_strs[key] != pickle.dumps(content_here[key]):
                        contents[key].append(content_here[key])
                    else:  # need to add a placeholder
                        contents[key].append([])
                else:
                    contents[key].append(content_here[key])
            # anythign else should be the sma ein all instances,
            # or only be found once
            for key in content_here:
                if key in hyperparameter_columns or key in columns:
                    continue  # already been delt with
                if key not in contents:
                    contents[key] = content_here[key]
                    pickle_strs[key] = pickle.dumps(content_here[key])
                else:
                    pickle_strs[key] == pickle.dumps(content_here[key]), f"What is {key}?"
        # sort out the columns that should be averaged
        for name, (values, lengths) in hyperparameter_to_average.items():
            contents[name] = np.average(values, weights=lengths)
        # make all non hyperparameters into awkward arrays
        for key in contents:
            if key not in hyperparameter_columns:
                contents[key] = awkward.fromiter(contents[key])
        # if it's possible to order the contents correctly, do so
        if "Event_n" in contents:
            # assume that each fragment contains a continuous set of events
            start_point = [numbering[0] for numbering in contents["Event_n"]]
            order = np.argsort(start_point)
            for key in columns:
                contents[key] = contents[key][order].flatten()
        else:
            for key in columns:
                contents[key] = contents[key].flatten()
            lengths = {len(contents[key]) for key in columns}
            assert len(lengths) == 1, "Columns with differing length, but no Event_n"
        new_eventWise = cls(os.path.join(dir_name, save_base+"_joined.awkd"),
                            columns=columns, contents=contents,
                            hyperparameter_columns=hyperparameter_columns)
        new_eventWise.write()
        if del_fragments:
            for fragment in fragments:
                os.remove(os.path.join(dir_name, fragment))
        return new_eventWise


def event_matcher(eventWise1, eventWise2):
    """
    Find the indices required to match two eventWise objects.
    Currently untested... used with caution.
    

    Parameters
    ----------
    eventWise1 : EventWise
        eventWise object to match from
    eventWise2 : EventWise
        eventWise object to match from
        

    Returns
    -------
    order : list of int
        order required to match eventWise2 to eventWise1
    
    """
    eventWise1.selected_index = None
    eventWise2.selected_index = None
    common_columns = set(eventWise1.columns).intersection(set(eventWise2.columns))
    common_columns = np.array(list(common_columns))
    column_type = []
    for name in common_columns:
        part = getattr(eventWise1, name)
        while hasattr(part[0], '__iter__'):
            part = part.flatten()
            if len(part) == 0 or isinstance(part[0], str):
                break
        if len(part) > 0:
            column_type.append(type(part[0]))
        else:
            column_type.append(None)
    column_type = np.array(column_type)
    isint = np.array([col == np.int64 for col in column_type])
    int_cols = common_columns[isint]
    isfloat = np.array([col == np.float64 for col in column_type])
    float_cols = common_columns[isfloat]
    assert "Event_n" in common_columns
    length1 = len(eventWise1.Event_n)
    length2 = len(eventWise2.Event_n)
    order = np.full(length1, -1, dtype=int)
    best_float_matches = np.zeros(length2)
    for i in range(length1):
        possibles = np.where(eventWise2.Event_n == eventWise1.Event_n[i])[0]
        if len(possibles) == 0:
            continue
        int_gap = np.zeros_like(possibles, dtype=int)
        for col in int_cols:
            f1 = getattr(eventWise1, col)[i]
            f2s = getattr(eventWise2, col)[possibles]
            try:
                int_gap += [np.int64(np.nan_to_num(recursive_distance(f1, f2))) for f2 in f2s]
            except Exception:
                int_gap = [np.int64(np.nan_to_num(recursive_distance(f1, f2))) for f2 in f2s]
        possibles = possibles[int_gap == 0]
        if len(possibles) == 1 and possibles[0] not in order:
            order[i] = possibles[0]
            continue
        if len(possibles) == 0:
            continue
        float_gap = np.zeros_like(possibles, dtype=float)
        for col in float_cols:
            f1 = getattr(eventWise1, col)[i]
            f2s = getattr(eventWise2, col)[possibles]
            float_gap += [recursive_distance(f1, f2) for f2 in f2s]
        best = np.argmin(float_gap)
        best_idx = possibles[best]
        if best_idx not in order:  # hasen't been used yet
            order[i] = best_idx
            best_float_matches[best_idx] = float_gap[best]
        elif float_gap[best] < best_float_matches[best_idx]:
            # then steal it
            order[order == best_idx] = -1
            order[i] = best_idx
            best_float_matches[best_idx] = float_gap[best]
    return order


def recursive_distance(awkward1, awkward2):
    """
    Get the sum of the absolute diference between two arrays
    with the same shape. Shape can be arbitary.

    Parameters
    ----------
    awkward1 : array like
        first array of values
    awkward2 : array like
        second array of values

    Returns
    -------
    distance : float
        absolute cumulative diference
    
    """
    distance = 0.
    iter1 = hasattr(awkward1, '__iter__')
    iter2 = hasattr(awkward2, '__iter__')
    if iter1 and iter2:
        if len(awkward1) != len(awkward2):
            return np.inf
        return np.sum([recursive_distance(a1, a2) for a1, a2 in zip(awkward1, awkward2)])
    if iter1 or iter2:
        return np.inf
    distance += np.abs(awkward1 - awkward2)
    return distance


def add_rapidity(eventWise, base_name=None):
    """
    Append a calculated rapidity to the eventWise.

    Parameters
    ----------
    eventWise : EventWise
        contains data and will store result
    base_name : string
        prefix for inputs to calculation
        (Default value = '')
    """
    eventWise.selected_index = None
    if base_name is None:
        # find all the things with an angular property
        pt_cols = [c[:-3] for c in eventWise.columns if c.endswith("PT")]
        pz_cols = [c[:-3] for c in pt_cols if c+"Pz" in eventWise.columns]
        all_cols = [c[:-3] for c in pz_cols if c+"Energy" in eventWise.columns]
        base_names = [c for c in all_cols if (c+"Rapidity") not in eventWise.columns]
    else:
        if not base_name.endswith('_'):
            base_name += '_'
        base_names = [base_name]
    new_content = {}
    for base_name in base_names:
        pts = getattr(eventWise, base_name+"PT")
        pzs = getattr(eventWise, base_name+"Pz")
        es = getattr(eventWise, base_name+"Energy")
        n_events = len(getattr(eventWise, base_name+"PT"))
        rapidities = []
        for event_n in range(n_events):
            if event_n % 10 == 0:
                print(f"{event_n/n_events:.1%}", end='\r')
            rap_here = []
            eventWise.selected_index = event_n
            pts = getattr(eventWise, base_name+"PT")
            pzs = getattr(eventWise, base_name+"Pz")
            es = getattr(eventWise, base_name+"Energy")
            rap_here = ptpze_to_rapidity(pts, pzs, es)
            rapidities.append(awkward.fromiter(rap_here))
        rapidities = awkward.fromiter(rapidities)
        new_content[base_name+"Rapidity"] = rapidities
    eventWise.selected_index = None
    eventWise.append(**new_content)


def add_thetas(eventWise, basename=None):
    """
    Append a calculated theta to the eventWise.

    Parameters
    ----------
    eventWise : EventWise
        contains data and will store result
    base_name : string
        prefix for inputs to calculation
        if None will calculate for all prefixes assocated with sutable inputs
        (Default value = None)
    """
    eventWise.selected_index = None
    contents = {}
    if basename is None:
        # find all the things with an angular property
        phi_cols = [c[:-3] for c in eventWise.columns if c.endswith("Phi")]
        missing_theta = [c for c in phi_cols if (c+"Theta") not in eventWise.columns]
    else:
        if len(basename) > 0:
            if not basename.endswith('_'):
                basename = basename + '_'
        missing_theta = [basename]
    for name in missing_theta:
        avalible_components = [c[len(name):] for c in eventWise.columns
                               if c.startswith(name)]
        if (set(("PT", "Pz")) <= set(avalible_components) or
            #set(("Birr", "PT")) <= set(avalible_components) or  # problem, we don't have a direction
            set(("Birr", "Pz")) <= set(avalible_components) or
            set(("Px", "Py", "Pz")) <= set(avalible_components)):
            # then we will work with momentum
            pz = getattr(eventWise, name+"Pz")
            if "Birr" in avalible_components:
                birr = getattr(eventWise, name+"Birr")
                theta = np.arccos(pz/birr)
            else:
                if "PT" in avalible_components:
                    pt = getattr(eventWise, name+"PT")
                else:
                    pt = np.sqrt(getattr(eventWise, name+"Px")**2 +
                                 getattr(eventWise, name+"Py")**2)
                # tan(theta) = oposite/adjacent = pt/pz
                theta = ptpz_to_theta(pt, pz)
            #else:
            #    birr = getattr(eventWise, name+"Birr")
            #    pt = getattr(eventWise, name+"PT")
            #    theta = np.arcsin(pt/birr)
        #elif "ET" in avalible_components:  # TODO, need to add a workaround for towers
        #    # we work with energy          # problem is knowing direction
        #    et = getattr(eventWise, name+"ET")
        #    e = getattr(eventWise, name+"Energy")
        #    theta = np.arcsin(et/e)
        else:
            print(f"Couldn't calculate Theta for {name}")
            continue
        contents[name+"Theta"] = theta
    eventWise.append(**contents)


def add_pseudorapidity(eventWise, basename=None):
    """
    Append a calculated pseudorapidity to the eventWise.

    Parameters
    ----------
    eventWise : EventWise
        contains data and will store result
    base_name : string
        prefix for inputs to calculation
        if None will calculate for all prefixes assocated with sutable inputs
        (Default value = None)
    """
    eventWise.selected_index = None
    contents = {}
    if basename is None:
        # find all the things with theta
        theta_cols = [c[:-5] for c in eventWise.columns if c.endswith("_Theta")]
        missing_ps = [c for c in theta_cols if (c+"PseudoRapidity") not in eventWise.columns]
        if "Theta" in eventWise.columns and "PseudoRapidity" not in eventWise.columns:
            missing_ps.append('')
    else:
        if len(basename) > 0:
            if not basename.endswith('_'):
                basename = basename + '_'
        missing_ps = [basename]
    for name in missing_ps:
        theta = getattr(eventWise, name+"Theta")
        pseudorapidity = apply_array_func(theta_to_pseudorapidity, theta)
        contents[name+"PseudoRapidity"] = pseudorapidity
    eventWise.append(**contents)


def ptpz_to_theta(pt_list, pz_list):
    """
    Given pt (transverse momentum) and pz (momentum in the z direction)
    calculate theta.

    Parameters
    ----------
    pt_list : float or array like
        pt inputs
    pz_list : float or array like
        pz inputs

    Returns
    -------
    theta : float or array like
        theta as a float or a numpy array, depending on the input
    
    """
    return np.arctan2(pt_list, pz_list)


def pxpy_to_phipt(px_list, py_list):
    """
    Given px and py (momentum in the x and y directions) calculate
    phi angle and pt (transverse momentum)

    Parameters
    ----------
    px_list : float or array like
        px inputs
    py_list : float or array like
        py inputs

    Returns
    -------
    phi : float or array like
        phi as a float or a numpy array, depending on the input
    pt : float or array like
        pt as a float or a numpy array, depending on the input
    """
    return np.arctan2(py_list, px_list), np.sqrt(px_list**2 + py_list**2)


def theta_to_pseudorapidity(theta_list):
    """
    Given the angle theta calculate pseudorapidity.

    Parameters
    ----------
    theta_list : float or array like
        theta inputs

    Returns
    -------
    pseudorapidity : float or array like
        pseudorapidity as a float or a numpy array, depending on the input
    
    """
    return_float = isinstance(theta_list, float)
    # awkward arrays also return true for isinstance np.ndarray
    if not isinstance(theta_list, np.ndarray):
        theta_list = np.array(theta_list, dtype=float)
    else:
        theta_list = theta_list.astype(float)
    with np.errstate(invalid='ignore'):
        infinite = np.logical_or(theta_list == 0, theta_list == np.pi)
        restricted_theta = confine_angle(theta_list)
        restricted_theta = np.minimum(restricted_theta, np.pi - restricted_theta)
    tan_restricted = np.tan(np.abs(restricted_theta)/2)
    pseudorapidity = np.full_like(theta_list, np.inf)
    pseudorapidity[~infinite] = -np.log(tan_restricted[~infinite])
    pseudorapidity[theta_list > np.pi/2] *= -1.
    if return_float:
        pseudorapidity = float(pseudorapidity)
    return pseudorapidity


def ptpze_to_rapidity(pt_list, pz_list, e_list):
    """
    Given pt and pz (momentum in the transverse and z directions) and energy
    calculate the rapidity.

    Parameters
    ----------
    pt_list : float or array like
        pt inputs
    pz_list : float or array like
        pz inputs
    e_list : float or array like
        energy inputs

    Returns
    -------
    rapidity : float or array like
        rapidity as a float or a numpy array, depending on the input
    
    """
    # can apply it to arrays of floats or floats, not ints
    rapidity = np.zeros_like(e_list)
    # deal with the infinite rapidities
    large_num = np.inf  # change to large number???
    inf_mask = np.logical_and(pt_list == 0, e_list == np.abs(pz_list))
    rapidity[inf_mask] = large_num
    # don't caculate rapidity that should be exactly 0 either
    to_calculate = np.logical_and(~inf_mask, pz_list != 0)
    e_use = np.array(e_list)[to_calculate]
    pz_use = np.array(pz_list)[to_calculate]
    pt2_use = np.array(pt_list)[to_calculate]**2
    m2 = np.clip(e_use**2 - pz_use**2 - pt2_use, 0, None)
    mag_rapidity = 0.5*np.log((pt2_use + m2)/((e_use - np.abs(pz_use))**2))
    rapidity[to_calculate] = mag_rapidity
    with np.errstate(invalid='ignore'):
        rapidity *= np.sign(pz_list)
    if not hasattr(pt_list, '__iter__'):
        rapidity = float(rapidity)
    return rapidity


def add_PT(eventWise, basename=None):
    """
    Append a calculated pt (transverse momentum) to the eventWise.

    Parameters
    ----------
    eventWise : EventWise
        contains data and will store result
    base_name : string
        prefix for inputs to calculation
        if None will calculate for all prefixes assocated with sutable inputs
        (Default value = None)
    """
    eventWise.selected_index = None
    contents = {}
    if basename is None:
        # find all the things with px, py
        px_cols = [c[:-2] for c in eventWise.columns if c.endswith("Px")]
        pxpy_cols = [c[:-2] for c in eventWise.columns if c.endswith("Py") and c[:-2] in px_cols]
        missing_pt = [c for c in pxpy_cols if (c+"PT") not in eventWise.columns]
    else:
        if len(basename) > 0:
            if not basename.endswith('_'):
                basename = basename + '_'
        missing_pt = [basename]
    for name in missing_pt:
        px = getattr(eventWise, name+"Px")
        py = getattr(eventWise, name+"Py")
        pt = apply_array_func(lambda x, y: np.sqrt(x**2 + y**2), px, py)
        contents[name+"PT"] = pt
    eventWise.append(**contents)


def add_phi(eventWise, basename=None):
    """
    Append a calculated phi (barrel angle) to the eventWise.

    Parameters
    ----------
    eventWise : EventWise
        contains data and will store result
    base_name : string
        prefix for inputs to calculation
        if None will calculate for all prefixes assocated with sutable inputs
        (Default value = None)
    """
    eventWise.selected_index = None
    contents = {}
    if basename is None:
        # find all the things with px, py
        px_cols = [c[:-2] for c in eventWise.columns if c.endswith("Px")]
        pxpy_cols = [c[:-2] for c in eventWise.columns if c.endswith("Py") and c[:-2] in px_cols]
        missing_phi = [c for c in pxpy_cols if (c+"Phi") not in eventWise.columns]
    else:
        if len(basename) > 0:
            if not basename.endswith('_'):
                basename = basename + '_'
        missing_phi = [basename]
    for name in missing_phi:
        px = getattr(eventWise, name+"Px")
        py = getattr(eventWise, name+"Py")
        phi = apply_array_func(lambda x, y: np.arctan2(y, x), px, py)
        contents[name+"Phi"] = phi
    eventWise.append(**contents)


def add_mass(eventWise, basename=None):
    """
    Append a calculated invarient mass to the eventWise.

    Parameters
    ----------
    eventWise : EventWise
        contains data and will store result
    base_name : string
        prefix for inputs to calculation
        if None will calculate for all prefixes assocated with sutable inputs
        (Default value = None)
    """
    eventWise.selected_index = None
    contents = {}
    if basename is None:
        # find all the things with e, px, py, pz
        px_cols = [c[:-2] for c in eventWise.columns if c.endswith("Px")]
        pxpypze_cols = [c[:-2] for c in eventWise.columns if c.endswith("Py") and c[:-2] in px_cols
                and c[:-2]+"Pz" in eventWise.columns and c[:-2]+"Energy" in eventWise.columns]
        missing_mass = [c for c in pxpypze_cols if (c+"Mass") not in eventWise.columns]
    else:
        if len(basename) > 0:
            if not basename.endswith('_'):
                basename = basename + '_'
        missing_mass = [basename]
    for name in missing_mass:
        px = getattr(eventWise, name+"Px")
        py = getattr(eventWise, name+"Py")
        pz = getattr(eventWise, name+"Pz")
        e = getattr(eventWise, name+"Energy")
        contents[name+"Mass"] = np.sqrt(e**2 - px**2 - py**2 - pz**2)
    eventWise.append(**contents)


def add_all(eventWise, basename=None, inc_mass=False):
    add_thetas(eventWise, basename)
    add_PT(eventWise, basename)
    add_phi(eventWise, basename)
    add_rapidity(eventWise, basename)
    add_pseudorapidity(eventWise, basename)
    if inc_mass:
        add_mass(eventWise, basename)


def last_instance(eventWise, particle_idx):
    """
    Find the particle_idx at which the specified particle decays.

    Parameters
    ----------
    eventWise : EventWise
        EventWise object containign particles,
        should have an event specifed with selected_index
        
    particle_idx : int
        the idx of the particle within the event

    Returns
    -------
    particle_idx : int
        the idx of the particle in the instance it decays

    """
    assert eventWise.selected_index is not None
    mcpid = eventWise.MCPID[particle_idx]
    has_next = True
    while has_next:
        children = eventWise.Children[particle_idx]
        has_next = False
        for child in children:
            if eventWise.MCPID[child] == mcpid:
                has_next = True
                particle_idx = child
                break
    # if we get here there was no child with the mcipd
    return particle_idx


class RootReadout(EventWise):
    """Reads arbitary components from a root file created by Delphes"""
    def __init__(self, file_name, component_names,
                 component_of_root_file="Delphes",
                 key_selection_function=None, all_prefixed=False):
        """
        Default constructor.

        Parameters
        ----------
        file_name : string
            file name of the root file to read
        component_names : list of strings
            names of the subcomponents
            inside the specified component to be read
        component_of_root_file : string
            start point for finding components
            (Default; "Delphes")
        key_selection_function : callable
            a function to filter what is read
            should reduce the junk entries
        all_prefixed : bool
            True if the first component has a prefix too
            otherwise replace the first component with an empty list
            (Default; False)

        """
        # read the root file
        self._root_file = uproot.open(file_name)[component_of_root_file]
        if key_selection_function is not None:
            self._key_selection_function = key_selection_function
        else:
            # remove keys starting with lower case letters (they seem to be junk)
            def func(key):
                """
                Decide if a key should be filtered
                

                Parameters
                ----------
                key : string
                    name of key

                Returns
                -------
                : bool
                    should the key be kept?
                """
                return (key.decode().split('.', 1)[1][0]).isupper()
            self._key_selection_function = func
        if all_prefixed:
            prefixes = component_names
        else:
            # the first component has no prefix
            prefixes = [''] + component_names[1:]
        self.columns = []
        self.hyperparameter_columns = []  # need to make this to prevent attr errors
        self._column_contents = {}
        for prefix, component in zip(prefixes, component_names):
            self.add_component(component, prefix)
        if "Delphes" in component_of_root_file:
            # some tweeks for files produced by Delphes
            self._fix_Birr()
            self._unpack_TRefs()
            self._insert_inf_rapidities()
            self._fix_Tower_NTimeHits()
            self._remove_Track_Birr()
        super().__init__(file_name, columns=self.columns, contents=self._column_contents)

    def add_component(self, component_name, key_prefix=''):
        """
        Reads a given component from the root file and adds it to this object.

        Parameters
        ----------
        component_name : string
            the subcomponent of the root file to add
        key_prefix : string
            prefix to used when saving in this object
        """
        if key_prefix != '':
            key_prefix = key_prefix[0].upper() + key_prefix[1:]
            # check it ends in an underscore
            if not key_prefix.endswith('_'):
                key_prefix += '_'
            # check that this prefix is not seen anywhere else
            for key in self._column_contents.keys():
                assert not key.startswith(key_prefix)
        full_keys = [k for k in self._root_file[component_name].keys()
                     if self._key_selection_function(k)]
        all_arrays = self._root_file.arrays(full_keys)
        # process the keys to ensure they are usable a attribute names and unique
        attr_names = []
        # some attribute names get altered
        # this is becuase it it not possible to save when one name is a prefix of another
        rename = {'E': 'Energy', 'P':'Birr', 'ErrorP': 'ErrorBirr',
                  'T': 'Time',
                  'EtaOuter': 'OuterEta', 'PhiOuter': 'OuterPhi',
                  'TOuter': 'OuterTime', 'XOuter': 'OuterX',
                  'YOuter': 'OuterY', 'ZOuter': 'OuterZ',
                  'Xd': 'dX', 'Yd': 'dY', 'Zd': 'dZ'}
        for key in full_keys:
            new_attr = key.decode().split('.', 1)[1]
            new_attr = ''.join(filter(str.isalnum, new_attr))
            new_attr = rename.get(new_attr, new_attr)
            new_attr = key_prefix + new_attr[0].upper() + new_attr[1:]
            if new_attr[0].isdigit():
                # atribute naems can contain numbers
                # but they must not start with one
                new_attr = 'x' + new_attr
            attr_names.append(new_attr)
        # now process to prevent duplicates
        attr_names = np.array(attr_names)
        for i, name in enumerate(attr_names):
            if name in attr_names[i+1:]:
                locations = np.where(attr_names == name)[0]
                n = 1
                for index in locations:
                    # check it's not already there (for some reason)
                    while attr_names[index]+str(n) in attr_names:
                        n += 1
                    attr_names[index] += str(n)
        assert len(set(attr_names)) == len(attr_names), "Duplicates in columns; "+\
                f"{[n for i, n in enumerate(attr_names) if n in attr_names[i+1:]]}"
        new_column_contents = {name: all_arrays[key]
                               for key, name in zip(full_keys, attr_names)}
        self._column_contents = {**new_column_contents, **self._column_contents}
        # make this a list for fixed order
        self.columns += sorted(new_column_contents.keys())

    def _unpack_TRefs(self):
        """ Lists of TRefs must be converted from uproot.rootio.trefs into integer indices. """
        for name in self.columns:
            is_tRef, converted = self._recursive_to_id(self._column_contents[name])
            if is_tRef:
                converted = awkward.fromiter(converted)
                self._column_contents[name] = converted
            #  the Tower_Particles column is infact a tref array,
            # but due to indiosyncracys will be converted to an object array by uproot
            if isinstance(self._column_contents[name],
                          awkward.array.objects.ObjectArray):
                if name == "Tower_Particles":
                    new_tower_refs = []
                    for event in self._column_contents[name]:
                        event = [[idx-1 for idx in tower] for tower in event]
                        new_tower_refs.append(awkward.fromiter(event))
                    self._column_contents[name] = awkward.fromiter(new_tower_refs)
                else:
                    msg = f"{name} is an Object array " +\
                          "Only expected Tower_Particles to be an object array"
                    warnings.warn(msg, RuntimeWarning)  # don't raise an error
                    # it may be possible to fix this in the save file
        # reflect the tracks and towers
        shape_ref = "Energy"
        name = "Particle_Track"
        self.columns.append(name)
        self._column_contents[name] = self._reflect_references("Track_Particle",
                                                               shape_ref, depth=1)
        name = "Particle_Tower"
        self.columns.append(name)
        self._column_contents[name] = self._reflect_references("Tower_Particles",
                                                               shape_ref, depth=2)

    def _recursive_to_id(self, jagged_array):
        """
        For a given, unknown depth, iterable convert any trefs into indices.
        Assumes either all non iterables are trefs or none are.

        Parameters
        ----------
        jagged_array : iterable
            array that may contain uproot.rootio.trefs

        Returns
        -------
        is_tRef : bool
            if trefs have been found and converted
        results : iterable
            output of conversion attempt, same shape as jagged_array
        """
        results = []
        is_tRef = True  # an empty list is assumed to be a tRef list
        for item in jagged_array:
            if hasattr(item, '__iter__'):
                is_tRef, sub_results = self._recursive_to_id(item)
                if not is_tRef:
                    # cut losses and return up the stack now
                    return is_tRef, results
                results.append(sub_results)
            else:
                is_tRef = isinstance(item, uproot.rootio.TRef)
                if not is_tRef:
                    # cut losses and return up the stack now
                    return is_tRef, results
                # the trefs are 1 indexed not 0 indexed, so subtract 1
                results.append(item.id - 1)
        return is_tRef, results

    def _reflect_references(self, reference_col, target_shape_col, depth=1):
        """
        From a list of pointers a->b and an object that has the shape of b
        create a list of pointers b->a

        Parameters
        ----------
        reference_col : string
            name of column with integer pointers from a->b, depth 1 or 2
        target_shape_col : iterable
            name of column with shape of b
        depth : int
            the depth at which the pointers refer to objects in b
            (Default value = 1)

        Returns
        -------
        reflection : awkward array of ints
            array of pointers from b->a
        
        """
        references = getattr(self, reference_col)
        target_shape = getattr(self, target_shape_col)
        reflection = []
        for event_shape, event in zip(target_shape, references):
            event_reflection = [-1 for _ in event_shape]
            if depth == 1:
                # this is the level on which the indices refer
                for i, ref in enumerate(event):
                    event_reflection[ref] = i
            elif depth == 2:
                # this is the level on which the indices refer
                for i, ref_list in enumerate(event):
                    for ref in ref_list:
                        event_reflection[ref] = i
            else:
                raise NotImplementedError
            reflection.append(event_reflection)
        reflection = awkward.fromiter(reflection)
        return reflection

    def _fix_Birr(self):
        """
        for reasons known unto god and some lonely coder
        some momentum values are incorrectly set to zero fix them
        """
        self.selected_index = None
        pt = self.PT
        birr = self.Birr
        pz = self.Pz
        n_events = len(birr)
        for event_n in range(n_events):
            is_zero = np.abs(birr[event_n] < 0.001)
            calc_birr = np.sqrt(pt[event_n][is_zero]**2 + pz[event_n][is_zero]**2)
            birr[event_n][is_zero] = calc_birr
        self._column_contents["Birr"] = birr

    def _fix_Tower_NTimeHits(self):
        """ """
        particles = self.Tower_Particles
        times_hit = apply_array_func(len, particles)
        self._column_contents["Tower_NTimeHits"] = times_hit

    def _insert_inf_rapidities(self):
        """ use np.inf not 999.9 for infinity, replace this number with np.inf"""
        def big_to_inf(arry):
            """
            Replace 999.9 with np.inf

            Parameters
            ----------
            arry : mutable array like
                array in which to make substitutions

            Returns
            -------
            arry : mutable array like
                array with substitutions made
            """
            # we expect inf to be 999.9
            arry[np.nan_to_num(np.abs(arry)) > 999.] *= np.inf
            return arry
        for name in self._column_contents.keys():
            if "Rapidity" in name:
                new_values = apply_array_func(big_to_inf, self._column_contents[name])
                self._column_contents[name] = new_values

    def _remove_Track_Birr(self):
        """ Remove Track_Birr from self """
        name = "Track_Birr"
        self.columns.remove(name)
        del self._column_contents[name]

    def write(self):
        """ Overwrite the write function in EventWise becuase RootReadout is not writable """
        raise NotImplementedError("This interface is read only")

    @classmethod
    def from_file(cls, path, component_names):
        """
        Read a root file from file, by specifying the full path and components to read

        Parameters
        ----------
        path : string
            path to root file
        component_names : list of strings
            names of the subcomponents
            inside the specified component to be read

        Returns
        -------
        : RootReadout
            data read from disk
        """
        return cls(*os.path.split(path), component_names)


def fix_nonexistent_columns(eventWise):
    """
    Sometimes when a manipulation goes wrong entries are added to the 
    columns (or hyperparameter_columns) list of an eventwise that don't corrispond to
    any content. This removes them and reports what was removed.

    Parameters
    ----------
    eventWise : EventWise
        data with pottential problems

    Returns
    -------
    h_problems: list of strings
        names of removed hyperparameters
    eventWise : EventWise
        data with problems removed
    """
    problems = []
    for name in eventWise.columns:
        if name not in eventWise._column_contents:
            problems.append(name)
    print(f"removing {problems} from columns")
    for name in problems:
        eventWise.columns.remove(name)
    h_problems = []
    for name in eventWise.hyperparameter_columns:
        if name not in eventWise._column_contents:
            h_problems.append(name)
    print(f"removing {h_problems} from hyperparameter columns")
    for name in h_problems:
        eventWise.hyperparameter_columns.remove(name)
    return h_problems, eventWise


def check_even_length(eventWise, interactive=True, raise_error=False, ignore_prefixes=None):
    """
    Check that all the items in the list of columns contain the same
    number of events (i.e. have the same length). either delete or
    raise errors for columns with the wrong length.
    Has interactive mode. Doesn't write changes.
    No return value, works in place.

    Parameters
    ----------
    eventWise : EventWise
        data to check
    interactive : bool
        Should the user be asked about changes?
         (Default value = True)
    raise_error : bool
        Should the function raise an error
        instead of deleteing columns of the wrong length?
         (Default value = False)
    ignore_prefixes : list of strings
        Prefixes of columns not to check
         (Default value = None)

    """
    eventWise.selected_index = None
    if ignore_prefixes is None:
        ignore_prefixes = []
    column_lengths = {name: len(getattr(eventWise, name)) for name in eventWise.columns}
    max_len = np.max(list(column_lengths.values()), initial=0)
    if interactive and not InputTools.yesNo_question(
            f"Max length is {max_len}, require this length? "):
        max_len = InputTools.get_literal("What should the length be? ")
    ask_apply_same_choice = 0
    apply_to_prefix = True
    prefix_to_remove = []
    for name, length in column_lengths.items():
        remove = False
        # get the prefix
        prefix = name.split('_')[0]
        if prefix == name:
            prefix = ''
        if prefix in ignore_prefixes:
            continue
        if prefix in prefix_to_remove:  # ded we already to decide to remove this?
            remove = True
        elif length != max_len:  # is this one a problem?
            problem = f"Column {name} has {length} items, (should be {max_len})"
            if raise_error:
                raise ValueError(problem)
            if interactive:
                print(problem)
                remove = InputTools.yesNo_question("Remove this column? ")
                apply_to_prefix = InputTools.yesNo_question(
                    f"Apply this choice to all columns with prefix '{prefix}'? ")
                ask_apply_same_choice += 1
                if ask_apply_same_choice % 5 == 0:
                    interactive = not InputTools.yesNo_question(
                        "Do you want to apply this choice to all future columns? ")
            else:
                remove = True
                prefix_to_remove.append(prefix)
        if remove:
            eventWise.remove(name)


def check_no_tachions(eventWise, interactive=True, raise_error=False, ignore_prefixes=None, relaxation=0.0001):
    """
    Check no objects in an eventwise appear to go faster than light.
    Works in place, dosn't write changes to disk.

    Parameters
    ----------
    eventWise : EventWise
        data to check
    interactive : bool
        Ask user before changing data
         (Default value = True)
    raise_error : bool
        Raise an error insead of deleteing the columns
        when a tachyon is found.
         (Default value = False)
    ignore_prefixes : list of str
        Prefixes not to check
         (Default value = None)
    relaxation : float
        amount by which the invarient mass squared may be negative
        without considering the particle tachyonic
         (Default value = 0.0001)
    """
    eventWise.selected_index = None
    if ignore_prefixes is None:
        ignore_prefixes = []
    else:
        for i in range(len(ignore_prefixes)):
            if ignore_prefixes[i] and not ignore_prefixes[i].endswith('_'):
                ignore_prefixes[i] += '_'
    energy_prefixes = [name[:-len('Energy')] for name in eventWise.columns
                       if name.endswith('Energy')]
    ask_apply_same_choice = 0
    for prefix in energy_prefixes:
        if prefix in ignore_prefixes:
            continue
        try:
            energy = getattr(eventWise, prefix + "Energy")
            px = getattr(eventWise, prefix + "Px")
            py = getattr(eventWise, prefix + "Py")
            pz = getattr(eventWise, prefix + "Pz")
            while hasattr(energy[0], '__iter__'):
                energy = energy.flatten()
                px = px.flatten()
                py = py.flatten()
                pz = pz.flatten()
        except AttributeError:
            continue  # thos doesn't have enough columns to make that error
        restmass2 = energy**2 - px**2 - py**2 - pz**2
        if np.all(restmass2 >= -relaxation):
            continue  # it's not tachyonic
        remove = False
        min_restmass2 = np.min(restmass2)
        problem = f"Prefix {prefix} is tachyonic with min rest mass {min_restmass2}"
        if raise_error:
            raise ValueError(problem)
        if interactive:
            print(problem)
            remove = InputTools.yesNo_question("Remove this column? ")
            ask_apply_same_choice += 1
            if ask_apply_same_choice % 5 == 0:
                interactive = not InputTools.yesNo_question(
                    "Do you want to apply this choice to all future prefixes? ")
        else:
            remove = True
        if remove:
            eventWise.remove_prefix(prefix)


def find_eventWise_in_dir(dir_name):
    """
    Look though a directory for valid eventWise saves.
    Not recursive.

    Parameters
    ----------
    dir_name : string
        directory to look in

    Returns
    -------
    eventWises : list of EventWise
        loaded eventWise objects found in dir
    
    """
    eventWises = []
    content = os.listdir(dir_name)
    for name in content:
        if name.endswith('awkd'):
            path = os.path.join(dir_name, name)
            try:
                eventWise = EventWise.from_file(path)
                eventWises.append(eventWise)
            except Exception:
                print(f"{name} is not an eventWise")
    return eventWises


