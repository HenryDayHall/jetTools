"""Low level components, format apes that of root """
import pickle
import warnings
import uproot
import awkward
import itertools
import contextlib
import os
import csv
from ipdb import set_trace as st
import numpy as np
from tree_tagger import Constants, PDGNames, InputTools

def flatten(nested):
    for part in nested:
        if hasattr(part, '__iter__'):
            yield from flatten(part)
        else:
            yield part
    return


def detect_depth(nested):
    # assumed arrays are wider than deep
    max_depth = 0
    for x in nested:
        # some thigs have ittr methods, but are the end goal
        if isinstance(x, (str, bytes)) or not hasattr(x, '__iter__'):
            return True, 0  # absolute end found
        elif len(x) > 0:
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
    if depth is None:
        abs_end, depth = detect_depth(nested[0])
        if not abs_end:  # no objects in array
            print("Warning, no object found in array!")
    return _apply_array_func(func, depth, *nested)


def _apply_array_func(func, depth, *nested):
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
    """ confine an angle between -pi and pi"""
    return ((angle + np.pi)%(2*np.pi)) - np.pi


def safe_convert(cls, string):
    """ Safe conversion out of strings
    designed to deal with None gracefully, and identify sensible bools

    Parameters
    ----------
    cls : class
        the class to convert to
    string : str
        the string that needs converting

    Returns
    -------
    : object
        converted object, either None type or type cls
    """
    if string == "None": return None
    elif cls == bool: 
        return Constants.lowerCase_truthies.get(string.lower(), True)
    else: return cls(string)


class EventWise:
    """ The most basic properties of collections that exist in an eventwise sense """
    selected_index = None
    EVENT_DEPTH=1 # events, objects in events
    JET_DEPTH=2 # events, jets, objects in jets

    def __init__(self, dir_name, save_name, columns=None, contents=None, hyperparameter_columns=None):
        self._loaded_contents = {}  # keep columns that have been accessed in ram
        # the init method must generate some table of items,
        # nomally a jagged array
        self.dir_name = dir_name
        self.save_name = save_name
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
    
    def _gen_alias(self):
        alias_dict = {}
        if 'alias' in self._column_contents:
            for row in self._column_contents['alias']:
                alias_dict[row[0]] = row[1]
                self.columns.append(row[0])
        else:
            self._column_contents['alias'] = awkward.fromiter([])
        return alias_dict

    def _remove_alias(self, to_remove):
        alias_list = list(self._column_contents['alias'][:, 0])
        alias_idx = alias_list.index(to_remove)
        # if to_remove is not infact an alias the line above will throw an error
        del self._alias_dict[to_remove]
        del alias_list[alias_idx]
        self._column_contents['alias'] = awkward.fromiter(alias_list)
        self.columns.remove(to_remove)

    def add_alias(self, name, target):
        assert target in self.columns
        assert name not in self.columns
        alias_list = self._column_contents['alias']
        alias_list += [name, target]
        self._column_contents['alias'] = awkward.fromiter(alias_list)
        self._alias_dict[name] = target
        self.columns.append(name)

    def __getattr__(self, attr_name):
        """ the columns are all avalible attrs """
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
                self._loaded_contents[attr_name] = self._column_contents[attr_name][:]
                return getattr(self, attr_name)
            except AttributeError: # we dont have a loaded dict yet
                self._loaded_contents = {}
                return getattr(self, attr_name)
        if attr_name in self.hyperparameter_columns:
            return self._column_contents[attr_name]
        raise AttributeError(f"{self.__class__.__name__} does not have {attr_name}")

    def match_indices(self, attr_name, match_from, match_to=None, event_n=None):
        """
        return the elements of each row for which match_for and match_to are equal
        one of match_from or match_to may be projected to make the required dimentions
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
                out = [row[f==t] for f, t, row in zip(match_from, match_to, attr)]
            except TypeError:
                mask = match_from == match_to
                out = [row[m] for m, row in zip(mask, attr)]
        else:
            out = [row[m] for m, row in zip(match_from, attr)]
        return awkward.fromiter(out)

    def __dir__(self):
        new_attrs = set(super().__dir__() + self.columns + self.hyperparameter_columns)
        return sorted(new_attrs)

    def __str__(self):
        msg = f"<EventWise with {len(self.columns)} columns; {os.path.join(self.dir_name, self.save_name)}>"
        return msg

    def __eq__(self, other):
        return self.save_name == other.save_name and self.dir_name == other.dir_name

    def write(self):
        """ write to disk """
        path = os.path.join(self.dir_name, self.save_name)
        assert len(self.columns) == len(set(self.columns)), "Columns contains duplicates"
        non_alias_cols = [c for c in self.columns if c not in self._alias_dict]
        column_order = awkward.fromiter(non_alias_cols)
        try:
            del self._column_contents['column_order']
        except KeyError:
            pass
        non_alias_hcols = [c for c in self.hyperparameter_columns if c not in self._alias_dict]
        hyperparameter_column_order = awkward.fromiter(non_alias_hcols)
        try:
            del self._column_contents['hyperparameter_column_order']
        except KeyError:
            pass
        all_content = {'column_order': column_order,
                       'hyperparameter_column_order': hyperparameter_column_order,
                       **self._column_contents}
        awkward.save(path, all_content, mode='w')

    @classmethod
    def from_file(cls, path):
        contents = awkward.load(path)
        columns = list(contents['column_order'])
        hyperparameter_columns = list(contents['hyperparameter_column_order'])
        new_eventWise = cls(*os.path.split(path), columns=columns,
                            hyperparameter_columns=hyperparameter_columns,
                            contents=contents)
        return new_eventWise

    def append(self, **kwargs):
        new_content = kwargs
        new_columns = sorted(kwargs.keys())
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
        self.write()

    def append_hyperparameters(self, **kwargs):
        new_content = kwargs
        new_columns = sorted(kwargs.keys())
        # enforce the first letter of each attrbute to be capital
        New_columns = [c[0].upper() + c[1:] for c in new_columns]
        new_content = {C: new_content[c] for C, c in zip(New_columns, new_columns)}
        # check none of there are itterables asside from strings
        # assumption is that someone trying to set an itterable hyperparameter
        # is actually setting an eventWise object
        #assert all([isinstance(content, str) or not hasattr(content, '__iter__')
        #            for content in new_content.values()])
        # scratch that, there are tuple hyper parameters too
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
        self.write()

    def remove(self, col_name):
        if col_name in self._alias_dict:
            self._remove_alias(col_name)
        else:
            if col_name in self.columns:
                self.columns.remove(col_name)
            elif col_name in self.hyperparameter_columns:
                self.hyperparameter_columns.remove(col_name)
            else:
                raise KeyError(f"Don't have a column called {col_name}")
            if type(self._column_contents) != dict:
                self._column_contents = {k:v for k, v in self._column_contents.items()}
            del self._column_contents[col_name]
            if col_name in self._loaded_contents:
                del self._loaded_contents[col_name]

    def remove_prefix(self, col_prefix):
        to_remove = [c for c in self.columns if c.startswith(col_prefix)]
        for c in to_remove:
            self.remove(c)

    def rename(self, old_name, new_name):
        if old_name in self._alias_dict:
            target = self._alias_dict[old_name]
            self._remove_alias(old_name)
            self.add_alias(new_name, target)
        else:
            if old_name not in self.columns:
                raise KeyError(f"Don't have a column called {old_name}")
            self.columns[self.columns.index(old_name)] = new_name
            if type(self._column_contents) != dict:
                self._column_contents = {k:v for k, v in self._column_contents.items()}
            self._column_contents[new_name] = self._column_contents[old_name]
            del self._column_contents[old_name]

    def fragment(self, per_event_component, **kwargs):
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
        return self.split(lower_bounds, upper_bounds, per_event_component, part_name="fragment", **kwargs)

    def split_unfinished(self, per_event_component, unfinished_component, **kwargs):
        self.selected_index = None
        if not isinstance(per_event_component, str):
            n_events = len(getattr(self, per_event_component[0]))
        else:
            n_events = len(getattr(self, per_event_component))
        if not isinstance(unfinished_component, str):
            # check they all have the same length
            lengths = set([len(getattr(self, name)) for name in unfinished_component])
            assert len(lengths) == 1, "Error not all unfinished components have the same length"
            num_unfinished = list(lengths)[0]
        else:
            num_unfinished = len(getattr(self, unfinished_component))
        assert num_unfinished <= n_events
        if num_unfinished == n_events:
            # the whole thing is unfinished
            return None, os.path.join(self.dir_name, self.save_name)
        if num_unfinished == 0:
            # the event is complete
            return os.path.join(self.dir_name, self.save_dir), None
        lower_bounds = [0, num_unfinished]
        upper_bounds = [num_unfinished, n_events]
        return self.split(lower_bounds, upper_bounds, per_event_component, part_name="progress", **kwargs)

    def split(self, lower_bounds, upper_bounds, per_event_component="Energy", part_name="part", **kwargs):
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
            to_check = set(per_event_component[1:])
            per_event_component = per_event_component[0]
        else:
            to_check = set()
        n_events = len(getattr(self, per_event_component))
        # work out which lists have this property
        per_event_cols = [c for c in self.columns
                          if len(self._column_contents[c]) == n_events]
        per_event_cols += self.hyperparameter_columns
        assert to_check.issubset(per_event_cols)
        new_contents = []
        for lower, upper in zip(lower_bounds, upper_bounds):
            if lower > upper:
                raise ValueError(f"lower bound {lower} greater than upper bound {upper}")
            elif lower == upper:
                # append none as a placeholder
                new_contents.append(None)
            else:
                new_content = {k: self._column_contents[k][lower:upper] for k in per_event_cols}
                new_contents.append(new_content)
        unchanged_parts = {k: self._column_contents[k][:] for k in self._column_contents.keys()
                           if k not in per_event_cols}
        no_dups = kwargs.get('no_dups', True)  # if no dupes only put the unchanged content in the first event
        all_paths = []
        i = 0
        add_unsplit = True
        for new_content in new_contents:
            if new_content is None:
                all_paths.append(None)
                continue
            # this bit isn't thread safe and could create a race condition
            while name_format.format(i) in os.listdir(save_dir):
                i+=1
            name = name_format.format(i)
            if add_unsplit:
                new_content = {**new_content, **unchanged_parts}
                ew = type(self)(save_dir, name,
                                columns=self.columns,
                                hyperparameter_columns=self.hyperparameter_columns,
                                contents=new_content)
                if no_dups:
                    # don't do this again
                    add_unsplit = False
            else:
                ew = type(self)(save_dir, name,
                                columns=per_event_cols,
                                hyperparameter_columns=self.hyperparameter_columns,
                                contents=new_content)
            all_paths.append(os.path.join(save_dir, name))
            ew.write()
        return all_paths

    @classmethod
    def recursive_combine(cls, dir_name, check_for_dups=False, del_framgents=True):
        if dir_name.endswith('/'):
            dir_name = dir_name[:-1]
        root_dir = '/'.join(*dir_name.split('/')[:-1])
        save_base = dir_name.split('/')[-1].split('_', 1)[0]
        dir_content = os.listdir(dir_name)
        fragments = [n for n in dir_content if n.endswith('.awkd')]
        for name in fragments:
            if name[:-4] in dir_content:
                joined_name = cls.recursive_combine(name[:-4], check_for_dups, del_framgents)
                os.replace(joined_name, name)
        combined_eventWise = cls.combine(dir_name, save_base, fragments, check_for_dups, del_fragments=True)
        joined_name = os.path.join(root_dir, save_base+".awkd")
        os.rename(os.path.join(combined_eventWise.dir_name, combined_eventWise.save_name),
                  joined_name)
        os.rmdir(dir_name)
        return joined_name

    @classmethod
    def combine(cls, dir_name, save_base, fragments=None, check_for_dups=False, del_fragments=False):
        in_dir = os.listdir(dir_name)
        if fragments is None:
            fragments = [name for name in in_dir
                         if name.startswith(save_base) 
                         and name.endswith(".awkd")]
        columns = []
        hyperparameter_columns = []
        contents = {}
        for fragment in fragments:
            path = os.path.join(dir_name, fragment)
            content_here = awkward.load(path)
            found_hcols = list(content_here.get("hyperparameter_column_order", []))
            if hyperparameter_columns and found_hcols:  # check they are the same here
                for name in found_hcols:
                    if name not in hyperparameter_columns:
                        hyperparameter_columns.append(name)
                        contents[name] = content_here[name]
                    error_msg = f"Missmatch in hyperparameter {name}"
                    try:
                        np.testing.assert_allclose(content_here[name], contents[name], err_msg=error_msg)
                    except TypeError:
                        assert content_here[name] == contents[name], error_msg
            elif found_hcols:
                hyperparameter_columns = found_hcols
                for name in found_hcols:
                    contents[name] = content_here[name]
            pickle_strs = {}  # when checking for dups
            for key in content_here:
                if key in hyperparameter_columns:
                    continue
                if key not in contents:
                    contents[key] = list(content_here[key])
                elif check_for_dups:
                    # then add iff not a duplicate of the existign data
                    if key not in pickle_strs:
                        pickle_strs[key] = pickle.dumps(content_here[key])
                    elif pickle.dumps(content_here[key]) != pickle_strs[key]:
                        contents[key] += list(content_here[key])

                else:
                    contents[key] += list(content_here[key])
            column_here = list(contents['column_order'])
            for name in column_here:
                if name not in columns:
                    columns.append(name)
        for key in contents.keys():
            if key not in hyperparameter_columns:
                contents[key] = awkward.fromiter(contents[key])
        new_eventWise = cls(dir_name, save_base+"_joined.awkd", columns=columns, contents=contents, hyperparameter_columns=hyperparameter_columns)
        new_eventWise.write()
        if del_fragments:
            for fragment in fragments:
                os.remove(os.path.join(dir_name, fragment))
        return new_eventWise


def event_matcher(eventWise1, eventWise2):
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
    isfloat = np.array([ col == np.float64 for col in column_type])
    float_cols = common_columns[isfloat]
    assert "Event_n" in common_columns
    length1 = len(eventWise1.Event_n)
    order = np.full(length1, -1, dtype=int)
    for i in range(length1):
        possibles = np.where(eventWise2.Event_n == eventWise1.Event_n[i])[0]
        if len(possibles) == 0:
            continue
        int2 = [getattr(eventWise2, col)[possibles] for col in int_cols]
        int1 = [getattr(eventWise1, col)[i] for col in int_cols]
        int_gap = np.zeros_like(possibles, dtype=int)
        for col in int_cols:
            f1 = getattr(eventWise1, col)[i]
            f2s = getattr(eventWise2, col)[possibles]
            try:
                int_gap += [np.int64(np.nan_to_num(recursive_distance(f1, f2))) for f2 in f2s]
            except:
                int_ap = [np.int64(np.nan_to_num(recursive_distance(f1, f2))) for f2 in f2s]
        possibles = possibles[int_gap==0]
        if len(possibles) == 1:
            order[i] = possibles[0]
            continue
        if len(possibles) == 0:
            continue
        float2 = [getattr(eventWise2, col)[possibles] for col in float_cols]
        float1 = [getattr(eventWise1, col)[i] for col in float_cols]
        float_gap = np.zeros_like(possibles, dtype=float)
        for col in float_cols:
            f1 = getattr(eventWise1, col)[i]
            f2s = getattr(eventWise2, col)[possibles]
            float_gap += [recursive_distance(f1, f2) for f2 in f2s]
        best = np.argmin(float_gap)
        order[i] = possibles[best]
    return order
                

def recursive_distance(awkward1, awkward2):
    distance = 0.
    iter1 = hasattr(awkward1, '__iter__')
    iter2 = hasattr(awkward2, '__iter__')
    if iter1 and iter2:
        if len(awkward1) != len(awkward2):
            return np.inf
        return np.sum([recursive_distance(a1, a2) for a1, a2 in zip(awkward1, awkward2)])
    elif iter1 or iter2:
        return np.inf
    else:
        distance += np.abs(awkward1 - awkward2)
    return distance


def add_rapidity(eventWise, base_name=''):
    if base_name != '':
        if not base_name.endswith('_'):
            base_name += '_'
    pts = getattr(eventWise, base_name+"PT")
    pzs = getattr(eventWise, base_name+"Pz")
    es = getattr(eventWise, base_name+"Energy")
    assert not hasattr(pts[0][0], '__iter__')
    n_events = len(getattr(eventWise, base_name+"PT"))
    rapidities = []
    for event_n in range(n_events):
        if event_n % 10 == 0:
            print(f"{100*event_n/n_events}%", end='\r')
        rap_here = []
        eventWise.selected_index = event_n
        pts = getattr(eventWise, base_name+"PT")
        pzs = getattr(eventWise, base_name+"Pz")
        es = getattr(eventWise, base_name+"Energy")
        rap_here = ptpze_to_rapidity(pts, pzs, es)
        rapidities.append(awkward.fromiter(rap_here))
    eventWise.selected_index = None
    rapidities = awkward.fromiter(rapidities)
    eventWise.append(**{base_name+"Rapidity": rapidities})


def add_thetas(eventWise, basename=None):
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
            #if "Pz" in avalible_components:
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
    return np.arctan2(pt_list, pz_list)


def pxpy_to_phipt(px_list, py_list):
    return np.arctan2(py_list, px_list), np.sqrt(px_list**2 + py_list**2)


def theta_to_pseudorapidity(theta_list):
    with np.errstate(invalid='ignore'):
        restricted_theta = np.minimum(theta_list, np.pi - theta_list)
    tan_restricted = np.tan(np.abs(restricted_theta)/2)
    infinite = tan_restricted <= 0.
    pseudorapidity = np.full_like(theta_list, np.inf)
    pseudorapidity[~infinite] = -np.log(tan_restricted[~infinite])
    pseudorapidity[theta_list>np.pi/2] *= -1.
    if not hasattr(theta_list, '__iter__'):
        pseudorapidity = float(pseudorapidity)
    return pseudorapidity


def ptpze_to_rapidity(pt_list, pz_list, e_list):
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
    contents = {}
    if basename is None:
        # find all the things with px, py
        px_cols = [c[:-2] for c in eventWise.columns if c.endswith("Px")]
        pxpy_cols = [c[:-2] for c in eventWise.columns if c.endswith("Py") and c in px_cols]
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
    contents = {}
    if basename is None:
        # find all the things with px, py
        px_cols = [c[:-2] for c in eventWise.columns if c.endswith("Px")]
        pxpy_cols = [c[:-2] for c in eventWise.columns if c.endswith("Py") and c in px_cols]
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


class RootReadout(EventWise):
    """ Reads arbitary components from a root file created by Delphes """
    def __init__(self, dir_name, save_name, component_names, component_of_root_file="Delphes", key_selection_function=None, all_prefixed=False):
        # read the root file
        path = os.path.join(dir_name, save_name)
        self._root_file = uproot.open(path)[component_of_root_file]
        if key_selection_function is not None:
            self._key_selection_function = key_selection_function
        else:
            # remove keys starting with lower case letters (they seem to be junk)
            def func(key):
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
        super().__init__(dir_name, save_name, columns=self.columns, contents=self._column_contents)

    def add_component(self, component_name, key_prefix=''):
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
            if new_attr in rename:
                new_attr = rename[new_attr]
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
        assert len(set(attr_names)) == len(attr_names), f"Duplicates in columns; {[n for i, n in enumerate(attr_names) if n in attr_names[i+1:]]}"
        new_column_contents = {name: all_arrays[key]
                               for key, name in zip(full_keys, attr_names)}
        self._column_contents = {**new_column_contents, **self._column_contents}
        # make this a list for fixed order
        self.columns += sorted(new_column_contents.keys())

    def _unpack_TRefs(self):
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
        """ for reasons known unto god and some lonely coder 
            some momentum values are incorrectly set to zero
            fix them """
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
        particles = self.Tower_Particles
        times_hit = apply_array_func(len, particles)
        self._column_contents["Tower_NTimeHits"] = times_hit

    def _insert_inf_rapidities(self):
        """ we use np.inf not 999.9 for infinity"""
        def big_to_inf(arry):
            # we expect inf to be 999.9
            arry[np.nan_to_num(np.abs(arry))>999.] *= np.inf
            return arry
        for name in self._column_contents.keys():
            if "Rapidity" in name:
                new_values = apply_array_func(big_to_inf, self._column_contents[name])
                self._column_contents[name] = new_values

    def _remove_Track_Birr(self):
        name = "Track_Birr"
        self.columns.remove(name)
        del self._column_contents[name]
                
    def write(self):
        raise NotImplementedError("This interface is read only")

    @classmethod
    def from_file(cls, path, component_name):
        return cls(*os.path.split(path), component_name)


def angular_distance(phi1, phi2):
    angular_diffrence = np.abs(phi1 - phi2) % (2*np.pi)
    return np.minimum(angular_diffrence, 2*np.pi - angular_diffrence)

