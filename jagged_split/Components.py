"""Low level components, format apes that of root """
import pickle
import awkward
import os
from ipdb import set_trace as st
import numpy as np

class xEventWise:
    def __init__(self, dir_name, save_name, contents=None):
        self.save_name = save_name
        self.dir_name = dir_name
        if contents is None:
            self.contents = {}
        else:
            self.contents = contents

    def split(self, lower_bounds, upper_bounds, per_event_component="Energy", part_name="part", **kwargs):
        name_format = self.save_name.split('.', 1)[0] + "_" + part_name + "{}.awkd"
        save_dir = os.path.join(self.dir_name, name_format.replace('{}.awkd', ''))
        try:
            os.mkdir(save_dir)
        except FileExistsError:
            pass
        # not all the components are obliged to have the same number of items,
        # if a component is identified as being the correct length, and all components
        # of that length are split between the fragments
        n_events = len(self.contents[per_event_component])
        # work out which lists have this property
        per_event_cols = [c for c in self.contents.keys()
                          if len(self.contents[c]) == n_events]
        new_contents = []
        for lower, upper in zip(lower_bounds, upper_bounds):
            if lower > upper:
                raise ValueError(f"lower bound {lower} greater than upper bound {upper}")
            elif lower == upper:
                # append none as a placeholder
                new_contents.append(None)
            else:
                new_content = {k: self.contents[k][lower:upper] for k in per_event_cols}
                new_contents.append(new_content)
        all_ew = []
        i = 0
        add_unsplit = True
        for new_content in new_contents:
            if new_content is None:
                all_ew.append(None)
                continue
            # this bit isn't thread safe and could create a race condition
            while name_format.format(i) in os.listdir(save_dir):
                i+=1
            name = name_format.format(i)
            ew = type(self)(save_dir, name, contents=new_content)
            all_ew.append(ew)
        return all_ew

class EventWise:
    """ The most basic properties of collections that exist in an eventwise sense """
    # putting them out here ensures they will be overwritten
    auxilleries = []
    selected_index = None
    EVENT_DEPTH=1 # events, objects in events
    JET_DEPTH=2 # events, jets, objects in jets
    _alias_dict = {}  # need to make a blank defalut alias dict incase the __getattr__
    # method is calle dbefore th eobject is initialised

    def __init__(self, dir_name, save_name, contents=None):
        # the init method must generate some table of items,
        # nomally a jagged array
        self.dir_name = dir_name
        self.save_name = save_name
        # by using None as the default value it os possible to detect non entry
        if contents is not None:
            self.contents = {k:v for k, v in contents.items()}
        else:
            self.contents = {}
    
    def __getattr__(self, attr_name):
        """ the columns are all avalible attrs """
        # capitalise raises the case of the first letter
        attr_name = attr_name[0].upper() + attr_name[1:]
        # if it is an alias get the true name
        attr_name = self._alias_dict.get(attr_name, attr_name)
        if attr_name in self.contents:
            if self.selected_index is not None:
                return self.contents[attr_name][self.selected_index]
            return self.contents[attr_name]
        raise AttributeError(f"{self.__class__.__name__} does not have {attr_name}")


    def __dir__(self):
        new_attrs = set(super().__dir__() + self.contents.keys())
        return sorted(new_attrs)

    def __str__(self):
        msg = f"<EventWise with {len(self.columns)} columns; {os.path.join(self.dir_name, self.save_name)}>"
        return msg

    def write(self):
        """ write to disk """
        path = os.path.join(self.dir_name, self.save_name)
        awkward.save(path, self.contents, mode='w')

    @classmethod
    def from_file(cls, path):
        contents = awkward.load(path)
        new_eventWise = cls(*os.path.split(path), contents=contents)
        return new_eventWise

    def append(self, *args):
        if len(args) == 2:
            new_columns = args[0]
            new_content = args[1]
            assert sorted(new_columns) == sorted(new_content.keys())
        else:
            new_content = args[0]
            new_columns = new_content.keys()
        # enforce the first letter of each attrbute to be capital
        New_columns = [c[0].upper() + c[1:] for c in new_columns]
        new_content = {C: new_content[c] for C, c in zip(New_columns, new_columns)}
        # delete existing duplicates
        for name in New_columns:
            if name in self.contents.keys():
                self.remove(name)
        self.contents = {**self.contents, **new_content}
        self.write()

    def remove(self, col_name):
        if col_name not in self.contents:
            raise KeyError(f"Don't have a column called {col_name}")
        if type(self.contents) != dict:
            self.contents = {k:v for k, v in self.contents.items()}
        del self.contents[col_name]

    def remove_prefix(self, col_prefix):
        to_remove = [c for c in self.contents if c.startswith(col_prefix)]
        for c in to_remove:
            self.remove(c)


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
        per_event_cols = [c for c in self.contents.keys()
                          if len(self.contents[c]) == n_events]
        assert to_check.issubset(per_event_cols)
        new_contents = []
        for lower, upper in zip(lower_bounds, upper_bounds):
            if lower > upper:
                raise ValueError(f"lower bound {lower} greater than upper bound {upper}")
            elif lower == upper:
                # append none as a placeholder
                new_contents.append(None)
            else:
                new_content = {k: self.contents[k][lower:upper] for k in per_event_cols}
                new_contents.append(new_content)
        unchanged_parts = {k: self.contents[k][:] for k in self.contents.keys()
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
                ew = type(self)(save_dir, name, contents=new_content)
                if no_dups:
                    # don't do this again
                    add_unsplit = False
            else:
                ew = type(self)(save_dir, name, contents=new_content)
            all_paths.append(os.path.join(save_dir, name))
            ew.write()
        return all_paths

    @classmethod
    def combine(cls, dir_name, save_base, fragments=None, check_for_dups=False, del_fragments=False):
        in_dir = os.listdir(dir_name)
        if fragments is None:
            fragments = [name for name in in_dir
                         if name.startswith(save_base) 
                         and name.endswith(".awkd")]
        columns = []
        contents = {}
        for fragment in fragments:
            path = os.path.join(dir_name, fragment)
            content_here = awkward.load(path)
            pickle_strs = {}  # when checking for dups
            for key in content_here:
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
            contents[key] = awkward.fromiter(contents[key])
        new_eventWise = cls(dir_name, save_base+"_joined.awkd", contents=contents)
        new_eventWise.write()
        if del_fragments:
            for fragment in fragments:
                os.remove(os.path.join(dir_name, fragment))
        return new_eventWise

