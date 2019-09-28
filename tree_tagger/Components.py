"""Low level components, format apes that of root """
import warnings
import uproot
import awkward
import itertools
import contextlib
import os
import csv
from ipdb import set_trace as st
import numpy as np
from skhep import math as hepmath
from tree_tagger import Constants, PDGNames, InputTools

def flatten(nested):
    for part in nested:
        if hasattr(part, '__iter__'):
            yield from flatten(part)
        else:
            yield part
    raise StopIteration

def detect_depth(nested):
    # assumed arrays are wider than deep
    max_depth = 2
    for x in nested:
        if not hasattr(x, '__iter__'):
            return True, 0  # absolute end found
        elif len(x) > 0:
            abs_end, x_depth = detect_depth(x) 
            if abs_end:
                return abs_end, x_depth+1
            max_depth = max(max_depth, x_depth+1)
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
    if depth==1:
        for parts in zip(*nested):
            if len(parts[0]) == 0:
                out.append([])
            else:
                out.append(func(*parts))
    else:
        for parts in zip(*nested):
            out.append(_apply_array_func(func, depth-1, *parts))
    return awkward.fromiter(out)


def confine_angle(angle):
    """ confine an angle between -pi and pi"""
    return ((angle + np.pi)%(2*np.pi)) - np.pi


# context manager for folder that holds data
class DataIndex:
    index_delimiter = '|'
    index_columns = [('name', str), ('mutable', bool), ('size', int), ('save_name', str)]
    def __init__(self, dir_name):
        """ Context manager for folder that holds data
        Designed to provide some checks that the contents of the folder are correct,
        also provides paths fo opaning them """
        self.dir_name = dir_name
        # fetch the index file
        self.index_name = os.path.join(dir_name, "index.txt")
        self.contents = []
        try:
            with open(self.index_name, 'r') as index_file:
                reader = csv.reader(index_file, delimiter=self.index_delimiter)
                for row in reader:
                    assert len(row) == len(self.index_columns)
                    processed = {col: col_type(x) for ((col, col_type), x)
                                 in zip(self.index_columns, row)}
                    self.contents.append(processed)
            self.verify_index()
        except FileNotFoundError:  # new Datafolder, make an index
            self.contents = self._generate_contents()
            self._write_index()

    def _generate_contents(self):
        print(f"Creating index for {self.dir_name}")
        in_dir = os.listdir(self.dir_name)
        contents = []
        for save_name in in_dir:
            print(save_name)
            name = input("name= ")
            is_mutable = InputTools.yesNo_question("mutable= ")
            path = os.path.join(self.dir_name, save_name)
            size = os.stat(path).st_size
            contents.append({'name': name, 'mutable': is_mutable,
                             'size': size, 'save_name': save_name})
        return contents

    def verify_index(self, check_mutables=False):
        """ check the index file is correct about the contents of the Data folder """
        in_dir = os.listdir(self.dir_name)
        index_save = os.path.split(self.index_name)[-1]
        # check for the index file itself
        if index_save in in_dir:
            in_dir.remove(index_save)
        else:
            raise FileNotFoundError(f"{index_save} not in {self.dir_name}")
        # look for other files
        for file_data in self.contents:
            if file_data['save_name'] in in_dir:
                in_dir.remove(file_data['save_name'])
            else:
                raise FileNotFoundError(f"{file_data['save_name']} not in {self.dir_name}")
            path = os.path.join(self.dir_name, file_data['save_name'])
            if not file_data['mutable'] or check_mutables:
                found_size = os.stat(path).st_size
                if found_size != file_data['size']:
                    raise OSError(f"{path} is size {found_size} - expected size={file_data['size']}")
        # if the file isn't in the index it shouldn't be in the dir
        if len(in_dir) > 0:
            raise FileExistsError(f"Unindexed files found in {self.dir_name}; {in_dir}")

    def _write_index(self):
        with open(self.index_name, 'w') as index_file:
            writer = csv.writer(index_file, delimiter=self.index_delimiter)
            for entry in self.contents:
                line = [str(entry[col_name]) for col_name, _ in self.index_columns]
                writer.writerow(line)

    def __enter__(self):
        return self.contents

    def __exit__(self, *args):
        for entry in self.contents:
            if 'size' not in entry:
                path = os.path.join(self.dir_name, entry['save_name'])
                entry['size'] = os.stat(path).st_size
        self.verify_index()
        self._write_index()


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


class SafeLorentz(hepmath.LorentzVector):
    """ Like the hepmath lorentz vector class,
        but with some substritutions for robust behavior """
        
    def rapidity(self):
        """ overwrite the method in LorentzVector with a more robust method """
        if self.perp2 == 0 and self.e == abs(self.pz):
            large_num = 10**10
            return np.sign(self.pz)*large_num + self.pz
        if self.pz == 0.:
            return 0.
        m2 = max(self.m2, 0.)
        mag_rap = 0.5*np.log((self.perp2 + m2)/((self.e + abs(self.pz))**2))
        return -np.sign(self.pz) * mag_rap

    @property
    def et(self):
        if self.p == 0:
            return 0
        else:
            return super(SafeLorentz, self).et

    def setptetaphie(self, pt, eta, phi, e):
        # the default method, but using numpy to deal with possible infinity in pz
        inputs = [pt*np.cos(phi), pt*np.sin(phi),
                  pt*np.nan_to_num(np.sinh(eta), False),
                  e]
        self.setpxpypze(*inputs)


class EventWise:
    """ The most basic properties of collections that exist in an eventwise sense """
    # putting them out here ensures they will be overwritten
    columns = []
    _column_contents = {}
    auxilleries = []
    selected_index = None
    EVENT_DEPTH=1 # events, objects in events
    JET_DEPTH=2 # events, jets, objects in jets

    def __init__(self, dir_name, save_name, columns=None, contents=None):
        # the init method must generate some table of items,
        # nomally a jagged array
        self.dir_name = dir_name
        self.save_name = save_name
        if columns is not None:
            self.columns = columns
        if contents is not None:
            self._column_contents = contents
        assert len(set(self.columns)) == len(self.columns), f"Duplicates in columns; {self.columns}"

    def add_to_index(self, contents, name=None, mutable=True):
        """ add this dataset to the file index """
        if name is None:
            name = self.save_name.split('.', 1)[0]
        contents.append({'name': name, 'save_name': self.save_name, 'mutable':mutable})

    def __getattr__(self, attr_name):
        """ the columns are all avalible attrs """
        # capitalise raises the case of the first letter
        if attr_name[0].upper() + attr_name[1:] in self.columns:
            if self.selected_index is not None:
                return self._column_contents[attr_name][self.selected_index]
            return self._column_contents[attr_name]
        raise AttributeError(f"{self.__class__.__name__} does not have {attr_name}")

    def __dir__(self):
        new_attrs = set(super().__dir__() + self.columns)
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
        column_order = awkward.fromiter(self.columns)
        try:
            del self._column_contents['column_order']
        except KeyError:
            pass
        all_content = {'column_order': column_order, **self._column_contents}
        awkward.save(path, all_content, mode='w')

    def append(self, *args):
        if len(args) == 2:
            new_columns = args[0]
            new_content = args[1]
            for name in new_columns:
                if name in self.columns:
                    self.remove(name)
        else:
            new_content = args[0]
            new_columns = new_content.keys()
            for name in new_columns:
                if name in self.columns:
                    self.remove(name)
        self.columns += new_columns
        self._column_contents = {**self._column_contents, **new_content}
        self.write()

    def remove(self, col_name):
        if col_name not in self.columns:
            raise KeyError(f"Don't have a column called {col_name}")
        self.columns.remove(col_name)
        if type(self._column_contents) != dict:
            self._column_contents = {k:v for k, v in self._column_contents.items()}
        del self._column_contents[col_name]

    def remove_prefix(self, col_prefix):
        to_remove = [c for c in self.columns if c.startswith(col_prefix)]
        for c in to_remove:
            self.remove(c)

    @classmethod
    def from_file(cls, path):
        contents = awkward.load(path)
        columns = list(contents['column_order'])
        new_eventWise = cls(*os.path.split(path), columns=columns, contents=contents)
        return new_eventWise


def add_rapidity(eventWise, base_name=''):
    if base_name != '':
        if not base_name.endwith('_'):
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
        for pt, pz, e in zip(pts, pzs, es):
            if pt == 0 and e == np.abs(pz):
                large_num = 10**10
                rap_here.append(np.sign(pz)*large_num + pz)
            elif pz == 0.:
                rap_here.append(0)
            else:
                m2 = e**2 - pz**2 - pt**2
                m2 = min(m2, 0.)  # m2 should be strictly positive, but floating point happens
                mag_rap = 0.5*np.log((pt**2 + m2)/((e + np.abs(pz))**2))
                rap_here.append(-np.sign(pz) * mag_rap)
        rapidities.append(awkward.fromiter(rap_here))
    eventWise.selected_index = None
    rapidities = awkward.fromiter(rapidities)
    eventWise.append({base_name+"Rapidity": rapidities})


def add_thetas(eventWise, basename=None):
    columns = []
    contents = {}
    if basename is None:
        # find all the things with an angular property
        phi_cols = [c[:-4] for c in eventWise.columns if c.endswith("_Phi")]
        missing_theta = [c for c in phi_cols if (c+"_Theta") not in eventWise.columns]
    else:
        missing_theta = [basename]
    for name in missing_theta:
        avalible_components = [c[len(name)+1:] for c in eventWise.columns
                               if c.startswith(name)]
        if (set(("PT", "Pz")) <= set(avalible_components) or
            set(("Birr", "PT")) <= set(avalible_components) or
            set(("Birr", "Pz")) <= set(avalible_components) or
            set(("Px", "Py", "Pz")) <= set(avalible_components)):
            # then we will work with momentum
            if "Pz" in avalible_components:
                pz = getattr(eventWise, name+"_Pz")
                if "Birr" in avalible_components:
                    birr = getattr(eventWise, name+"_Birr")
                    theta = np.arccos(pz/birr)
                else:
                    if "PT" in avalible_components:
                        pt = getattr(eventWise, name+"_PT")
                    else:
                        pt = np.sqrt(getattr(eventWise, name+"_Px")**2 +
                                     getattr(eventWise, name+"_Py")**2)
                    theta = np.arctan(pt/pz)
            else:
                birr = getattr(eventWise, name+"_Birr")
                pt = getattr(eventWise, name+"_PT")
                theta = np.arcsin(pt/birr)
        elif "ET" in avalible_components:
            # we work with energy
            et = getattr(eventWise, name+"_ET")
            e = getattr(eventWise, name+"_Energy")
            theta = np.arcsin(et/e)
        else:
            print(f"Couldn't calculate Theta for {name}")
            continue
        columns.append(name + "_Theta")
        contents[name+"_Theta"] = theta
    if "Theta" not in eventWise.columns:
        theta = np.arctan(eventWise.PT/eventWise.Pz)
        columns.append("Theta")
        contents["Theta"] = theta
    eventWise.append(columns, contents)


def add_pseudorapidity(eventWise, basename=None):
    columns = []
    contents = {}
    if basename is None:
        # find all the things with theta
        theta_cols = [c[:-5] for c in eventWise.columns if c.endswith("_Theta")]
        missing_ps = [c for c in theta_cols if (c+"PseudoRapidity") not in eventWise.columns]
        if "Theta" in eventWise.columns and "PseudoRapidity" not in eventWise.columns:
            missing_ps.append('')
    else:
        if basename == '':
            missing_ps = [basename]
        else:
            missing_ps = [basename + '_']
    for name in missing_ps:
        theta = getattr(eventWise, name+"Theta")
        pseudorapidity = apply_array_func(theta_to_pseudorapidity, theta)
        columns.append(name + "PseudoRapidity")
        contents[name+"PseudoRapidity"] = pseudorapidity
    eventWise.append(columns, contents)


def theta_to_pseudorapidity(theta_list):
    tan_theta = np.tan(theta_list)
    mag_tan_theta = np.abs(tan_theta)
    infinite = mag_tan_theta <= 0.
    pseudorapidity = np.full_like(theta_list, np.inf)
    pseudorapidity[~infinite] = np.log(mag_tan_theta/2)
    pseudorapidity *= -np.sign(tan_theta)
    return pseudorapidity

class RootReadout(EventWise):
    """ Reads arbitary components from a root file created by Delphes """
    def __init__(self, dir_name, save_name, component_names):
        # read the root file
        path = os.path.join(dir_name, save_name)
        self._root_file = uproot.open(path)["Delphes"]
        # the first component has no prefix
        prefixes = [''] + component_names[1:]
        for prefix, component in zip(prefixes, component_names):
            self.add_component(component, prefix)
        self._unpack_TRefs()
        super().__init__(dir_name, save_name)

    def add_component(self, component_name, key_prefix=''):
        if key_prefix != '':
            # check it ends in an underscore
            if not key_prefix.endswith('_'):
                key_prefix += '_'
            # check that this prefix is not seen anywhere else
            for key in self._column_contents.keys():
                assert not key.startswith(key_prefix)
        # remove keys starting with lower case letters (they seem to be junk)
        full_keys = [k for k in self._root_file[component_name].keys()
                     if (k.decode().split('.', 1)[1][0]).isupper()]
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
            new_attr = key_prefix + new_attr
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
        assert len(set(attr_names)) == len(attr_names), f"Duplicates in columns; {attr_names}"
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
                        try:
                            event_reflection[ref] = i
                        except IndexError:
                            st()
                            print(i)
            else:
                raise NotImplementedError
            reflection.append(event_reflection)
        reflection = awkward.fromiter(reflection)
        return reflection
                
    def write(self):
        raise NotImplementedError("This interface is read only")

    @classmethod
    def from_file(cls, path, component_name):
        return cls(*os.path.split(path), component_name)

