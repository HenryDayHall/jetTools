"""Low level components, format apes that of root """
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


class MyParticle(SafeLorentz):
    """Aping genparticle."""
    def __init__(self, *args, **kwargs):
        # shower position
        self.mother_ids = []
        self.daughter_ids = []
        # status
        self.generated_mass = kwargs.get('generated_mass', None)
        # knimatics
        if len(args) == 4:
            # has format x y z t
            # or px py pz e
            super().__init__(*args)
        elif 'px' in kwargs:
            px = kwargs['px']
            py = kwargs['py']
            pz = kwargs['pz']
            if 'm' in kwargs:
                m = kwargs['m']
                super().__init__()
                self.setpxpypzm(px, py, pz, m)
            elif 'e' in kwargs:
                e = kwargs['e']
                super().__init__(px, py, pz, e)
            else:
                super().__init__(px, py, pz, 0.)
        elif 'pt' in kwargs:
            # initilise blank vector
            super().__init__()
            pt = kwargs['pt']
            eta = kwargs['eta']
            phi = kwargs.get('phi', 0.)
            if 'm' in kwargs:
                m = kwargs['m']
                self.setptetaphim(pt, eta, phi, m)
            elif 'e' in kwargs:
                e = kwargs['e']
                self.setptetaphie(pt, eta, phi, e)
            else:
                self.setptetaphie(pt, eta, phi, 0.)

    def __repr__(self):
        body_contents = [getattr(self, name) for name in self.repr_variables]
        body = self.variable_sep.join([repr(x) for x in body_contents])
        mothers = self.variable_sep.join([repr(m) for m in self.mother_ids])
        daughters = self.variable_sep.join([repr(m) for m in self.daughter_ids])
        rep = self.components_sep.join([type(self).__name__, body, mothers, daughters])
        return rep
    id_name = "id"
    repr_variables = [id_name, "generated_mass", "px", "py", "pz", "e"]
    repr_variable_types = [int, float, float, float, float, float]
    variable_sep = ','
    repr_body = variable_sep.join(repr_variables)
    repr_components = ["MyParticle", repr_body, 'mothers', 'daughters']
    components_sep = '|'
    repr_format = components_sep.join(repr_components)

    @classmethod
    def from_repr(cls, rep):
        if rep == '' or rep == 'None': return None
        name, body, mothers, daughters = rep.split(cls.components_sep)
        assert name == cls.__name__
        body_parts = body.split(cls.variable_sep)
        class_dict = {name: safe_convert(v_type, part) for name, v_type, part
                      in zip(cls.repr_variables, cls.repr_variable_types, body_parts)}
        new_particle = cls(**class_dict)
        if mothers != '':
            new_particle.mother_ids = [int(m) for m in mothers.split(cls.variable_sep)]
        if daughters != '':
            new_particle.daughter_ids = [int(d) for d in daughters.split(cls.variable_sep)]
        return new_particle


class EventWise:
    """ The most basic properties of collections that exist in an eventwise sense """
    # putting them out here ensures they will be overwritten
    columns = []
    _column_contents = {}
    auxilleries = []

    def __init__(self, dir_name, save_name, columns=None, contents=None):
        # the init method must generate some table of items,
        # nomally a jagged array
        self.dir_name = dir_name
        self.save_name = save_name
        if columns is not None:
            self.columns = columns
        if contents is not None:
            self._column_contents = contents

    def add_to_index(self, contents, name=None, mutable=True):
        """ add this dataset to the file index """
        if name is None:
            name = self.save_name.split('.', 1)[0]
        contents.append({'name': name, 'save_name': self.save_name, 'mutable':mutable})

    def __getattr__(self, attr_name):
        """ the columns are all avalible attrs """
        # capitalise raises the case of the first letter
        if attr_name[0].upper() + attr_name[1:] in self.columns:
            return self._column_contents[attr_name]
        raise AttributeError(f"{self.__class__.__name__} does not have {attr_name}")

    def __dir__(self):
        new_attrs = set(super().__dir__() + self.columns)
        return sorted(new_attrs)

    def write(self):
        """ write to disk """
        path = os.path.join(self.dir_name, self.save_name)
        column_order = awkward.fromiter(self.columns)
        all_content = {'column_order': column_order, **self._column_contents}
        awkward.save(path, all_content, mode='w')

    @classmethod
    def from_file(cls, path):
        contents = awkward.load(path)
        columns = list(contents['column_order'])
        new_eventWise = cls(*os.path.split(path), columns=columns, contents=contents)
        return new_eventWise


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
            st()
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
        for key in full_keys:
            array = self._root_file.array(key)
        all_arrays = self._root_file.arrays(full_keys)
        # process the keys to ensure they are usable a attribute names and unique
        attr_names = []
        # some attribute names get altered
        rename = {'E': 'Energy', 'P':'Birr'}
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
        new_column_contents = {name: all_arrays[key]
                               for key, name in zip(full_keys, attr_names)}
        self._column_contents = {**new_column_contents, **self._column_contents}
        # make this a list for fixed order
        self.columns += sorted(self._column_contents.keys())

    def _unpack_TRefs(self):
        for name in self.columns:
            try:
                is_tRef, converted = self._recursive_to_id(self._column_contents[name])
            except Exception:
                st()
                is_tRef, converted = self._recursive_to_id(self._column_contents[name])
            if is_tRef:
                converted = awkward.fromiter(converted)
                self._column_contents[name] = converted

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
                results.append(item.id)
        return is_tRef, results
                
    def write(self):
        raise NotImplementedError("This interface is read only")

    @classmethod
    def from_file(cls, path, component_name):
        return cls(*os.path.split(path), component_name)


class RecoParticle(MyParticle):
    """Aping genparticle."""
    def __init__(self, *args, reco_particle_id, **kwargs):
        self.reco_particle_id = reco_particle_id
        self.charge = kwargs.get('charge', None)
        super().__init__(*args, **kwargs)
    
    id_name = "reco_particle_id"
    repr_variables = [id_name, "generated_mass", "charge", "px", "py", "pz", "e"]
    repr_variable_types = [int, float, float, float, float, float, float]
    variable_sep = ','
    repr_body = variable_sep.join(repr_variables)
    repr_components = ["RecoParticle", repr_body, 'mothers', 'daughters']
    components_sep = '|'
    repr_format = components_sep.join(repr_components)

    @classmethod
    def from_MC(cls, mcparticle, reco_particle_id, charge_map=None):
        common_variables = set(cls.repr_variables).intersection(MCParticle.repr_variables)
        kwargs = {name: getattr(mcparticle, name) for name in common_variables}
        kwargs["reco_particle_id"] = reco_particle_id
        if not charge_map:
            charge_map = PDGNames.Identities().charges
        kwargs["charge"] = charge_map[mcparticle.pid]
        return cls(**kwargs)


class MCParticle(MyParticle):
    """Aping genparticle."""
    def __init__(self, *args, global_id, **kwargs):
        # IDs
        self.global_id = global_id
        self.pid = kwargs.get('pid', None)
        self.sql_key = kwargs.get('sql_key', None)
        self.hepmc_barcode = kwargs.get('hepmc_barcode', None)
        # shower position
        self.is_root = kwargs.get('is_root', False)
        self.is_leaf = kwargs.get('is_leaf', False)
        self.start_vertex_barcode = kwargs.get('start_vertex_barcode', None)
        self.end_vertex_barcode = kwargs.get('end_vertex_barcode', None)
        self.mother_ids = []
        self.daughter_ids = []
        # status
        self.status = kwargs.get('status', None)
        super().__init__(*args, **kwargs)

    def __str__(self):
        return f"MCParticle[{self.global_id}] pid;{self.pid} energy;{self.e:.2e}"

    id_name = "global_id"
    repr_variables = [id_name, "pid", "sql_key", "hepmc_barcode",
                      "is_root", "is_leaf",
                      "start_vertex_barcode", "end_vertex_barcode",
                      "status", "generated_mass",
                      "px", "py", "pz", "e"]
    repr_variable_types = [int, int, int, int, 
                           bool, bool,
                           int, int, 
                           int, float,
                           float, float, float, float]
    variable_sep = ','
    repr_body = variable_sep.join(repr_variables)
    repr_components = ["MCParticle", repr_body, 'mothers', 'daughters']
    components_sep = '|'
    repr_format = components_sep.join(repr_components)


class MyVertex(SafeLorentz):
    """Aping GenVertex."""
    def __init__(self, *args, global_vertex_id, **kwargs):
        # IDs
        self.global_vertex_id = global_vertex_id
        self.sql_key = kwargs.get('sql_key', None)
        self.hepmc_barcode = kwargs.get('hepmc_barcode', None)
        # particle interactions
        self.n_out = kwargs.get('n_out', None)
        self.n_weights = kwargs.get('n_weights', None)
        self.n_orphans = kwargs.get('n_orphans', None)
        # location
        if len(args) == 4:
            self.ctau = args[3]
            super().__init__(*args)
        elif 'x' in kwargs:
            x = kwargs.get('x', None)
            y = kwargs.get('y', None)
            z = kwargs.get('z', None)
            self.ctau = kwargs.get('ctau', None)
            super().__init__(x, y, z, self.ctau)


class MyTower(SafeLorentz):
    def __init__(self, *args, global_tower_id, **kwargs):
        # IDs
        self.global_tower_id = global_tower_id
        self._global_obs_id = None
        self.sql_key = kwargs.get('sql_key', None)
        self.particles = kwargs.get('particles', [])
        if self.particles is not None:
            self.global_ids = np.array([p.global_id for p in self.particles])
        else:
            self.global_ids = None
        self._t = kwargs.get('t', None)  # may differ from the Lorentz vector t, so hide it.
        self.nTimeHits = kwargs.get('nTimeHits', None)
        self.eem = kwargs.get('eem', None)
        self.ehad = kwargs.get('ehad', None)
        self.edges = kwargs.get('edges', [None, None, None, None])
        if len(args) == 4:
            # x y z t
            super().__init__(*args)
        elif 'e' in kwargs:
            e = kwargs['e']
            et = kwargs['et']
            eta = kwargs['eta']
            phi = kwargs['phi']
            super().__init__()
            self.setptetaphie(et, eta, phi, e)

    @property
    def global_obs_id(self):
        return self._global_obs_id

    @global_obs_id.setter
    def global_obs_id(self, value):
        if self.particles is not None:
            for particle in self.particles:
                particle.global_obs_id = value
        self._global_obs_id = value

    def __str__(self):
        return f"MyTower[{self.global_tower_id}] energy;{self.e:.2e}"

    def __repr__(self):
        body_content = [self.global_tower_id,
                        self.sql_key,
                        self._t,
                        self.nTimeHits,
                        self.eem,
                        self.ehad,
                        self.edges[0],
                        self.edges[1],
                        self.edges[2],
                        self.edges[3],
                        self.px,
                        self.py,
                        self.pz,
                        self.e]
        # | and , are used inside MyPaticle string rep
        # so diferent deliminators are needed here
        body = '&'.join([repr(x) for x in body_content])
        particles = '&'.join([repr(p) for p in self.particles])
        rep = "!".join([MyTower.__name__, body, particles])
        return rep
    repr_body = '&'.join(["global_tower_id", "sql_key",
                          "_t", "nTimeHits", "eem", "ehad",
                          "edges[0]", "edges[1]", "edges[2]", "edges[3]",
                          "px", "py", "pz", "e"])
    # need to use px, py, pz, e as angle and pt will break for massive particles along the beamline
    repr_format = '!'.join(["MyTower", repr_body, "particles"])

    @classmethod
    def from_repr(cls, rep):
        if rep == '': return None
        name, body, particles = rep.split('!')
        assert name == cls.__name__
        if particles != '':
            particles = [MCParticle.from_repr(p) for p in particles.split('&')]
        else:
            particles = []
        body_parts = body.split('&')
        global_tower_id = int(body_parts[0])
        # need to use px, py, pz, e as angle and pt will break for massive particles along the beamline
        class_dict = {"sql_key": safe_convert(int, body_parts[1]),
                      "t": safe_convert(float, body_parts[2]),
                      "nTimesHits": safe_convert(int, body_parts[3]),
                      "eem": safe_convert(bool, body_parts[4]),
                      "ehad": safe_convert(bool, body_parts[5]),
                      "edges": [safe_convert(float, body_parts[6]),
                                safe_convert(float, body_parts[7]),
                                safe_convert(float, body_parts[8]),
                                safe_convert(float, body_parts[9]),],
                      "particles": particles,
                      "px": safe_convert(float, body_parts[10]),  # these kwargs wont be used
                      "py": safe_convert(float, body_parts[11]),  # but they should be safely ignored
                      "pz": safe_convert(float, body_parts[12]),
                      "e": safe_convert(float, body_parts[13])}
        new_tower = cls(class_dict['px'], class_dict['py'], class_dict['pz'], 
                        class_dict['e'], global_tower_id=global_tower_id, **class_dict)
        return new_tower


class MyTrack(SafeLorentz):
    def __init__(self, *args, global_track_id, **kwargs):
        # IDs
        self.global_track_id = global_track_id
        self.pid = kwargs.get('pid', None)
        self.sql_key = kwargs.get('sql_key', None)
        self.particle_sql_key = kwargs.get('particle_sql_key', None)
        self.particle = kwargs.get('particle', None)
        if self.particle is not None:
            self.global_id = self.particle.global_id
        else:
            self.global_id = None
        # kinematics
        self.charge = kwargs.get('charge', None)
        # tracks are always constructed base on outers, so store the momentum info seperatly
        self._p = kwargs.get('p', None)
        self._pT = kwargs.get('pT', None)
        self._eta = kwargs.get('eta', None)
        self._phi = kwargs.get('phi', None)
        self.ctgTheta = kwargs.get('ctgTheta', None)
        self.etaOuter = kwargs.get('etaOuter', None)
        self.phiOuter = kwargs.get('phiOuter', None)
        self._t = kwargs.get('t', None)
        self._x = kwargs.get('x', None)
        self._y = kwargs.get('y', None)
        self._z = kwargs.get('z', None)
        self.xd = kwargs.get('xd', None)
        self.yd = kwargs.get('yd', None)
        self.zd = kwargs.get('zd', None)
        self.l = kwargs.get('l', None)
        self.d0 = kwargs.get('d0', None)
        self.dZ = kwargs.get('dZ', None)
        # don't bother with the errors for now
        if len(args) == 4:
            super().__init__(*args)
        elif 't_outer' in kwargs:
            t_outer = kwargs['t_outer']
            x_outer = kwargs['x_outer']
            y_outer = kwargs['y_outer']
            z_outer = kwargs['z_outer']
            super().__init__(x_outer, y_outer, z_outer, t_outer)

    def __str__(self):
        return f"MyTrack[{self.global_track_id}] energy;{self.e:.2e}"

    def __repr__(self):
        body_content = [self.global_track_id,
                        self.pid,
                        self.sql_key,
                        self.particle_sql_key,
                        self.particle,
                        self.charge,
                        self._p,
                        self._pT,
                        self._eta,
                        self._phi,
                        self.ctgTheta,
                        self.etaOuter,
                        self.phiOuter,
                        self._t,
                        self._x,
                        self._y,
                        self._z,
                        self.xd,
                        self.yd,
                        self.zd,
                        self.l,
                        self.d0,
                        self.dZ,
                        self.t,
                        self.x,
                        self.y,
                        self.z]
        # | and , are used inside MyPaticle string rep
        # so diferent deliminators are needed here
        body = '&'.join([repr(x) for x in body_content])
        rep = "!".join([MyTrack.__name__, body])
        return rep

    repr_body = '&'.join(["global_track_id",
                          "pid", "sql_key", "particle_sql_key",
                          MCParticle.repr_format, "charge",
                          "_p", "_pT", "_eta", "_phi",
                          "ctgTheta", "etaOuter", "phiOuter",
                          "_t", "_x", "_y", "_z",
                          "xd", "yd", "zd", "l",
                          "d0", "dZ",
                          "t", "x", "y", "z"])
    repr_format = '!'.join(["MyTrack", repr_body])
    @classmethod
    def from_repr(cls, rep):
        if rep == '' or rep == 'None': return None
        name, body = rep.split('!')
        assert name == cls.__name__
        body_parts = body.split('&')
        global_track_id = int(body_parts[0])
        class_dict = {'pid': safe_convert(int, body_parts[1]),
                      'sql_key': safe_convert(int, body_parts[2]),
                      'particle_sql_key': safe_convert(int, body_parts[3]),
                      'particle': MCParticle.from_repr(body_parts[4]),
                      'charge': safe_convert(float, body_parts[5]),
                      'p': safe_convert(float, body_parts[6]),
                      'pT': safe_convert(float, body_parts[7]),
                      'eta': safe_convert(float, body_parts[8]),
                      'phi': safe_convert(float, body_parts[9]),
                      'ctgTheta': safe_convert(float, body_parts[10]),
                      'etaOuter': safe_convert(float, body_parts[11]),
                      'phiOuter': safe_convert(float, body_parts[12]),
                      't': safe_convert(float, body_parts[13]),
                      'x': safe_convert(float, body_parts[14]),
                      'y': safe_convert(float, body_parts[15]),
                      'z': safe_convert(float, body_parts[16]),
                      'xd': safe_convert(float, body_parts[17]),
                      'yd': safe_convert(float, body_parts[18]),
                      'zd': safe_convert(float, body_parts[19]),
                      'l': safe_convert(float, body_parts[20]),
                      'd0': safe_convert(float, body_parts[21]),
                      'dZ': safe_convert(float, body_parts[22]),
                      't_outer': safe_convert(float, body_parts[23]),
                      'x_outer': safe_convert(float, body_parts[24]),
                      'y_outer': safe_convert(float, body_parts[25]),
                      'z_outer': safe_convert(float, body_parts[26])}
        new_track = cls(global_track_id=global_track_id, **class_dict)
        return new_track


class ParticleCollection:
    """Holds a group of particles together"""
    particle_type = MyParticle
    indexed_variables = MyParticle.repr_variables
    variable_types = MyParticle.repr_variable_types
    # prevent messing with the particles directly
    __is_frozen = False
    def __setattr__(self, key, value):
        if '__is_frozen' in key or not self.__is_frozen:
            super().__setattr__(key, value)
        else:
            raise TypeError("Do not set the attributes directly, " +
                            "add particles with addParticles()")

    def _freeze(self):
        if not self.__is_frozen:
            self.__is_frozen = True

    def _unfreeze(self):
        self.__is_frozen = False

    def __init__(self, *args, **kwargs):
        self.name = kwargs.get("name", "ParticleCollection")
        self._ptetaphie = np.empty((0, 4), dtype=float)
        self.columns = ["$p_T$", "$\\eta$", "$\\phi$", "$E$"]
        self.pts = np.array([], dtype=float)
        self.etas = np.array([], dtype=float)
        self.phis = np.array([], dtype=float)
        for var_type, var_name in zip(self.variable_types, self.indexed_variables):
            var_names = var_name + 's'
            setattr(self, var_names, np.array([], dtype=var_type))
        self.particle_list = []
        self.addParticles(*args)
        self._freeze()
    
    def updateParticles(self):
        self._unfreeze()
        self._ptetaphie = np.array([[p.pt, p.eta, p.phi(), p.e]
                              for p in self.particle_list])
        self.pts = np.array([p.pt for p in self.particle_list])
        self.etas = np.array([p.eta for p in self.particle_list])
        self.phis = np.array([p.phi() for p in self.particle_list])
        for var_type, var_name in zip(self.variable_types, self.indexed_variables):
            none_value = 0 if var_type is int else (np.nan if var_type is float else None)
            var_names = var_name + 's'
            var_val = [getattr(p, var_name, none_value) for p in self.particle_list]
            var_val = [none_value if v is None else v for v in var_val]
            var_val = np.array(var_val, dtype=var_type)
            setattr(self, var_names, var_val)
        self._freeze()

    def addParticles(self, *args):
        if len(args) == 0: return  # no particles to add
        self._unfreeze()
        new_particles = args
        if len(new_particles) > 0:
            self.particle_list += new_particles
            self._ptetaphie = np.vstack((self._ptetaphie,
                       np.array([[p.pt, p.eta, p.phi(), p.e]
                                  for p in new_particles])))
            self.pts = np.hstack((self.pts,
                       np.array([p.pt for p in new_particles])))
            self.etas = np.hstack((self.etas,
                       np.array([p.eta for p in new_particles])))
            self.phis = np.hstack((self.phis,
                       np.array([p.phi() for p in new_particles])))
        for var_type, var_name in zip(self.variable_types, self.indexed_variables):
            none_value = 0 if var_type is int else (np.nan if var_type is float else None)
            var_names = var_name + 's'
            old_vals = getattr(self, var_names)
            new_vals = [getattr(p, var_name, none_value) for p in new_particles]
            new_vals = [none_value if v is None else v for v in new_vals]
            new_vals = np.array(new_vals, dtype=var_type)
            setattr(self, var_names, np.hstack((old_vals, new_vals)))
        self._freeze()

    def __getitem__(self, idx):
        return self._ptetaphie[idx]

    def __len__(self):
        return len(self._ptetaphie)

    @property
    def shape(self):
        return self._ptetaphie.shape

    @property
    def size(self):
        return self._ptetaphie.shape

    def __str__(self):
        return f"<{self.name}, {len(self)} particles>"

    def __repr__(self):
        lines = ['# ' + self.name]
        for particle in self.particle_list:
            lines.append(repr(particle))
        return '\n'.join(lines)

    def collective_pt(self):
        return np.sum(self.pts)

    def collective_eta(self):
        raise NotImplementedError #TODO - also any other collective properties

    def write(self, file_name):
        with open(file_name, 'w') as save_file:
            save_file.write(self.__repr__())

    def write_summary(self, dir_name):
        summary_file_name = os.path.join(dir_name, f"summary_observables.csv")
        # remove a summary file if it exists
        with contextlib.suppress(FileNotFoundError):
            os.remove(summary_file_name)
        # write the summaries
        summary_cols = [self.particle_type.id_name, "pt",
                        "eta", "rap", "phi", "e", 
                        "px", "py", "pz"]
        summary = np.hstack((getattr(self, self.particle_type.id_name+'s').reshape((-1, 1)),
                             self.pts.reshape((-1, 1)),
                             self.etas.reshape((-1, 1)),
                             np.array([p.rapidity() for p in self.particle_list]).reshape((-1, 1)),
                             self.phis.reshape((-1, 1)),
                             np.array([p.e for p in self.particle_list]).reshape((-1, 1)),
                             np.array([p.px for p in self.particle_list]).reshape((-1, 1)),
                             np.array([p.py for p in self.particle_list]).reshape((-1, 1)),
                             np.array([p.pz for p in self.particle_list]).reshape((-1, 1))))
        if len(summary) > 0:
            np.savetxt(summary_file_name, summary,
                       header=' '.join(summary_cols))
        else:
            with open(summary_file_name, 'w') as summary_file:
                summary_file.write('# ' + ' '.join(summary_cols) + '\n')

    @classmethod
    def from_repr(cls, repr_str):
        lines = repr_str.split('\n')
        name = lines[0][2:]
        particle_list = [cls.particle_type.from_repr(line)
                         for line in lines[1:]]
        return cls(*particle_list, name=name)

    @classmethod
    def from_file(cls, file_name):
        with open(file_name, 'r') as save_file:
            repr_str = save_file.read()
        return cls.from_repr(repr_str)


class MCParticleCollection(ParticleCollection):
    """Holds a group of particles together
    and facilitates quick indexing of pt, eta, phi, e
    A flat list, more complex forms are acheved by indexing the collection externally"""
    particle_type = MCParticle
    indexed_variables = MCParticle.repr_variables + ['m']
    variable_types = MCParticle.repr_variable_types + [float]


class RecoParticleCollection(ParticleCollection):
    """Holds a group of particles together
    and facilitates quick indexing of pt, eta, phi, e
    A flat list, more complex forms are acheved by indexing the collection externally"""
    particle_type = RecoParticle
    indexed_variables = RecoParticle.repr_variables + ['m']
    variable_types = RecoParticle.repr_variable_types + [float]


class MultiParticleCollections:
    def __init__(self, particle_lists):
        # figure out what kind of particles
        first_object = next(pl for pl in particle_lists if len(pl))
        if issubclass(type(first_object), ParticleCollection):
            self.collections_list = particle_lists
        elif isinstance(first_object[0], MCParticle):
            self.collections_list = [MCParticleCollection(*p_list)
                                     for p_list in particle_lists]
        elif isinstance(first_object[0], RecoParticle):
            self.collections_list = [RecoParticleCollection(*p_list)
                                     for p_list in particle_lists]
        else:
            self.collections_list = [ParticleCollection(*p_list)
                                     for p_list in particle_lists]
    
    def __getitem__(self, idx):
        if isinstance(idx, (int, slice)):
            return self.collections_list[idx]
        return self.collections_list[idx[0]][idx[1:]]

    def __repr__(self):
        components = [repr(p_collection) for p_collection in self.collections_list]
        cumulative_lines = []
        size = 0
        for block in components:
            size += block.count('\n') + 1
            cumulative_lines.append(str(size))
        components = ['# ' + ' '.join(cumulative_lines)] + components
        return '\n'.join(components)

    def write(self, file_name):
        with open(file_name, 'w') as save_file:
            save_file.write(self.__repr__())

    @classmethod
    def from_repr(cls, repr_str, batch_start=None, batch_end=None):
        lines = repr_str.split('\n')
        # the type can be determined by the name at the start of the line
        particle_type = lines[-1].split("Particle", 1)[0]
        cumulative_lines = [int(x) for x in lines[0].split(' ')[1:]]
        by_collection = np.split(np.array(lines[1:]), cumulative_lines)
        avalible = len(by_collection)
        if batch_start is None:
            batch_start = 0
        elif batch_start > avalible:
            return  # then return nothing
        if batch_end is None:
            batch_end = avalible
        elif batch_end > avalible:
            batch_end = avalible
        by_collection = by_collection[batch_start:batch_end]
        if particle_type == 'My':
            collections = [ParticleCollection.from_repr('\n'.join(coll))
                           for coll in by_collection]
        elif particle_type == 'MC':
            collections = [MCParticleCollection.from_repr('\n'.join(coll))
                           for coll in by_collection]
        elif particle_type == 'Reco':
            collections = [RecoParticleCollection.from_repr('\n'.join(coll))
                           for coll in by_collection]
        else:
            raise ValueError(f"Dont recognise particle_type {particle_type}")
        particle_lists = [col.particle_list for col in collections]
        return cls(particle_lists)

    @classmethod
    def from_file(cls, file_name, batch_start=None, batch_end=None):
        with open(file_name, 'r') as save_file:
            repr_str = save_file.read()
        return cls.from_repr(repr_str, batch_start, batch_end)


class Observables:
    def __init__(self, particle_collection=None, tracks=None, towers=None, event_num=None):
        self.event_num = event_num
        self.has_tracksTowers = tracks is not None or towers is not None
        global_obs_ids = []
        self.global_to_obs = {}
        self.obs_to_global = {}
        if self.has_tracksTowers:
            if tracks is None:
                tracks = []
            if towers is None:
                towers = []
            self.tracks_list = tracks
            self.towers_list = towers
            self.objects = tracks + towers
            self.pts = np.array([t.pt for t in tracks] +
                                [t.et for t in towers])
            # fill maps
            obs_id = 0
            for track in tracks:
                global_obs_ids.append(obs_id)
                self.obs_to_global[obs_id] = track.global_id
                self.global_to_obs[track.global_id] = obs_id
                obs_id += 1
            for tower in towers:
                global_obs_ids.append(obs_id)
                self.obs_to_global[obs_id] = tower.global_ids
                for global_id in tower.global_ids:
                    self.global_to_obs[global_id] = obs_id
                obs_id += 1
        elif particle_collection is not None:
            if isinstance(particle_collection, MCParticleCollection):
                self.objects = [p for p in particle_collection.particle_list if p.is_leaf]
            else:
                self.objects = [p for p in particle_collection.particle_list]
            self.pts = np.array([p.pt for p in self.objects])
            # fill maps
            obs_id = 0
            for particle in self.objects:
                global_obs_ids.append(obs_id)
                self.obs_to_global[obs_id] = particle.global_id
                self.global_to_obs[particle.global_id] = obs_id
                obs_id += 1
        else:
            raise ValueError("Need to provide particles, tracks or towers.")
        self.global_obs_ids = np.array(global_obs_ids)
        # tell the object their obs ids
        for oid, obj in zip(self.global_obs_ids, self.objects):
            obj.global_obs_id = oid
        self.etas = np.array([t.eta for t in self.objects])
        self.raps = np.array([t.rapidity() for t in self.objects])
        self.phis = np.array([t.phi() for t in self.objects])
        self.es = np.array([t.e for t in self.objects])
        self.pxs = np.array([t.px for t in self.objects])
        self.pys = np.array([t.py for t in self.objects])
        self.pzs = np.array([t.pz for t in self.objects])
        self.jet_allocation = np.full_like(self.global_obs_ids, np.nan)

    def __len__(self):
        return len(self.objects)

    def write_summary(self, dir_name, event_str):
        summary_file_name = os.path.join(dir_name, f"summary_observables{event_str}.csv")
        # remove a summary file if it exists
        with contextlib.suppress(FileNotFoundError):
            os.remove(summary_file_name)
        # write the summaries
        summary_cols = ["global_obs_id", "pt",
                        "eta", "rap", "phi", "e", 
                        "px", "py", "pz",
                        "jet_allocation"]
        summary = np.hstack((self.global_obs_ids.reshape((-1, 1)),
                            self.pts.reshape((-1, 1)),
                            self.etas.reshape((-1, 1)),
                            self.raps.reshape((-1, 1)),
                            self.phis.reshape((-1, 1)),
                            self.es.reshape((-1, 1)),
                            self.pxs.reshape((-1, 1)),
                            self.pys.reshape((-1, 1)),
                            self.pzs.reshape((-1, 1)),
                            self.jet_allocation.reshape((-1, 1))))
        if len(summary) > 0:
            np.savetxt(summary_file_name, summary,
                       header=' '.join(summary_cols))
        else:
            with open(summary_file_name, 'w') as summary_file:
                summary_file.write('# ' + ' '.join(summary_cols) + '\n')

    def write(self, dir_name):
        # make the directory if it dosn't exist
        os.makedirs(dir_name, exist_ok=True)
        if self.event_num is None:
            event_str = ''
        else:
            event_str = str(self.event_num)
        file_name = os.path.join(dir_name, f"observables{event_str}.dat")
        # if there are already obesrvables in that file
        # move them to a backup
        if os.path.exists(file_name):
            backup_format = file_name + ".bk{}"
            backup_number = 1
            while os.path.exists(backup_format.format(backup_number)):
                backup_number += 1
            print(f"Moving previous observables file to backup number {backup_number}")
            if backup_number > 3:
                print("Maybe you should tidy up....")
            os.rename(file_name, backup_format.format(backup_number))
        # write the objects
        with open(file_name, 'w') as obj_file:
            # some headers
            obj_file.writelines([MCParticle.repr_format + '\n',
                                 MyTrack.repr_format + '\n',
                                 MyTower.repr_format + '\n'])
            # content
            obj_file.writelines([repr(obj) + '\n' for obj in self.objects])
        self.write_summary(dir_name, event_str)

    @classmethod
    def from_file(cls, dir_name, event_num=None):
        if event_num is None:
            event_str = ''
        else:
            event_str = str(event_num)
        if event_num is None:
            file_name = os.path.join(dir_name, f"observables.dat")
        else:
            file_name = os.path.join(dir_name, f"observables{event_num}.dat")
        particles = []
        tracks = []
        towers = []
        with open(file_name, 'r') as obj_file:
            # skip the 3 header lines
            for _ in range(3):
                next(obj_file)
            for line in obj_file:
                line = line[:-1]  # kill the newline
                if line.startswith("MCParticle"):
                    particles.append(MCParticle.from_repr(line))
                elif line.startswith("MyTrack"):
                    tracks.append(MyTrack.from_repr(line))
                elif line.startswith("MyTower"):
                    towers.append(MyTower.from_repr(line))
        if len(tracks) + len(towers) > 0:
            return cls(tracks=tracks, towers=towers, event_num=event_num)
        else:
            particles = MCParticleCollection(*particles)
            return cls(particle_collection=particles, event_num=event_num)


def make_observables(pts, etas, phis, es, dir_name="./tmp"):
    num_obs = len(pts)
    particle_list = []
    for i in range(num_obs):
        particle = MCParticle(global_id=i, pt=pts[i], eta=etas[i], phi=phis[i], e=es[i],
                              is_leaf=True)
        particle_list.append(particle)
    collection = MCParticleCollection(*particle_list)
    observables = Observables(collection)
    observables.write(dir_name)
    return observables
   
# probably wont use
class CollectionStructure:
    def __init__(self, collection, structure=None, shape=None):
        self.collection = collection
        if structure is not None:
            self.structure = structure
            if shape is not None:
                assert structure.shape == shape,\
                        (f"Error structure has shape {structure.shape} " +
                         f"but shape argument is {shape}.")
        else:
            if shape is None:
                shape = collection.shape
            self.structure = np.arange(np.product(shape)).reshape(shape)

    def __getitem__(self, idx):
        idx = np.array(idx)
        return self.collection[idx]

