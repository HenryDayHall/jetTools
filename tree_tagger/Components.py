"""Low level components, format apes that of root """
import contextlib
import os
from ipdb import set_trace as st
import numpy as np
from skhep import math as hepmath
from tree_tagger import Constants


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

    def __str__(self):
        return f"MyParticle[{self.global_id}] pid;{self.pid} energy;{self.e:.2e}"

    def __repr__(self):
        body_content = [self.global_id,
                        self.pid,
                        self.sql_key,
                        self.hepmc_barcode,
                        self.is_root,
                        self.is_leaf,
                        self.start_vertex_barcode,
                        self.end_vertex_barcode,
                        self.status,
                        self.generated_mass,
                        self.px,
                        self.py,
                        self.pz,
                        self.e]
        body = ','.join([repr(x) for x in body_content])
        mothers = ','.join([repr(m) for m in self.mother_ids])
        daughters = ','.join([repr(m) for m in self.daughter_ids])
        rep = "|".join([MyParticle.__name__, body, mothers, daughters])
        return rep
    repr_body = ','.join(["global_id", "pid", "sql_key", "hepmc_barcode",
                          "is_root", "is_leaf",
                          "start_vertex_barcode", "end_vertex_barcode",
                          "status", "generated_mass",
                          "px", "py", "pz", "e"])
    repr_format = '|'.join(["MyParticle", repr_body,
                            'mothers', 'daughters'])


    @classmethod
    def from_repr(cls, rep):
        if rep == '' or rep == 'None': return None
        name, body, mothers, daughters = rep.split('|')
        assert name == cls.__name__
        body_parts = body.split(',')
        global_id = int(body_parts[0])
        class_dict = {"pid": safe_convert(int, body_parts[1]),
                      "sql_key": safe_convert(int, body_parts[2]),
                      "hepmc_barcode": safe_convert(int, body_parts[3]),
                      "is_root": safe_convert(bool, body_parts[4]),
                      "is_leaf": safe_convert(bool, body_parts[5]),
                      "start_vertex_barcode": safe_convert(int, body_parts[6]),
                      "end_vertex_barcode": safe_convert(int, body_parts[7]),
                      "status": safe_convert(int, body_parts[8]),
                      "generated_mass": safe_convert(float, body_parts[9]),
                      "px": safe_convert(float, body_parts[10]),
                      "py": safe_convert(float, body_parts[11]),
                      "pz": safe_convert(float, body_parts[12]),
                      "e": safe_convert(float, body_parts[13])}
        new_particle = cls(global_id=global_id, **class_dict)
        if mothers != '':
            new_particle.mother_ids = [int(m) for m in mothers.split(',')]
        if daughters != '':
            new_particle.daughter_ids = [int(d) for d in daughters.split(',')]
        return new_particle


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
            particles = [MyParticle.from_repr(p) for p in particles.split('&')]
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
                          MyParticle.repr_format, "charge",
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
                      'particle': MyParticle.from_repr(body_parts[4]),
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
    """Holds a group of particles together
    and facilitates quick indexing of pt, eta, phi, e
    A flat list, more complex forms are acheved by indexing the collection externally"""
    # prevent messing with the particles directly
    __is_frozen = False
    def __setattr__(self, key, value):
        if key == '_ParticleCollection__is_frozen' or not self.__is_frozen:
            super().__setattr__(key, value)
        else:
            raise TypeError("Do not set the attributes directly, " +
                            "add particles with addParticles()")

    def _freeze(self):
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
        self.es = np.array([], dtype=float)
        self.pxs = np.array([], dtype=float)
        self.pys = np.array([], dtype=float)
        self.pzs = np.array([], dtype=float)
        self.ms = np.array([], dtype=float)
        self.pids = np.array([], dtype=int)
        self.sql_keys = np.array([], dtype=int)
        self.hepmc_barcodes = np.array([], dtype=int)
        self.global_ids = np.array([], dtype=int)
        self.is_roots = np.array([], dtype=bool)
        self.is_leafs = np.array([], dtype=bool)
        self.start_vertex_barcodes = np.array([], dtype=int)
        self.end_vertex_barcodes = np.array([], dtype=int)
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
        self.es = np.array([p.e for p in self.particle_list])
        self.pxs = np.array([p.px for p in self.particle_list])
        self.pys = np.array([p.py for p in self.particle_list])
        self.pzs = np.array([p.pz for p in self.particle_list])
        self.ms = np.array([p.m for p in self.particle_list])
        self.pids = np.array([p.pid for p in self.particle_list])
        self.sql_keys = np.array([p.sql_key for p in self.particle_list])
        self.hepmc_barcodes = np.array([p.hepmc_barcode for p in self.particle_list])
        self.global_ids = np.array([p.global_id for p in self.particle_list])
        self.is_roots = np.array([p.is_root for p in self.particle_list])
        self.is_leafs = np.array([p.is_leaf for p in self.particle_list])
        self.start_vertex_barcodes = np.array([p.start_vertex_barcode for p in self.particle_list])
        self.end_vertex_barcodes = np.array([p.end_vertex_barcode for p in self.particle_list])
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
            self.es = np.hstack((self.es,
                       np.array([p.e for p in new_particles])))
            self.pxs = np.hstack((self.pxs,
                       np.array([p.px for p in new_particles])))
            self.pys = np.hstack((self.pys,
                       np.array([p.py for p in new_particles])))
            self.pzs = np.hstack((self.pzs,
                       np.array([p.pz for p in new_particles])))
            self.ms = np.hstack((self.ms,
                       np.array([p.m for p in new_particles])))
            self.pids = np.hstack((self.pids,
                       np.array([p.pid for p in new_particles])))
            self.sql_keys = np.hstack((self.sql_keys,
                       np.array([p.sql_key for p in new_particles])))
            self.hepmc_barcodes = np.hstack((self.hepmc_barcodes,
                       np.array([p.hepmc_barcode for p in new_particles])))
            self.global_ids = np.hstack((self.global_ids,
                       np.array([p.global_id for p in new_particles])))
            self.is_roots = np.hstack((self.is_roots,
                       np.array([p.is_root for p in new_particles])))
            self.is_leafs = np.hstack((self.is_leafs,
                       np.array([p.is_leaf for p in new_particles])))
            self.start_vertex_barcodes = np.hstack((self.start_vertex_barcodes,
                       np.array([p.start_vertex_barcode for p in new_particles])))
            self.end_vertex_barcodes = np.hstack((self.end_vertex_barcodes,
                       np.array([p.end_vertex_barcode for p in new_particles])))
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
        return f"<{self.name}, {self.global_ids}>"

    def collective_pt(self):
        return np.sum(self.pts)

    def collective_eta(self):
        raise NotImplementedError #TODO - also any other collective properties


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
            self.objects = [p for p in particle_collection.particle_list if p.is_leaf]
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

    def write(self, dir_name):
        # make the directory if it dosn't exist
        os.makedirs(dir_name, exist_ok=True)
        if self.event_num is None:
            event_str = ''
        else:
            event_str = str(event_num)
        file_name = os.path.join(dir_name, f"observables{event_str}.dat")
        summary_file_name = os.path.join(dir_name, "summary_observables{event_str}.csv")
        # remove a summary file if it exists
        with contextlib.suppress(FileNotFoundError):
            os.remove(summary_file_name)
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
            obj_file.writelines([MyParticle.repr_format + '\n',
                                 MyTrack.repr_format + '\n',
                                 MyTower.repr_format + '\n'])
            # content
            obj_file.writelines([repr(obj) + '\n' for obj in self.objects])
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

    @classmethod
    def from_file(cls, dir_name, event_num=None):
        if event_num is None:
            event_str = ''
        else:
            event_str = str(event_num)
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
                if line.startswith("MyParticle"):
                    particles.append(MyParticle.from_repr(line))
                elif line.startswith("MyTrack"):
                    tracks.append(MyTrack.from_repr(line))
                elif line.startswith("MyTower"):
                    towers.append(MyTower.from_repr(line))
        if len(tracks) + len(towers) > 0:
            return cls(tracks=tracks, towers=towers, event_num=event_num)
        else:
            particles = ParticleCollection(*particles)
            return cls(particle_collection=particles, event_num=event_num)


def make_observables(pts, etas, phis, es, dir_name="./tmp"):
    num_obs = len(pts)
    particle_list = []
    for i in range(num_obs):
        particle = MyParticle(global_id=i, pt=pts[i], eta=etas[i], phi=phis[i], e=es[i],
                              is_leaf=True)
        particle_list.append(particle)
    collection = ParticleCollection(*particle_list)
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
