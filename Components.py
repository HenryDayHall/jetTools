"""Low level components, format apes that of root """
from ipdb import set_trace as st
import numpy as np
from skhep import math as hepmath
import pyjet

class MyParticle(hepmath.vectors.LorentzVector):
    """Aping genparticle."""
    def __init__(self, *args, **kwargs):
        # IDs
        self.pid = kwargs.get('pid', None)
        self.sql_key = kwargs.get('sql_key', None)
        self.hepmc_barcode = kwargs.get('hepmc_barcode', None)
        self.global_id = kwargs.get('global_id', None)
        # shower position
        self.is_root = kwargs.get('is_root', False)
        self.is_leaf = kwargs.get('is_leaf', False)
        self.start_vertex_barcode = kwargs.get('start_vertex_barcode', None)
        self.end_vertex_barcode = kwargs.get('end_vertex_barcode', None)
        self.parent_ids = []
        self.child_ids = []
        # status
        self.status = kwargs.get('status', None)
        self.generated_mass = kwargs.get('generated_mass', None)
        # knimatics
        if len(args) == 4:
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


class MyVertex(hepmath.vectors.LorentzVector):
    """Aping GenVertex."""
    def __init__(self, *args, **kwargs):
        # IDs
        self.sql_key = kwargs.get('sql_key', None)
        self.hepmc_barcode = kwargs.get('hepmc_barcode', None)
        self.global_id = kwargs.get('global_id', None)
        # particle interactions
        self.n_out = kwargs.get('n_out', None)
        self.n_weights = kwargs.get('n_weights', None)
        self.n_orphans = kwargs.get('n_orphans', None)
        # location
        if len(args) == 4:
            super().__init__(*args)
        elif 'x' in kwargs:
            self.x = kwargs.get('x', None)
            self.y = kwargs.get('y', None)
            self.z = kwargs.get('z', None)
            self.ctau = kwargs.get('ctau', None)
            super(kwargs['x'], kwargs['y'], kwargs['z'], kwargs['ctau'])


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
        self.pts = np.empty((0, 1), dtype=float)
        self.etas = np.empty((0, 1), dtype=float)
        self.phis = np.empty((0, 1), dtype=float)
        self.es = np.empty((0, 1), dtype=float)
        self.pxs = np.empty((0, 1), dtype=float)
        self.pys = np.empty((0, 1), dtype=float)
        self.pzs = np.empty((0, 1), dtype=float)
        self.ms = np.empty((0, 1), dtype=float)
        self.pids = np.empty((0, 1), dtype=float)
        self.sql_keys = np.empty((0, 1), dtype=float)
        self.hepmc_barcode = np.empty((0, 1), dtype=float)
        self.global_ids = np.empty((0, 1), dtype=float)
        self.is_roots = np.empty((0, 1), dtype=float)
        self.is_leafs = np.empty((0, 1), dtype=float)
        self.start_vertex_barcodes = np.empty((0, 1), dtype=float)
        self.end_vertex_barcodes = np.empty((0, 1), dtype=float)
        self.particle_list = []
        self.addParticles(args)
        self._freeze()
    
    def addParticles(self, *args):
        self._unfreeze()
        self.particle_list += args[0][0]
        self._ptetaphie = np.vstack((self._ptetaphie,
                   np.array([[p.pt, p.eta, p.phi, p.e]
                              for p in self.particle_list])))
        self.pts = np.vstack((self.pts,
                   np.array([p.pt for p in self.particle_list]).reshape((-1, 1))))
        self.etas = np.vstack((self.etas,
                   np.array([p.eta for p in self.particle_list]).reshape((-1, 1))))
        self.phis = np.vstack((self.phis,
                   np.array([p.phi for p in self.particle_list]).reshape((-1, 1))))
        self.es = np.vstack((self.es,
                   np.array([p.e for p in self.particle_list]).reshape((-1, 1))))
        self.pxs = np.vstack((self.pxs,
                   np.array([p.px for p in self.particle_list]).reshape((-1, 1))))
        self.pys = np.vstack((self.pys,
                   np.array([p.py for p in self.particle_list]).reshape((-1, 1))))
        self.pzs = np.vstack((self.pzs,
                   np.array([p.pz for p in self.particle_list]).reshape((-1, 1))))
        self.ms = np.vstack((self.ms,
                   np.array([p.m for p in self.particle_list]).reshape((-1, 1))))
        self.pids = np.vstack((self.pids,
                   np.array([p.pid for p in self.particle_list]).reshape((-1, 1))))
        self.sql_keys = np.vstack((self.sql_keys,
                   np.array([p.sql_key for p in self.particle_list]).reshape((-1, 1))))
        self.hepmc_barcode = np.vstack((self.hepmc_barcode,
                   np.array([p.hepmc_barcode for p in self.particle_list]).reshape((-1, 1))))
        self.global_ids = np.vstack((self.global_ids,
                   np.array([p.global_id for p in self.particle_list]).reshape((-1, 1))))
        self.is_roots = np.vstack((self.is_roots,
                   np.array([p.is_root for p in self.particle_list]).reshape((-1, 1))))
        self.is_leafs = np.vstack((self.is_leafs,
                   np.array([p.is_leaf for p in self.particle_list]).reshape((-1, 1))))
        self.start_vertex_barcodes = np.vstack((self.start_vertex_barcodes,
                   np.array([p.start_vertex_barcode for p in self.particle_list]).reshape((-1, 1))))
        self.end_vertex_barcodes = np.vstack((self.end_vertex_barcodes,
                   np.array([p.end_vertex_barcode for p in self.particle_list]).reshape((-1, 1))))
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

