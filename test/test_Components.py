import numpy as np
from ipdb import set_trace as st
import collections
from numpy import testing as tst
from tree_tagger import Components
from tools import generic_equality_comp, TempTestDir

def test_safe_convert():
    # edge case tests
    examples = (('None', str, None),
                ('None', int, None),
                ('None', float, None),
                ('None', bool, None),
                ('1', str, '1'),
                ('1', int, 1),
                ('1', float, 1.),
                ('1', bool, True),
                ('0', str, '0'),
                ('0', int, 0),
                ('0', float, 0.),
                ('0', bool, False),
                ('-1', str, '-1'),
                ('-1', int, -1),
                ('-1', float, -1.),
                ('-1', bool, True),
                ('0.5', str, '0.5'),
                ('0.5', float, 0.5),
                ('0.5', bool, True))
    for inp, cls, exptd in examples:
        out = Components.safe_convert(cls, inp)
        assert out == exptd, "Components.safe_convert failed to convert " +\
                             f"{inp} to {exptd} via {cls} " +\
                             f"instead got {out}."
    # try some random numbers
    for num in np.random.uniform(-1000, 1000, 20):
        inp = str(num)
        cls = float
        exptd = num
        out = Components.safe_convert(cls, inp)
        assert out == exptd, "Components.safe_convert failed to convert " +\
                             f"{inp} to {exptd} via {cls} " +\
                             f"instead got {out}."
        cls = bool
        exptd = True
        out = Components.safe_convert(cls, inp)
        assert out == exptd, "Components.safe_convert failed to convert " +\
                             f"{inp} to {exptd} via {cls} " +\
                             f"instead got {out}."
        cls = int
        exptd = int(num)
        out = Components.safe_convert(cls, inp.split('.', 1)[0])
        assert out == exptd, "Components.safe_convert failed to convert " +\
                             f"{inp} to {exptd} via {cls} " +\
                             f"instead got {out}."


def test_MCParticle():
    # some variations on the general inputs
    default_gen_inps = {'global_id' : 1,
                        'pid' : None,
                        'sql_key' : None,
                        'hepmc_barcode' : None,
                        'is_root' : False,
                        'is_leaf' : False,
                        'start_vertex_barcode' : None,
                        'end_vertex_barcode' : None,
                        'status' : None,
                        'generated_mass' : None}
    alternate_gen_inps = {'global_id' : 100,
                          'pid' : 1,
                          'sql_key' : 2,
                          'hepmc_barcode' : 3,
                          'is_root' : True,
                          'is_leaf' : True,
                          'start_vertex_barcode' : 4,
                          'end_vertex_barcode' : 5,
                          'status' : 0,
                          'generated_mass' : 3.4}
    # dummy kinematic variables
    kinematic_xyzs = ((0., 0., 0., 0.),
                      (0., 0., 0., 100.),
                      (-1000., 1000., 0.002, 100.),
                      (-1000., 1000., 0.002, 0.),
                      (-1000, 1000, 0, 0),
                      (-252, -6, -2.34, -2.6))
    kinematic_angle = ((0., 0., 0., 0.),
                       (0., 0., 0., 100.),
                       (1., -1., 0.002, 100.),
                       (0.002, -2*np.pi, 2*np.pi, 0.),
                       (0, -2*np.pi, 2*np.pi, 0.),
                       (2.52, -.6, -2.34, -2.6))
    for kset in kinematic_xyzs:
        # test the px py pz version
        particle1 = Components.MCParticle(*kset, global_id=default_gen_inps['global_id'])
        assert particle1.global_id == 1
        assert particle1.pid == None
        assert particle1.sql_key == None
        assert particle1.hepmc_barcode == None
        assert particle1.is_root == False
        assert particle1.is_leaf == False
        assert particle1.start_vertex_barcode == None
        assert particle1.end_vertex_barcode == None
        assert particle1.status == None
        assert particle1.generated_mass == None
        assert particle1.daughter_ids == []
        particle2 = Components.MCParticle(*kset, **default_gen_inps)
        assert generic_equality_comp(particle1, particle2)
        particle3 = Components.MCParticle(px=kset[0], py=kset[1], pz=kset[2],
                                          e=kset[3], **default_gen_inps)
        assert generic_equality_comp(particle1, particle3)
        particle4 = Components.MCParticle(px=kset[0], py=kset[1], pz=kset[2],
                                          **default_gen_inps)
        if kset[-1] == 0:
            assert generic_equality_comp(particle1, particle4)
        else:
            assert not generic_equality_comp(particle1, particle4)
        particle5 = Components.MCParticle.from_repr(repr(particle1))
        assert generic_equality_comp(particle1, particle5)
        # test alternate settings
        particle6 = Components.MCParticle(*kset, **alternate_gen_inps)
        assert particle6.global_id == 100
        assert particle6.pid == 1
        assert particle6.sql_key == 2
        assert particle6.hepmc_barcode == 3
        assert particle6.is_root == True
        assert particle6.is_leaf == True
        assert particle6.start_vertex_barcode == 4
        assert particle6.end_vertex_barcode == 5
        assert particle6.status == 0
        assert particle6.generated_mass == 3.4
        assert not generic_equality_comp(particle1, particle6)
        # test setting mass
        particle7 = Components.MCParticle(px=kset[0], py=kset[1], pz=kset[2],
                                          m=kset[3], **default_gen_inps)
        # with no momentum, the energy == mass
        if sum([abs(k) for k in kset[:3]]) == 0.:
            assert generic_equality_comp(particle1, particle7)
        else:
            assert not generic_equality_comp(particle1, particle7)
    for kset in kinematic_angle:
        # test the eta phi pt version
        particle8 = Components.MCParticle(pt=kset[0], eta=kset[1], phi=kset[2],
                                          e=kset[3], **default_gen_inps)
        particle9 = Components.MCParticle(pt=kset[0], eta=kset[1], phi=kset[2],
                                          **default_gen_inps)
        if kset[-1] == 0:
            assert generic_equality_comp(particle8, particle9)
        else:
            assert not generic_equality_comp(particle8, particle9)
        particle10 = Components.MCParticle(pt=kset[0], eta=kset[1], phi=kset[2],
                                           m=kset[3], **default_gen_inps)
        # with no momentum, the energy == mass
        if kset[0] == 0:
            assert generic_equality_comp(particle8, particle10)
        else:
            assert not generic_equality_comp(particle8, particle10)
        # test repr
        particle11 = Components.MCParticle.from_repr(repr(particle8))
        assert generic_equality_comp(particle8, particle11)


def test_MyVertex():
    # some variations on the general inputs
    default_gen_inps = {'global_vertex_id' : 1,
                        'sql_key' : None,
                        'hepmc_barcode' : None,
                        'n_out' : None,
                        'n_weights' : None,
                        'n_orphans' : None}
    alternate_gen_inps = {'global_vertex_id' : 100,
                          'sql_key' : 2,
                          'hepmc_barcode' : 3,
                          'n_out' : 6,
                          'n_weights' : 8,
                          'n_orphans' : 3}
    # dummy kinematic variables
    kinematic_xyzs = ((0., 0., 0., 0.),
                      (0., 0., 0., 100.),
                      (-1000., 1000., 0.002, 100.),
                      (-1000., 1000., 0.002, 0.),
                      (-1000, 1000, 0, 0),
                      (-252, -6, -2.34, -2.6))
    for kset in kinematic_xyzs:
        # test the px py pz version
        vertex1 = Components.MyVertex(*kset, global_vertex_id=default_gen_inps['global_vertex_id'])
        assert vertex1.global_vertex_id == 1
        assert vertex1.sql_key == None
        assert vertex1.hepmc_barcode == None
        assert vertex1.n_out == None
        assert vertex1.n_weights == None
        assert vertex1.n_orphans == None
        vertex2 = Components.MyVertex(*kset, **default_gen_inps)
        assert generic_equality_comp(vertex1, vertex2)
        vertex3 = Components.MyVertex(x=kset[0], y=kset[1], z=kset[2],
                                        ctau=kset[3], **default_gen_inps)
        assert generic_equality_comp(vertex1, vertex3)
        # test alternate settings
        vertex6 = Components.MyVertex(*kset, **alternate_gen_inps)
        assert vertex6.global_vertex_id == 100
        assert vertex6.sql_key == 2
        assert vertex6.hepmc_barcode == 3
        assert vertex6.n_out == 6
        assert vertex6.n_weights == 8
        assert vertex6.n_orphans == 3
        assert not generic_equality_comp(vertex1, vertex6)
    

def test_MyTower():
    # some variations on the general inputs
    default_gen_inps = {'global_tower_id' : 1,
                        'sql_key' : None,
                        't' : None,
                        'nTimeHits' : None,
                        'eem' : None,
                        'ehad' : None,
                        'edges' : [None, None, None, None]}
    particles = [Components.MCParticle(3.4, 5.6, 1., 3.4, global_id=10)]
    alternate_gen_inps = {'global_tower_id' : 100,
                          'sql_key' : 2,
                          'particles' : particles,
                          't' : 4.5,
                          'nTimeHits' : 10,
                          'eem' : 3.3,
                          'ehad' : -2.3,
                          'edges' : [0.4, 5.5, 10.2, 1.1]}
    # dummy kinematic variables
    kinematic_angle = ((0., 0., 0., 0.),
                       (0., 0., 0., 100.),
                       (1., -1., 0.002, 100.),
                       (0.002, -2*np.pi, 2*np.pi, 0.),
                       (0, -2*np.pi, 2*np.pi, 0.),
                       (2.52, -.6, -2.34, -2.6))
    for kset in kinematic_angle:
        tower1 = Components.MyTower(*kset, global_tower_id=default_gen_inps['global_tower_id'])
        assert tower1.global_tower_id == 1
        assert tower1.sql_key == None
        assert tower1._t == None
        assert tower1.nTimeHits == None
        assert tower1.eem == None
        assert tower1.ehad == None
        assert tower1.edges == [None, None, None, None]
        tower2 = Components.MyTower(*kset, **default_gen_inps)
        assert generic_equality_comp(tower1, tower2)
        tower3 = Components.MyTower(et=kset[0], eta=kset[1], phi=kset[2],
                                    e=kset[3], **default_gen_inps)
        # default is to set by x y z t
        if sum([abs(k) for k in kset[1:3]]) != 0.:
            # particle stationary, or only momentum in x axis
            assert not generic_equality_comp(tower1, tower3)
        else:
            assert generic_equality_comp(tower1, tower3)
        # two options, set py px, py, pz
        tower5 = Components.MyTower.from_repr(repr(tower1))
        assert generic_equality_comp(tower1, tower5)
        # set by angle
        tower6 = Components.MyTower.from_repr(repr(tower3))
        assert generic_equality_comp(tower3, tower6)
        # test alternate settings
        tower7 = Components.MyTower(*kset, **alternate_gen_inps)
        assert tower7.global_tower_id == 100
        assert tower7.sql_key == 2
        assert tower7.particles == particles
        assert tower7._t == 4.5
        assert tower7.nTimeHits == 10
        assert tower7.eem == 3.3
        assert tower7.ehad == -2.3
        assert tower7.edges == [0.4, 5.5, 10.2, 1.1]
        assert not generic_equality_comp(tower1, tower7)


def test_MyTrack():
    # some variations on the general inputs
    default_gen_inps = {'global_track_id' : 1,
                        'sql_key' : None,
                        'particle_sql_key' : None,
                        'particle' : None,
                        'charge' : None,
                        'p' : None,
                        'pT' : None,
                        'eta' : None,
                        'phi' : None,
                        'ctgTheta' : None,
                        'etaOuter' : None,
                        'phiOuter' : None,
                        't' : None,
                        'x' : None,
                        'y' : None,
                        'z' : None,
                        'xd' : None,
                        'yd' : None,
                        'zd' : None,
                        'l' : None,
                        'd0' : None,
                        'dZ' : None}
    particle = Components.MCParticle(3.4, 5.6, 1., 3.4, global_id=10)
    alternate_gen_inps = {'global_track_id' : 100,
                          'sql_key' : 124,
                          'particle_sql_key' : 8916,
                          'particle' : particle,
                          'charge' : 2,
                          'p' : 82.,
                          'pT' : 3.4,
                          'eta' : 1.4,
                          'phi' : 2.2,
                          'ctgTheta' : 3.4,
                          'etaOuter' : 2.3,
                          'phiOuter' : 1.2,
                          't' : 3.4,
                          'x' : -3.5,
                          'y' : 7.8,
                          'z' : 2.2,
                          'xd' : -3.4,
                          'yd' : -10000,
                          'zd' : 10000,
                          'l' : 6.7,
                          'd0' : 3.4,
                          'dZ' : 2.2}
    # dummy kinematic variables
    kinematic_xyzs = ((0., 0., 0., 0.),
                      (0., 0., 0., 100.),
                      (-1000., 1000., 0.002, 100.),
                      (-1000., 1000., 0.002, 0.),
                      (-1000, 1000, 0, 0),
                      (-252, -6, -2.34, -2.6))
    kinematic_angle = ((0., 0., 0., 0.),
                       (0., 0., 0., 100.),
                       (1., -1., 0.002, 100.),
                       (0.002, -2*np.pi, 2*np.pi, 0.),
                       (0, -2*np.pi, 2*np.pi, 0.),
                       (2.52, -.6, -2.34, -2.6))
    for kset in kinematic_xyzs:
        track1 = Components.MyTrack(*kset, global_track_id=default_gen_inps['global_track_id'])
        assert track1.global_track_id == 1
        assert track1.sql_key == None
        assert track1.particle_sql_key == None
        assert track1.particle == None
        assert track1.charge == None
        assert track1._p == None
        assert track1._pT == None
        assert track1._eta == None
        assert track1._phi == None
        assert track1.ctgTheta == None
        assert track1.etaOuter == None
        assert track1.phiOuter == None
        assert track1._t == None
        assert track1._x == None
        assert track1._y == None
        assert track1._z == None
        assert track1.xd == None
        assert track1.yd == None
        assert track1.zd == None
        assert track1.l == None
        assert track1.d0 == None
        assert track1.dZ == None
        track2 = Components.MyTrack(*kset, **default_gen_inps)
        assert generic_equality_comp(track1, track2)
        track3 = Components.MyTrack(x_outer=kset[0], y_outer=kset[1], z_outer=kset[2],
                                    t_outer=kset[3], **default_gen_inps)
        assert generic_equality_comp(track1, track3)
        track5 = Components.MyTrack.from_repr(repr(track1))
        assert generic_equality_comp(track1, track5)
        # test alternate settings
        track6 = Components.MyTrack(*kset, **alternate_gen_inps)
        assert track6.global_track_id == 100
        assert track6.sql_key == 124
        assert track6.particle_sql_key == 8916
        assert track6.particle == particle
        assert track6.charge == 2
        assert track6._p == 82.
        assert track6._pT == 3.4
        assert track6._eta == 1.4
        assert track6._phi == 2.2
        assert track6.ctgTheta == 3.4
        assert track6.etaOuter == 2.3
        assert track6.phiOuter == 1.2
        assert track6._t == 3.4
        assert track6._x == -3.5
        assert track6._y == 7.8
        assert track6._z == 2.2
        assert track6.xd == -3.4
        assert track6.yd == -10000
        assert track6.zd == 10000
        assert track6.l == 6.7
        assert track6.d0 == 3.4
        assert track6.dZ == 2.2
        assert not generic_equality_comp(track1, track6)
        # test setting mass
        track8 = Components.MyTrack.from_repr(repr(track6))
        assert generic_equality_comp(track8, track6)


def test_MCParticleCollection():
    # empty collection
    empty = Components.MCParticleCollection(name="empty")
    assert empty.name == "empty"
    assert empty.columns == ["$p_T$", "$\\eta$", "$\\phi$", "$E$"]
    assert len(empty.pts) == 0
    assert len(empty.etas) == 0
    assert len(empty.phis) == 0
    assert len(empty.es) == 0
    assert len(empty.pxs) == 0
    assert len(empty.pys) == 0
    assert len(empty.pzs) == 0
    assert len(empty.ms) == 0
    assert len(empty.pids) == 0
    assert len(empty.sql_keys) == 0
    assert len(empty.hepmc_barcodes) == 0
    assert len(empty.global_ids) == 0
    assert len(empty.is_roots) == 0
    assert len(empty.is_leafs) == 0
    assert len(empty.start_vertex_barcodes) == 0
    assert len(empty.end_vertex_barcodes) == 0
    assert len(empty.particle_list) == 0
    assert len(empty) == 0
    # create some particles
    pxpypz = np.random.uniform(-100, 100, (100, 3))
    e = np.random.uniform(0, 200, (100, 1))
    kinematic = np.hstack((pxpypz, e))
    # set some specile cases at the end
    kinematic[-1] = [0, 0, 0, 0]
    kinematic[-2] = [10000, 0, 0, 0]
    kinematic[-3] = [-10000, 0, 0, 0]
    kinematic[-4] = [0, 0, 0, 1000]
    # make particles of them
    particles = [Components.MCParticle(*k, global_id=i) for i, k in enumerate(kinematic)]
    random = Components.MCParticleCollection(*particles)
    assert np.all(random.pxs == kinematic[:, 0])
    assert np.all(random.pys == kinematic[:, 1])
    assert np.all(random.pzs == kinematic[:, 2])
    assert np.all(random.es == kinematic[:, 3])
    assert np.all(random.global_ids == np.arange(100))
    assert len(random) == 100

# try with just particles


def test_Observables():
    # empty observables
    empty = Components.Observables(particle_collection=Components.MCParticleCollection())
    assert len(empty) == 0
    assert len(empty.global_obs_ids) == 0
    assert len(empty.pts) == 0
    assert len(empty.etas) == 0
    assert len(empty.raps) == 0
    assert len(empty.phis) == 0
    assert len(empty.es) == 0
    assert len(empty.pxs) == 0
    assert len(empty.pys) == 0
    assert len(empty.pzs) == 0
    assert len(empty.jet_allocation) == 0
    assert empty.global_to_obs == {}
    assert empty.obs_to_global == {}
    # test file read write
    with TempTestDir("ObsTest") as dir_name:
        empty.write(dir_name)
        read_obs = Components.Observables.from_file(dir_name)
        generic_equality_comp(empty, read_obs)
    # sample sets
    # create kinematics
    pxpypz = np.random.uniform(-100, 100, (100, 3))
    e = np.random.uniform(0, 200, (100, 1))
    kinematic = np.hstack((pxpypz, e))
    # set some specile cases at the end
    kinematic[-1] = [0, 0, 0, 0]
    kinematic[-2] = [10000, 0, 0, 0]
    kinematic[-3] = [-10000, 0, 0, 0]
    kinematic[-4] = [0, 0, 0, 1000]
    # make particles of them
    particles = [Components.MCParticle(*k, global_id=i, is_leaf=True) for i, k in enumerate(kinematic)]
    random = Components.MCParticleCollection(*particles)
    tracks = [Components.MyTrack(*k, global_track_id=i) for i, k in enumerate(kinematic)]
    towers = [Components.MyTower(*k, global_tower_id=i) for i, k in enumerate(kinematic)]
    # make some input versions
    inps = [(random,), {"tracks": tracks}, {"towers": towers},
            {"tracks": tracks[:50], "towers": towers[50:]},
            {"tracks": tracks[:50], "towers": towers[50:], "particle_collection": random},
            # it should make no diference what is in particle collection if tracks and towers are entered
            {"tracks": tracks[:50], "towers": towers[50:], "particle_collection": 6.555}]
    for inp in inps:
        # assert basic properties
        if isinstance(inp, dict):
            obs = Components.Observables(**inp)
        else:
            obs = Components.Observables(*inp)
        assert len(obs) == 100
        assert np.all(obs.pxs == kinematic[:, 0])
        assert np.all(obs.pys == kinematic[:, 1])
        assert np.all(obs.pzs == kinematic[:, 2])
        assert np.all(obs.es == kinematic[:, 3])
        assert np.all(obs.global_obs_ids == np.arange(100))
        # read and write to disk
        with TempTestDir("ObsTest") as dir_name:
            obs.write(dir_name)
            read_obs = Components.Observables.from_file(dir_name)
            generic_equality_comp(obs, read_obs)


