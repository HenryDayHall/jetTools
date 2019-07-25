import numpy as np
from ipdb import set_trace as st
import matplotlib
#from tvtk.api import tvtk
#from mayavi.scripts import mayavi2
from mayavi import mlab

def barrel_cuboids(barrel_radius, thetas, phis, cuboid_lengths):
    half_basesize = barrel_radius * np.pi / 200.
    cos_phis = np.cos(phis)
    sin_phis = np.sin(phis)
    center_x = barrel_radius * cos_phis
    center_y = barrel_radius * sin_phis
    center_z = barrel_radius / np.tan(thetas)
    center_vec = np.vstack((center_x, center_y, center_z)).T
    normal_vec = np.vstack((cos_phis, sin_phis, np.zeros_like(phis))).T
    height_vec = (normal_vec.T * cuboid_lengths).T
    surface_1 = np.vstack((-sin_phis, cos_phis, np.zeros_like(phis))).T
    surface_2 = np.vstack((np.zeros_like(phis), np.zeros_like(phis), np.ones_like(phis))).T
    cuboids = make_cuboids(center_vec, half_basesize, surface_1, surface_2, height_vec)
    return cuboids

def endcap_cuboids(barrel_length, barrel_radius, thetas, phis, cuboid_lengths):
    half_basesize = barrel_radius * np.pi / 200.
    cos_phis = np.cos(phis)
    sin_phis = np.sin(phis)
    abstan_thetas = np.abs(np.tan(thetas))
    center_x = barrel_length * abstan_thetas * cos_phis
    center_y = barrel_length * abstan_thetas * sin_phis
    positive_end = np.any((np.abs(thetas) < 0.5 * np.pi,
                           np.abs(thetas) > 1.5 * np.pi), axis=0)
    center_z = np.where(positive_end, barrel_length, -barrel_length)
    center_vec = np.vstack((center_x, center_y, center_z)).T
    normal_vec = np.vstack((np.zeros_like(phis), np.zeros_like(phis), np.ones_like(phis))).T
    height_vec = (np.sign(positive_end - 0.5) * normal_vec.T * cuboid_lengths).T
    surface_1 = np.vstack((np.zeros_like(phis), np.ones_like(phis), np.zeros_like(phis))).T
    surface_2 = np.vstack((np.ones_like(phis), np.zeros_like(phis), np.zeros_like(phis))).T
    cuboids = make_cuboids(center_vec, half_basesize, surface_1, surface_2, height_vec)
    return cuboids

def make_cuboids(center_vec, half_basesize, surface_1, surface_2, height_vec):
    point_1 = center_vec + half_basesize * (surface_1 + surface_2)
    point_2 = center_vec + half_basesize * (surface_1 - surface_2)
    point_3 = center_vec + half_basesize * (-surface_1 + surface_2)
    point_4 = center_vec + half_basesize * (-surface_1 - surface_2)
    point_5 = point_1 + height_vec
    point_6 = point_2 + height_vec
    point_7 = point_3 + height_vec
    point_8 = point_4 + height_vec
    cuboids = []
    for i in range(len(point_1)):
        cuboid = []
        for dim in range(3):
            coords = np.array(([[point_1[i, dim], point_2[i, dim], point_3[i, dim], point_4[i, dim]],
                                [point_1[i, dim], point_2[i, dim], point_5[i, dim], point_6[i, dim]],
                                [point_2[i, dim], point_3[i, dim], point_6[i, dim], point_7[i, dim]],
                                [point_3[i, dim], point_4[i, dim], point_7[i, dim], point_8[i, dim]],
                                [point_4[i, dim], point_1[i, dim], point_8[i, dim], point_5[i, dim]],
                                [point_5[i, dim], point_6[i, dim], point_7[i, dim], point_8[i, dim]]]))
            cuboid.append(coords)
        cuboids.append(cuboid)
    return cuboids

    
def plot_tracks_towers(track_list, tower_list):
    names = list()

    names.append("approch")
    approch_pos = [[track.xd, track.yd, track.zd] 
                  for track in track_list]
    approch_scl = [track.e for track in track_list]

    names.append("vertex")
    vertex_pos = [[track._x, track._y, track._z]
                   for track in track_list]
    vertex_scl = [track.e for track in track_list]

    names.append("outer")
    outer_pos = [[track.x, track.y, track.z]
                   for track in track_list]
    outer_scl = [track.e for track in track_list]

    # calculate barrel dimensions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    factor = 1.3
    radiusplus = np.max(np.array(outer_pos)[:, :2]) * factor
    halflengthplus = np.max(np.array(outer_pos)[:, 2]) * factor
    # angles under this are on the endcap
    barrel_theta = np.arctan(radiusplus/halflengthplus)
    tower_angles = np.array([[t.phi(), t.theta()]
                             for t in tower_list])
    on_endcap = np.any((np.abs(tower_angles[:, 1]) < barrel_theta, 
                        np.abs(tower_angles[:, 1])
                               > np.pi-barrel_theta),
                       axis=0)
    tower_energies = np.array([t.e for t in tower_list])
    tower_heights = 0.5 * radiusplus * tower_energies/np.max(tower_energies)
    dot_barrel = False
    cos_phi = np.cos(tower_angles[:, 0])
    sin_phi = np.sin(tower_angles[:, 0])
    tan_theta = np.tan(tower_angles[:, 1])
    tower_x = np.where(on_endcap,
                       halflengthplus * np.abs(tan_theta) * cos_phi,
                       radiusplus * cos_phi)
    tower_y = np.where(on_endcap,
                       halflengthplus * np.abs(tan_theta) * sin_phi,
                       radiusplus * sin_phi)
    tower_z = np.clip(radiusplus/tan_theta, -halflengthplus, halflengthplus)
    tower_pos = np.vstack((tower_x, tower_y, tower_z)).T
    tower_pos = tower_pos.tolist()
    tower_scl = [10*endcap for tower, endcap in zip(tower_list, on_endcap)]
    b_cuboids = barrel_cuboids(radiusplus, tower_angles[~on_endcap, 1], tower_angles[~on_endcap, 0], tower_energies[~on_endcap])
    c_cuboids = endcap_cuboids(halflengthplus, radiusplus, tower_angles[on_endcap, 1], tower_angles[on_endcap, 0], tower_heights[on_endcap])
    cuboids = b_cuboids + c_cuboids

    #colourmap = matplotlib.cm.get_cmap('Set2')
    #colours = [tuple(colourmap(c)[:3]) for c in np.linspace(0.03, 0.97, 9)]

    # make figure ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    mlab.figure(1, bgcolor=(0, 0, 0))
    mlab.clf()

    # plot anchor points for approch-outer lines ~~~~~~~~~~~~~~~~~~~~~
    num_tracks = len(approch_pos)
    o_a_anchor_pos = np.array(approch_pos + outer_pos)
    o_a_scalars = np.array(approch_scl + outer_scl)

    o_a_anchors = mlab.points3d(o_a_anchor_pos[:, 0], o_a_anchor_pos[:, 1],
                            o_a_anchor_pos[:, 2], o_a_scalars,
                            name='o_a_anchor', opacity=0.)

    # add approch to outer lines
    num_tracks = len(approch_pos)
    o_a_connections = np.array([[x , x + num_tracks]
                            for x in range(num_tracks)])
    o_a_anchors.mlab_source.dataset.lines = np.array(o_a_connections)
    tube = mlab.pipeline.tube(o_a_anchors, tube_radius=8)
    tube.filter.vary_radius = 'vary_radius_off'
    mlab.pipeline.surface(tube, color=(0.2, 0.9, 0.9), opacity=0.3)

    # plot anchor points for outer-tower lines ~~~~~~~~~~~~~~~~~~~~~~~
    t_o_anchor_pos = np.array(outer_pos + tower_pos)
    t_o_scalars = np.array(outer_scl + tower_scl)

    t_o_anchors = mlab.points3d(t_o_anchor_pos[:, 0], t_o_anchor_pos[:, 1],
                            t_o_anchor_pos[:, 2], t_o_scalars,
                            name='t_o_anchor', opacity=0.)

    # get particle ids for the tracks
    track_global_id = [t.global_id for t in track_list]
    t_o_connections = []
    # go through the tower list lookignf or matches
    for i, tower in enumerate(tower_list):
        gids = tower.global_ids
        for gid in gids:
            if gid in track_global_id:
                t_o_connections.append([track_global_id.index(gid), i+num_tracks])
    t_o_connections = np.array(t_o_connections)
    t_o_anchors.mlab_source.dataset.lines = np.array(t_o_connections)
    tube = mlab.pipeline.tube(t_o_anchors, tube_radius=8)
    tube.filter.vary_radius = 'vary_radius_off'
    mlab.pipeline.surface(tube, color=(0.9, 0.2, 0.9), opacity=0.3)


    # plot the vertices
    vertex_pos = np.array(vertex_pos)
    vertex_scl = np.array(vertex_scl)
    vertices = mlab.points3d(vertex_pos[:, 0], vertex_pos[:, 1],
                             vertex_pos[:, 2], vertex_scl,
                             name='vertex', colormap='hot', scale_mode='none',
                             scale_factor=100, opacity=0.7)
    # plot the approch
    approch_pos = np.array(approch_pos)
    approch_scl = np.array(approch_scl)
    vertices = mlab.points3d(approch_pos[:, 0], approch_pos[:, 1],
                             approch_pos[:, 2], approch_scl,
                             name='approch', colormap='hot', scale_mode='none',
                             scale_factor=100, opacity=0.7)
    # plot the outer
    outer_pos = np.array(outer_pos)
    outer_scl = np.array(outer_scl)
    vertices = mlab.points3d(outer_pos[:, 0], outer_pos[:, 1],
                             outer_pos[:, 2], outer_scl,
                             name='outer', colormap='hot', scale_mode='none',
                             scale_factor=100, opacity=0.7)
    # plot the towers
    if dot_barrel:
        tower_pos = np.array(tower_pos)
        tower_scl = np.array(tower_scl)
        vertices = mlab.points3d(tower_pos[:, 0], tower_pos[:, 1],
                                 tower_pos[:, 2], tower_scl,
                                 name='tower', colormap='cool', scale_mode='none',
                                 scale_factor=100, opacity=0.7)
    else:
        for cuboid in cuboids:
            mlab.mesh(*cuboid, color=(0, 0, 1))

    mlab.show()

def main():
    from tree_tagger import ReadSQL
    event, tracks, towers, observations = ReadSQL.main()
    plot_tracks_towers(tracks, towers)


if __name__ == '__main__':
    main()
