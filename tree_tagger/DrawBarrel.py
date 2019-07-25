import numpy as np
from ipdb import set_trace as st
import matplotlib
#from tvtk.api import tvtk
#from mayavi.scripts import mayavi2
from mayavi import mlab


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
                            name='tracks', colormap='Set2', scale_mode='none',
                            scale_factor=150, opacity=0.7)

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
                            name='tracks', colormap='Set2', scale_mode='none',
                            scale_factor=150, opacity=0.7)

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

    mlab.show()

def main():
    from tree_tagger import ReadSQL
    event, tracks, towers, observations = ReadSQL.main()
    plot_tracks_towers(tracks, towers)


if __name__ == '__main__':
    main()
