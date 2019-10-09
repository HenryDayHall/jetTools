import numpy as np
from ipdb import set_trace as st
import matplotlib
#from tvtk.api import tvtk
#from mayavi.scripts import mayavi2
from mayavi import mlab

def generate_cuboids(eventWise, barrel_length, barrel_radius, cuboid_lengths, tower_name="Tower"):
    # angles under this are on the endcap
    barrel_theta = np.arctan(barrel_radius/barrel_length)
    thetas = 2*np.arctan(np.exp(-getattr(eventWise, tower_name+"_Eta")))
    phis = getattr(eventWise, tower_name+"_Phi")
    on_endcap = np.any((np.abs(thetas) < barrel_theta, 
                        np.abs(thetas) > np.pi-barrel_theta),
                       axis=0)
    b_cuboids, b_centers, b_heights = \
        barrel_cuboids(barrel_radius, thetas[~on_endcap], phis[~on_endcap], cuboid_lengths[~on_endcap])
    e_cuboids, e_centers, e_heights = \
        endcap_cuboids(barrel_length, barrel_radius, thetas[on_endcap], phis[on_endcap], cuboid_lengths[on_endcap])
    i_b_cuboids, i_e_cuboids = iter(b_cuboids), iter(e_cuboids)
    cuboids = [next(i_e_cuboids) if endcap else next(i_b_cuboids) for endcap in on_endcap]
    i_b_centers, i_e_centers = iter(b_centers.tolist()), iter(e_centers.tolist())
    centers = [next(i_e_centers) if endcap else next(i_b_centers) for endcap in on_endcap]
    i_b_heights, i_e_heights = iter(b_heights.tolist()), iter(e_heights.tolist())
    heights = [next(i_e_heights) if endcap else next(i_b_heights) for endcap in on_endcap]
    return cuboids, centers, heights

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
    return cuboids, center_vec, height_vec

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
    return cuboids, center_vec, height_vec

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


def highlight_pos(highlight_xyz, colours=None, colourmap="cool"):
    if colourmap is False: # use a flat colour
        for scale in np.linspace(120, 200, 4):
            highlights = mlab.points3d(highlight_xyz[:, 0], highlight_xyz[:, 1],
                                       highlight_xyz[:, 2],
                                       name='highlights', color=colours, scale_mode='none',
                                       scale_factor=scale, opacity=0.06)
    else:
        for scale in np.linspace(120, 200, 4):
            highlights = mlab.points3d(highlight_xyz[:, 0], highlight_xyz[:, 1],
                                       highlight_xyz[:, 2], colours,
                                       name='highlights', colormap=colourmap, scale_mode='none',
                                       scale_factor=scale, opacity=0.06)
    

def highlight_indices(all_positions, indices, colours, colourmap="Blues"):
    if isinstance(all_positions, list):
        all_positions = np.array(all_positions)
    if len(colours) != len(indices) and colourmap is not False:
        colours = colours[indices]
    highlight_pos(all_positions[indices], colours, colourmap)

    
def plot_tracks_towers(eventWise, track_name="Track", tower_name="Tower", colour=(0.9, 0.9, 0.9), has_vertex=True, new_figure=True, bg_color=(0., 0., 0.)):
    assert eventWise.selected_index is not None, "You must select an event to plot"
    names = list()
    
    # decide if we are making tracks from X/Y/Z or angles
    use_pxyz =  hasattr(eventWise, track_name+"_dX")
    if use_pxyz:
        # approch is short for "closest approch" (in delphes), it is the inner most point of the track
        names.append("approch")
        approch_pos = np.hstack([getattr(eventWise, track_name + "_dX").reshape((-1,1)),
                                 getattr(eventWise, track_name + "_dY").reshape((-1,1)),
                                 getattr(eventWise, track_name + "_dZ").reshape((-1,1))]).tolist()
        # the commented out stuff with 'scl' in it was being used to produce variable colours
        #approch_scl = eventWise.Track_Birr.tolist()

        if has_vertex:
            names.append("vertex")
            vertex_pos = np.hstack([getattr(eventWise, track_name+ "_X").reshape((-1,1)),
                                    getattr(eventWise, track_name+ "_Y").reshape((-1,1)),
                                    getattr(eventWise, track_name+ "_Z").reshape((-1,1))]).tolist()
            #vertex_scl = eventWise.Track_Birr.tolist()

        names.append("outer")
        outer_pos = np.hstack([getattr(eventWise, track_name + "_OuterX").reshape((-1,1)),
                               getattr(eventWise, track_name + "_OuterY").reshape((-1,1)),
                               getattr(eventWise, track_name + "_OuterZ").reshape((-1,1))]).tolist()
        #outer_scl = eventWise.Track_Birr.tolist()
    else:  #using eta/phi
        assert not has_vertex
        track_eta = getattr(eventWise, track_name+"_Eta")
        track_phi = getattr(eventWise, track_name+"_Phi")
        num_tracks = len(track_eta)
        # everything just starts at the center
        approch_pos = np.zeros((num_tracks, 3)).tolist()
        # then the outer edge is put on a "tracker barrel"
        tracker_radius = 1300
        half_tracker_length = 2800
        corner_theta = np.arctan(tracker_radius/half_tracker_length)
        track_theta = 2*np.arctan(np.exp(-track_eta))
        on_endcap = np.logical_or(track_theta < corner_theta, np.pi - track_theta < corner_theta)
        zs = tracker_radius / np.tan(track_theta)
        zs = np.where(on_endcap, half_tracker_length*np.sign(zs), zs)
        radius = np.abs(np.where(on_endcap, half_tracker_length*np.tan(track_theta), tracker_radius))
        xs = radius * np.cos(track_phi)
        ys = radius * np.sin(track_phi)
        outer_pos = np.hstack([xs.reshape((-1,1)), ys.reshape((-1,1)), zs.reshape((-1,1))]).tolist()

    # calculate barrel dimensions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    factor = 1.3
    if use_pxyz:
        radius = np.max(np.array(outer_pos)[:, :2])
        halflength = np.max(np.array(outer_pos)[:, 2])
    else:
        radius = 1400
        halflength = 3000
    radiusplus = radius * factor
    halflengthplus = halflength * factor
    # this is a long way from elegant but it finds something usable for the tower heights
    try:
        tower_energies = getattr(eventWise, tower_name+"_Energy")
    except AttributeError:
        try:
            tower_energies = getattr(eventWise, tower_name+"_Et")
        except AttributeError:
            tower_energies = getattr(eventWise, tower_name+"_Pt")
    tower_lengths = 0.5 * radiusplus * tower_energies/np.max(tower_energies)
    dot_barrel = False
    cuboids, tower_pos, tower_heights = \
        generate_cuboids(eventWise, halflengthplus, radiusplus, tower_lengths, tower_name)
    tower_scl = tower_energies.tolist()

    #colourmap = matplotlib.cm.get_cmap('Set2')
    #colours = [tuple(colourmap(c)[:3]) for c in np.linspace(0.03, 0.97, 9)]

    if new_figure:
        # make figure ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        mlab.figure(1, bgcolor=bg_color)
        mlab.clf()

    # plot anchor points for approch-outer lines ~~~~~~~~~~~~~~~~~~~~~
    num_tracks = len(approch_pos)
    o_a_anchor_pos = np.array(approch_pos + outer_pos)
    #o_a_scalars = np.array(approch_scl + outer_scl)

    o_a_anchors = mlab.points3d(o_a_anchor_pos[:, 0], o_a_anchor_pos[:, 1],
                            o_a_anchor_pos[:, 2], #o_a_scalars,
                            name='o_a_anchor', opacity=0.)

    # add approch to outer lines
    num_tracks = len(approch_pos)
    #                           index of approch point, index of outer point (starts after approch points)
    o_a_connections = np.array([[x, x + num_tracks]
                            for x in range(num_tracks)])
    o_a_anchors.mlab_source.dataset.lines = np.array(o_a_connections)
    tube = mlab.pipeline.tube(o_a_anchors, tube_radius=8)
    tube.filter.vary_radius = 'vary_radius_off'
    mlab.pipeline.surface(tube, color=colour, opacity=0.3)

    # plot anchor points for outer-tower lines ~~~~~~~~~~~~~~~~~~~~~~~
    t_o_anchor_pos = np.array(outer_pos + tower_pos)
    #t_o_scalars = np.array(outer_scl + tower_scl)

    t_o_anchors = mlab.points3d(t_o_anchor_pos[:, 0], t_o_anchor_pos[:, 1],
                            t_o_anchor_pos[:, 2], #t_o_scalars,
                            name='t_o_anchor', opacity=0.)

    # if the event has particles specified for tracks and towers link tracks to towers using those
    if hasattr(eventWise, track_name+ "_Particle"):
        # get particle ids for the tracks
        track_global_id = list(getattr(eventWise, track_name + "_Particle"))
        t_o_connections = []
        # go through the tower list lookignf or matches
        for i, gids in enumerate(getattr(eventWise, tower_name + "_Particles")):
            for gid in gids:
                if gid in track_global_id:
                    #                       index of outer point, index of tower (towers start after outers) 
                    t_o_connections.append([track_global_id.index(gid), num_tracks +i])
    else:  # assume the tracks and towers exist as two one-to-one lists
        t_o_connections = [(i, num_tracks+i) for i in range(num_tracks)]
    t_o_connections = np.array(t_o_connections)
    t_o_anchors.mlab_source.dataset.lines = np.array(t_o_connections)
    tube = mlab.pipeline.tube(t_o_anchors, tube_radius=8)
    tube.filter.vary_radius = 'vary_radius_off'
    mlab.pipeline.surface(tube, color=colour, opacity=0.3)

    if has_vertex:
        # plot the vertices
        vertex_pos = np.array(vertex_pos)
        #vertex_scl = np.array(vertex_scl)
        vertices = mlab.points3d(vertex_pos[:, 0], vertex_pos[:, 1],
                                 vertex_pos[:, 2], #vertex_scl,
                                 name='vertex', color=colour, scale_mode='none',
                                 scale_factor=100, opacity=0.7)
    # plot the approch
    approch_pos = np.array(approch_pos)
    #approch_scl = np.array(approch_scl)
    vertices = mlab.points3d(approch_pos[:, 0], approch_pos[:, 1],
                             approch_pos[:, 2], #approch_scl,
                             name='approch', color=colour, scale_mode='none',
                             scale_factor=100, opacity=0.7)
    # plot the outer
    outer_pos = np.array(outer_pos)
    #outer_scl = np.array(outer_scl)
    vertices = mlab.points3d(outer_pos[:, 0], outer_pos[:, 1],
                             outer_pos[:, 2], #outer_scl,
                             name='outer', color=colour, scale_mode='none',
                             scale_factor=100, opacity=0.7)
    # plot the towers
    if dot_barrel:
        tower_pos = np.array(tower_pos)
        #tower_scl = np.array(tower_scl)
        vertices = mlab.points3d(tower_pos[:, 0], tower_pos[:, 1],
                                 tower_pos[:, 2], #tower_scl,
                                 name='tower', color=colour, scale_mode='none',
                                 scale_factor=100, opacity=0.7)
    else:
        for cuboid in cuboids:
            mlab.mesh(*cuboid, color=colour)
    return outer_pos, tower_pos, radius, halflength


def add_single(pos, colour, scale=100, name=None):
    if len(pos) == 3:
        vertices = mlab.points3d([pos[0]], [pos[1]],
                                 [pos[2]], 
                                 name=name, scale_mode='none',
                                 scale_factor=scale, color=colour)
        if name is None:
            name='single'
        else:
            mlab.text3d(*pos[:3], name, scale=scale/2, color=colour)
    elif len(pos) == 6:
        vertices = mlab.quiver3d([pos[0]], [pos[1]],
                                 [pos[2]], [pos[3]],
                                 [pos[4]], [pos[5]],
                                 name=name, scale_mode='none',
                                 scale_factor=scale, color=colour,
                                 mode='arrow')
        if name is None:
            name='single'
        else:
            mlab.text3d(*pos[:3], name, scale=scale/5, color=colour)
    else:
        raise NotImplementedError(f"pos is len {len(pos)}; should be 3 or 6.")


def colour_set(num_colours, colourmap='gist_rainbow'):
    cmap = matplotlib.cm.get_cmap(colourmap)
    colours = [cmap(i)[:3] for i in np.linspace(0., 1., num_colours)]
    return colours


def plot_beamline(length, colour=(1., 0.7, 0.2), interaction=True):
    mlab.plot3d([0, 0], [0, 0], [-length, length], color=colour, tube_radius=length/1000.)
    if interaction:
        highlight_pos(np.array([[0, 0, 0]]), colours=colour, colourmap=False)

def main():
    from tree_tagger import Components, LinkingFramework, InputTools
    repeat = True
    try:
        eventWise = Components.EventWise.from_file("/home/henry/lazy/dataset2/h1bBatch2_particles.awkd")
    except Exception:
        print("This main method contains hardlinks to data on Henry's machine. You probably want to use the file as an import instead of running it")
    while repeat:
        eventWise.selected_index = int(input("Event number: "))
        outer_pos, tower_pos, barrel_radius, halfbarrel_length = plot_tracks_towers(eventWise)
        print(f"Barrel_radius = {barrel_radius}, half barrel length = {halfbarrel_length}")
        plot_beamline(halfbarrel_length*3)
        MCtruth = LinkingFramework.MC_truth_links(eventWise)
        tracks_near_tower, towers_near_track = LinkingFramework.tower_track_proximity(eventWise)
        look_at_towers = False
        if look_at_towers:
            highlight_track = np.random.randint(0, len(eventWise.Track_Birr))
            tower_friends = towers_near_track[highlight_track]
            highlight_indices(outer_pos, [highlight_track], [0], "Pastel1")
            highlight_indices(tower_pos, tower_friends, np.ones_like(tower_friends), "Set2")
        look_at_tracks = False
        if look_at_tracks:
            highlight_tower = np.random.randint(0, len(eventWise.Tower_Energy))
            track_friends = tracks_near_tower[highlight_tower]
            highlight_indices(tower_pos, [highlight_tower], [0], "Pastel1")
            highlight_indices(outer_pos, track_friends, np.ones_like(track_friends), "Set2")
        look_at_truth = False
        if look_at_truth:
            tracks_idx, towers_idx = zip(*[[track, tower] for track, tower in MCtruth.items() if tower is not None])
            tracks_idx, towers_idx = list(tracks_idx), list(towers_idx)
            colours = np.random.random(len(towers_idx))
            highlight_indices(tower_pos, towers_idx, colours, "nipy_spectral")
            highlight_indices(outer_pos, tracks_idx, colours, "nipy_spectral")
        mlab.show()
        repeat = InputTools.yesNo_question("Repeat? ")


if __name__ == '__main__':
    main()
