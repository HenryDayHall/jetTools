import numpy as np
from ipdb import set_trace as st
import matplotlib
#from tvtk.api import tvtk
#from mayavi.scripts import mayavi2
from mayavi import mlab

def generate_cuboids(eventWise, barrel_length, barrel_radius, cuboid_lengths, tower_indices, cell_basesize, tower_name="Tower"):
    """
    

    Parameters
    ----------
    eventWise :
        param barrel_length:
    barrel_radius :
        param cuboid_lengths:
    tower_name :
        Default value = "Tower")
    barrel_length :
        
    cuboid_lengths :
        

    Returns
    -------

    
    """
    # angles under this are on the endcap
    barrel_theta = np.arctan(barrel_radius/barrel_length)
    thetas = 2*np.arctan(np.exp(-getattr(eventWise, tower_name+"_Eta")[tower_indices]))
    phis = getattr(eventWise, tower_name+"_Phi")[tower_indices]
    cuboid_lengths = cuboid_lengths[tower_indices]
    on_endcap = np.any((np.abs(thetas) < barrel_theta, 
                        np.abs(thetas) > np.pi-barrel_theta),
                       axis=0)
    # calculate the barrel and endcap cuboids seperatly
    b_cuboids, b_centers, b_heights = \
        barrel_cuboids(barrel_radius, cell_basesize,
                       thetas[~on_endcap], phis[~on_endcap], cuboid_lengths[~on_endcap])
    e_cuboids, e_centers, e_heights = \
        endcap_cuboids(barrel_length, cell_basesize,
                       barrel_radius, thetas[on_endcap], phis[on_endcap],
                       cuboid_lengths[on_endcap])
    # zip them back together
    i_b_cuboids, i_e_cuboids = iter(b_cuboids), iter(e_cuboids)
    cuboids = [next(i_e_cuboids) if endcap else next(i_b_cuboids) for endcap in on_endcap]
    i_b_centers, i_e_centers = iter(b_centers.tolist()), iter(e_centers.tolist())
    centers = [next(i_e_centers) if endcap else next(i_b_centers) for endcap in on_endcap]
    i_b_heights, i_e_heights = iter(b_heights.tolist()), iter(e_heights.tolist())
    heights = [next(i_e_heights) if endcap else next(i_b_heights) for endcap in on_endcap]
    return cuboids, centers, heights


def barrel_cuboids(barrel_radius, cell_basesize, thetas, phis, cuboid_lengths):
    """
    

    Parameters
    ----------
    barrel_radius :
        param thetas:
    phis :
        param cuboid_lengths:
    thetas :
        
    cuboid_lengths :
        

    Returns
    -------

    
    """
    half_basesize = 0.5*cell_basesize
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


def endcap_cuboids(barrel_length, cell_basesize, barrel_radius, thetas, phis, cuboid_lengths):
    """
    

    Parameters
    ----------
    barrel_length :
        param barrel_radius:
    thetas :
        param phis:
    cuboid_lengths :
        
    barrel_radius :
        
    phis :
        

    Returns
    -------

    
    """
    half_basesize = 0.5*cell_basesize
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
    """
    

    Parameters
    ----------
    center_vec :
        param half_basesize:
    surface_1 :
        param surface_2:
    height_vec :
        
    half_basesize :
        
    surface_2 :
        

    Returns
    -------

    
    """
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
            coords = np.array(([[point_1[i, dim], point_2[i, dim], point_4[i, dim], point_3[i, dim]],
                                [point_1[i, dim], point_2[i, dim], point_6[i, dim], point_5[i, dim]],
                                [point_2[i, dim], point_4[i, dim], point_8[i, dim], point_6[i, dim]],
                                [point_4[i, dim], point_3[i, dim], point_7[i, dim], point_8[i, dim]],
                                [point_3[i, dim], point_1[i, dim], point_5[i, dim], point_7[i, dim]],
                                [point_7[i, dim], point_5[i, dim], point_6[i, dim], point_8[i, dim]]]))
            cuboid.append(coords)
        cuboids.append(cuboid)
    return cuboids


def highlight_pos(highlight_xyz, colours=None, colourmap="cool", scale=1):
    """
    

    Parameters
    ----------
    highlight_xyz :
        param colours: (Default value = None)
    colourmap :
        Default value = "cool")
    colours :
        (Default value = None)

    Returns
    -------

    
    """
    if colourmap is False: # use a flat colour
        for scale in np.linspace(scale*0.5, scale, 4):
            highlights = mlab.points3d(highlight_xyz[:, 0], highlight_xyz[:, 1],
                                       highlight_xyz[:, 2],
                                       name='highlights', color=colours, scale_mode='none',
                                       scale_factor=scale, opacity=0.06)
    else:
        for scale in np.linspace(scale*0.5, scale, 4):
            highlights = mlab.points3d(highlight_xyz[:, 0], highlight_xyz[:, 1],
                                       highlight_xyz[:, 2], colours,
                                       name='highlights', colormap=colourmap, scale_mode='none',
                                       scale_factor=scale, opacity=0.06)
    

def highlight_indices(all_positions, indices, colours, colourmap="Blues"):
    """
    

    Parameters
    ----------
    all_positions :
        param indices:
    colours :
        param colourmap: (Default value = "Blues")
    indices :
        
    colourmap :
        (Default value = "Blues")

    Returns
    -------

    
    """
    if isinstance(all_positions, list):
        all_positions = np.array(all_positions)
    if len(colours) != len(indices) and colourmap is not False:
        colours = colours[indices]
    highlight_pos(all_positions[indices], colours, colourmap)

    
def plot_tracks_towers(eventWise, track_name="Track", track_indices=None, 
                       tower_name="Tower", tower_indices=None,
                       colour=None, particle_jets=None,
                       has_vertex=True, connect_tracks_towers=False,
                       new_figure=True,
                       half_barrel_length=None, barrel_radius=None,
                       bg_color=(0., 0., 0.), cmap='gist_rainbow'):
    assert eventWise.selected_index is not None, "You must select an event to plot"
    names = list()
    if track_indices is None:
        track_indices = slice(None, None)
    else:
        if len(track_indices) == 0:
            return

    # if there is no particle colour and no single colour is given pick one
    if particle_jets is None and colour is None:
        colour = (0.9, 0.9, 0.9)
    if particle_jets is not None:
        track_colours, tower_colours = track_tower_colours(particle_jets, eventWise.Track_Particle, eventWise.Tower_Particles)
        

    # decide if we are making tracks from X/Y/Z or angles
    print("Getting track coordinates", flush=True)
    use_pxyz =  hasattr(eventWise, track_name+"_dX")
    if use_pxyz:
        # approch is short for "closest approch" (in delphes), it is the inner most point of the track
        names.append("approch")
        approch_pos = np.hstack([getattr(eventWise, track_name + "_dX").reshape((-1, 1)),
                                 getattr(eventWise, track_name + "_dY").reshape((-1, 1)),
                                 getattr(eventWise, track_name + "_dZ").reshape((-1, 1))])

        if has_vertex:
            names.append("vertex")
            vertex_pos = np.hstack([getattr(eventWise, track_name+ "_X").reshape((-1, 1)),
                                    getattr(eventWise, track_name+ "_Y").reshape((-1, 1)),
                                    getattr(eventWise, track_name+ "_Z").reshape((-1, 1))])

        names.append("outer")
        outer_pos = np.hstack([getattr(eventWise, track_name + "_OuterX").reshape((-1, 1)),
                               getattr(eventWise, track_name + "_OuterY").reshape((-1, 1)),
                               getattr(eventWise, track_name + "_OuterZ").reshape((-1, 1))])
    else:  #using eta/phi
        assert not has_vertex
        track_eta = getattr(eventWise, track_name+"_Eta")
        track_phi = getattr(eventWise, track_name+"_Phi")
        num_tracks = len(track_eta)
        # everything just starts at the center
        approch_pos = np.zeros((num_tracks, 3))
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
        outer_pos = np.hstack([xs.reshape((-1,1)), ys.reshape((-1,1)), zs.reshape((-1,1))])
    approch_pos = approch_pos[track_indices].tolist()
    outer_pos = outer_pos[track_indices].tolist()
    if has_vertex:
        vertex_pos = vertex_pos[track_indices].tolist()

    # calculate barrel dimensions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    factor = 1.3
    if barrel_radius is None or half_barrel_length is None:
        radius = 1
        halflength = 3
    else:
        radius = barrel_radius
        halflength = half_barrel_length
    tube_radius = radius/60
    radiusplus = radius * factor
    halflengthplus = halflength * factor
    print("Getting Tower coordinates", flush=True)
    # this is a long way from elegant but it finds something usable for the tower heights
    try:
        tower_energies = getattr(eventWise, tower_name+"_Energy")
    except AttributeError:
        try:
            tower_energies = getattr(eventWise, tower_name+"_Et")
        except AttributeError:
            tower_energies = getattr(eventWise, tower_name+"_Pt")
    tower_lengths =  radiusplus * tower_energies/np.max(tower_energies)
    dot_barrel = False
    circumference = 2*radiusplus*np.pi
    cells_round_circumference = 200
    cell_basesize = circumference / cells_round_circumference
    if tower_indices is None:
        tower_indices = slice(None, None)
    cuboids, tower_pos, tower_heights = \
        generate_cuboids(eventWise, halflengthplus, radiusplus, tower_lengths, tower_indices, cell_basesize,
                         tower_name)


    if new_figure:
        print("Making a figure", flush=True)
        # make figure ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #mlab.figure(1, bgcolor=bg_color)

        # plot the barrel ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        mlab.plot3d([0, 0], [0, 0], [-halflengthplus, halflengthplus],
                color=(0.5, 0.5, 0.8), opacity=0.6, tube_radius=radiusplus)
    # plot anchor points for approch-outer lines ~~~~~~~~~~~~~~~~~~~~~
    print("Creating anchors", flush=True)
    num_tracks = len(approch_pos)
    o_a_anchor_pos = np.array(approch_pos + outer_pos)
    if particle_jets is None:
        o_a_anchors = mlab.points3d(o_a_anchor_pos[:, 0], o_a_anchor_pos[:, 1],
                                o_a_anchor_pos[:, 2], color=colour,
                                name='o_a_anchor', opacity=0.)
    else:
        o_a_anchors = mlab.points3d(o_a_anchor_pos[:, 0], o_a_anchor_pos[:, 1],
                                o_a_anchor_pos[:, 2], np.tile(track_colours, 2),
                                name='o_a_anchor', opacity=0., colormap=cmap)


    # add approch to outer lines
    num_tracks = len(approch_pos)
    #                           index of approch point, index of outer point (starts after approch points)
    o_a_connections = np.array([[x, x + num_tracks]
                            for x in range(num_tracks)])
    o_a_anchors.mlab_source.dataset.lines = np.array(o_a_connections)
    tube = mlab.pipeline.tube(o_a_anchors, tube_radius=tube_radius)
    tube.filter.vary_radius = 'vary_radius_off'
    print("Plotting tracks", flush=True)
    mlab.pipeline.surface(tube, colormap=cmap, opacity=0.3)

    if connect_tracks_towers:
        # plot anchor points for outer-tower lines ~~~~~~~~~~~~~~~~~~~~~~~
        t_o_anchor_pos = np.array(outer_pos + tower_pos)

        if particle_jets is None:
            t_o_anchors = mlab.points3d(t_o_anchor_pos[:, 0], t_o_anchor_pos[:, 1],
                                    t_o_anchor_pos[:, 2], colour=colour,
                                    name='t_o_anchor', opacity=0.)
        else:
            t_o_anchors = mlab.points3d(t_o_anchor_pos[:, 0], t_o_anchor_pos[:, 1],
                                    t_o_anchor_pos[:, 2], np.tile(track_colours, 2),
                                    name='t_o_anchor', opacity=0., colormap=cmap)

        #if the event has particles specified for tracks and towers link tracks to towers using those
        print("Connecting tracks to towers", flush=True)
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
        tube = mlab.pipeline.tube(t_o_anchors, tube_radius=tube_radius)
        tube.filter.vary_radius = 'vary_radius_off'
        mlab.pipeline.surface(tube, colormap=cmap, opacity=0.3)

    #if has_vertex:
    #    print("Plotting vertices", flush=True)
    #    # plot the vertices
    #    vertex_pos = np.array(vertex_pos)
    #    vertices = mlab.points3d(vertex_pos[:, 0], vertex_pos[:, 1],
    #                             vertex_pos[:, 2],
    #                             name='vertex', color=colour, scale_mode='none',
    #                             scale_factor=1, opacity=0.7)
    # plot the approch
    #approch_pos = np.array(approch_pos)
    #approch = mlab.points3d(approch_pos[:, 0], approch_pos[:, 1],
    #                         approch_pos[:, 2],
    #                         name='approch', color=colour, scale_mode='none',
    #                         scale_factor=1, opacity=0.7)
    ## plot the outer
    #outer_pos = np.array(outer_pos)
    #outer = mlab.points3d(outer_pos[:, 0], outer_pos[:, 1],
    #                         outer_pos[:, 2],
    #                         name='outer', color=colour, scale_mode='none',
    #                         scale_factor=1, opacity=0.7)
    # plot the towers
    print("Plotting towers")
    if particle_jets is None:
        for cuboid in cuboids:
            mlab.mesh(*cuboid, color=colour)
    else:
        cmap = matplotlib.cm.get_cmap(cmap)
        for cuboid, colour in zip(cuboids, tower_colours):
            colour = cmap(colour)[:3]
            mlab.mesh(*cuboid, color=colour)
    return outer_pos, tower_pos, radius, halflength


def add_single(pos, colour, scale=100, name=None):
    """
    

    Parameters
    ----------
    pos :
        param colour:
    scale :
        Default value = 100)
    name :
        Default value = None)
    colour :
        

    Returns
    -------

    
    """
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
    """
    

    Parameters
    ----------
    num_colours :
        param colourmap: (Default value = 'gist_rainbow')
    colourmap :
        (Default value = 'gist_rainbow')

    Returns
    -------

    
    """
    cmap = matplotlib.cm.get_cmap(colourmap)
    colours = [cmap(i)[:3] for i in np.linspace(0., 1., num_colours)]
    return colours


def track_tower_colours(particle_jets, tracks_particle, towers_particles):
    jet_colours = {}
    i = 1
    for tower in towers_particles:
        for p in tower:
            jet = next((j for j, jet in enumerate(particle_jets) if p in jet), 0)
            jet_colours[jet] = i
            i += 1
    for jet in range(len(particle_jets)):
        if jet not in jet_colours:
            jet_colours[jet] = i
            i += 1
    particle_colours = {p:jet_colours[j] for j, jet in enumerate(particle_jets) for p in jet}
    map_down = {c: i for i, c in enumerate(set(particle_colours.values()))}
    particle_colours = {p: map_down[c] for p, c in particle_colours.items()}
    track_colours = np.fromiter((particle_colours.get(p, 0) for p in tracks_particle),
                                dtype=float)
    assert len(track_colours) == len(tracks_particle)
    tower_colours = []
    for tower in towers_particles:
        here = np.fromiter((particle_colours.get(p, np.nan) for p in tower), dtype=float)
        if np.all(np.isnan(here)):
            tower_colours.append(0.)
        else:
            tower_colours.append(np.nanmean(here))
    tower_colours = np.array(tower_colours)

    tower_max = np.max(tower_colours)
    track_max = np.max(track_colours)
    if tower_max > track_max:
        tower_mask1 = tower_colours == tower_max
        tower_mask2 = tower_colours == track_max
        track_mask1 = track_colours == tower_max
        track_mask2 = track_colours == track_max
        tower_colours[tower_mask1] = track_max
        tower_colours[tower_mask2] = tower_max
        track_colours[track_mask1] = track_max
        track_colours[track_mask2] = tower_max
    max_value = max(tower_max, track_max)
    track_colours /= max_value
    tower_colours /= max_value
    assert len(tower_colours) == len(towers_particles)
    return track_colours, tower_colours
    


def plot_beamline(length, colour=(1., 0.7, 0.2), interaction=True):
    """
    

    Parameters
    ----------
    length :
        param colour: (Default value = (1.)
    0 :
        7:
    0 :
        2):
    interaction :
        Default value = True)
    colour :
        (Default value = (1.)
    0.7 :
        
    0.2) :
        

    Returns
    -------

    
    """
    mlab.plot3d([0, 0], [0, 0], [-length, length], color=colour, tube_radius=length/1000.)
    if interaction:
        highlight_pos(np.array([[0, 0, 0]]), colours=colour, colourmap=False)


def main():
    """ """
    from jet_tools import Components, InputTools, FormJets
    eventWise_path = InputTools.get_file_name("Name the eventWise: ", '.awkd').strip()
    if eventWise_path:
        eventWise = Components.EventWise.from_file(eventWise_path)
        jets = FormJets.get_jet_names(eventWise)
        repeat = True
        barrel_radius2 = np.max((eventWise.Track_OuterX**2 +
                                 eventWise.Track_OuterY**2).flatten())
        barrel_radius = np.sqrt(barrel_radius2)
        half_barrel_length = np.max(np.abs(eventWise.Track_OuterZ.flatten()))
        while repeat:
            eventWise.selected_index = int(input("Event number: "))
            jet_name = InputTools.list_complete("Jet name (empty for none): ", jets).strip()
            if not jet_name:
                outer_pos, tower_pos, barrel_radius, halfbarrel_length\
                        = plot_tracks_towers(eventWise, barrel_radius=barrel_radius,
                                             half_barrel_length=half_barrel_length)
            else:
                jets_here = getattr(eventWise, jet_name + "_InputIdx")
                print(f"Number of jets = {len(jets_here)}")
                source_idx = eventWise.JetInputs_SourceIdx
                n_source = len(source_idx)
                jets_here = [source_idx[jet[jet < n_source]] for jet in jets_here]
                all_jet_particles = set(np.concatenate(jets_here))
                assert all_jet_particles == set(source_idx)
                results = plot_tracks_towers(eventWise, particle_jets=jets_here,
                                             barrel_radius=barrel_radius,
                                             half_barrel_length=half_barrel_length)
                outer_pos, tower_pos, barrel_radius, halfbarrel_length = results
            plot_beamline(halfbarrel_length*3)
            print(f"Barrel_radius = {barrel_radius}, half barrel length = {halfbarrel_length}")
            mlab.show()
            repeat = InputTools.yesNo_question("Repeat? ")


if __name__ == '__main__':
    main()
