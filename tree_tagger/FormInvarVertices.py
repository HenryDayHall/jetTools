""" Create primary and secondary vertices """
# TODO these ponts can have time like spereation 
# so the invarient length will be complex.....
# then you cannot cluster in it.
import numpy as np
from numpy import testing as tst
import awkward
from ipdb import set_trace as st
from scipy.cluster import hierarchy
import scipy.spatial

def truth_vertices(eventWise, jet_name, batch_length=np.inf):
    """
    

    Parameters
    ----------
    eventWise :
        
    jet_name :
        
    batch_length :
         (Default value = np.inf)

    Returns
    -------

    """
    tag_idx_name = jet_name + "_Tags"
    tag_vertex_name = jet_name + "_TrueSecondaryVertex"
    eventWise.selected_index = None
    n_events = len(getattr(eventWise, tag_idx_name))
    secondary_vertices = getattr(eventWise, tag_vertex_name, awkward.fromiter([]))
    secondary_vertices = secondary_vertices.tolist()
    start_event = len(secondary_vertices)
    end_event = min(batch_length+start_event, n_events)
    if start_event >= end_event:
        return
    for event_n in range(start_event, end_event):
        if event_n % 10 == 0:
            print(f"{100*event_n/n_events}%", end='\r', flush=True)
        eventWise.selected_index = event_n
        end_bar_codes = eventWise.End_vertex_barcode
        start_bar_codes = eventWise.Start_vertex_barcode
        bar_codes = eventWise.Vertex_barcode
        tag_idx = getattr(eventWise, tag_idx_name)
        x = eventWise.X
        y = eventWise.Y
        z = eventWise.Z
        ctau = eventWise.Ctau
        vertices_here = []
        for jet in tag_idx:
            # all tag particles should come from the primary vertex
            # almost always at the origin but not quite always
            start_vertex_indices = np.where(bar_codes == start_bar_codes[jet])[0]
            dist_primary2 = (ctau[start_vertex_indices]**2 -
                             x[start_vertex_indices]**2 -
                             y[start_vertex_indices]**2 -
                             z[start_vertex_indices]**2)

            # secondary vertices should be futher from the origin
            vertex_indices = np.where(bar_codes == end_bar_codes[jet])[0]
            dist_secondary2 = (ctau[vertex_indices]**2 -
                               x[vertex_indices]**2 -
                               y[vertex_indices]**2 -
                               z[vertex_indices]**2)
            assert np.all(dist_secondary2 > dist_primary2)
            vertices_here.append(vertex_indices)
        secondary_vertices.append(awkward.fromiter(vertices_here))
    secondary_vertices = awkward.fromiter(secondary_vertices)
    eventWise.append(**{tag_vertex_name: secondary_vertices})

# this only works on numpy vectors due to the use of numpy dot product
def closest_approches_np(start_points, direction_vectors):
    """
    

    Parameters
    ----------
    start_points :
        
    direction_vectors :
        

    Returns
    -------

    """
    # https://geomalgorithms.com/a07-_distance.html
    # m_u = ((u.v)(v.u0 - v.v0) - (v.v)(u.u0 - u.v0))/((u.u)(v.v) - (u.v)**2)
    # m_v = ((u.u)(v.u0 - v.v0) - (u.v)(u.u0 - u.v0))/((u.u)(v.v) - (u.v)**2)
    # parellel = (u.v0)/(u.u) - (u.u0)/(u.u)
    n_lines = len(start_points)
    assert len(direction_vectors) == n_lines
    start_dot_dir = np.dot(start_points, direction_vectors.T)
    dir_dot_dir = np.dot(direction_vectors, direction_vectors.T)
    dir_squared = np.diagonal(dir_dot_dir)
    dir_self = np.tile(dir_squared, (n_lines, 1))
    denominator = np.outer(dir_squared, dir_squared) - dir_dot_dir**2
    sd_diagonal = np.tile(np.diagonal(start_dot_dir), (n_lines, 1))
    numerator = dir_dot_dir * (start_dot_dir.T - sd_diagonal.T) - dir_self.T * (sd_diagonal - start_dot_dir)
    parellels = 0.5*(start_dot_dir/dir_self - sd_diagonal/dir_self)
    # if the denominator is zero the lines are parrellel
    with np.errstate(divide='ignore', invalid='ignore'):
        closest_multiples = np.where(denominator != 0, numerator/denominator, parellels)
    return closest_multiples


# this one works with the hepmath library
# so four vectors can be put in.
# should still work fine for regular numpy vectors
def closest_approches(start_points, direction_vectors, dot=None):
    """
    

    Parameters
    ----------
    start_points :
        
    direction_vectors :
        
    dot :
         (Default value = None)

    Returns
    -------

    """
    # https://geomalgorithms.com/a07-_distance.html
    # m_u = ((u.v)(v.u0 - v.v0) - (v.v)(u.u0 - u.v0))/((u.u)(v.v) - (u.v)**2)
    # m_v = ((u.u)(v.u0 - v.v0) - (u.v)(u.u0 - u.v0))/((u.u)(v.v) - (u.v)**2)
    # parellel = (u.v0)/(u.u) - (u.u0)/(u.u)
    if dot is None:
        n_lines = len(start_points)
        assert len(direction_vectors) == n_lines
        if not n_lines:
            return np.array([])
        dimensions = len(start_points[0])
        dot = define_dot(dimensions)
    start_dot_dir = np.array([[dot(s, d) for d in direction_vectors]
                              for s in start_points])
    dir_dot_dir = np.array([[dot(d1, d2) for d1 in direction_vectors]
                              for d2 in direction_vectors])
    dir_squared = np.diagonal(dir_dot_dir)
    dir_self = np.tile(dir_squared, (n_lines, 1))
    denominator = np.outer(dir_squared, dir_squared) - dir_dot_dir**2
    sd_diagonal = np.tile(np.diagonal(start_dot_dir), (n_lines, 1))
    numerator = dir_dot_dir * (start_dot_dir.T - sd_diagonal.T) - dir_self.T * (sd_diagonal - start_dot_dir)
    parellels = 0.5*(start_dot_dir/dir_self - sd_diagonal/dir_self)
    # if the denominator is zero the lines are parrellel
    with np.errstate(divide='ignore', invalid='ignore'):
        closest_multiples = np.where(denominator != 0, numerator/denominator, parellels)
    return closest_multiples


def approch_to_point(point, start_points, direction_vectors, dot=None):
    """
    

    Parameters
    ----------
    point :
        
    start_points :
        
    direction_vectors :
        
    dot :
         (Default value = None)

    Returns
    -------

    """
    if dot is None:
        if not len(start_points):
            return np.array([])
        dimensions = len(start_points[0])
        dot = define_dot(dimensions)
    multipliers = 0.5*np.fromiter((dot(s - point, d)/dot(d, d)
                                   for s, d in zip(start_points, direction_vectors)),
                                  dtype=float)
    return multipliers

def define_dot(dimensions):
    """
    

    Parameters
    ----------
    dimensions :
        

    Returns
    -------

    """
    if dimensions == 4:
        def dot(a, b):
            """
            

            Parameters
            ----------
            a :
                
            b :
                

            Returns
            -------

            """
            return a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3]
        return dot
    elif dimensions == 3:
        def dot(a, b):
            """
            

            Parameters
            ----------
            a :
                
            b :
                

            Returns
            -------

            """
            return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
        return dot
    else:
        raise NotImplementedError


def distance2_midpoints(start_points, direction_vectors, closest_multiples, midpoint2_limit=900.):
    """
    

    Parameters
    ----------
    start_points :
        
    direction_vectors :
        
    closest_multiples :
        
    midpoint2_limit :
         (Default value = 900.)

    Returns
    -------

    """
    n_lines = len(start_points)
    if not n_lines:
        return np.array([]), np.array([])
    # could do this with three vectors of 4 vectors
    # keep it flexable
    dimensions = len(start_points[0])
    dot = define_dot(dimensions)
    #start_matrix = np.tile(start_points, (n_lines, 1, 1))
    #direction_matrix = np.tile(direction_vectors, (n_lines, 1, 1))
    #displacement_matrix = np.tile(closest_multiples, (dimensions, 1, 1)).transpose(1, 2, 0)*direction_matrix
    #closest_matrix = start_matrix + displacement_matrix

    closest_points = np.zeros((*closest_multiples.shape, dimensions))
    upper_triangle = np.triu_indices(n_lines)
    closest_points[upper_triangle] = np.array([start_points[j] + closest_multiples[i, j]*direction_vectors[j]
                                                  for i, j in zip(*upper_triangle)])
    lower_triangle = np.tril_indices(n_lines)
    closest_points[lower_triangle] = np.array([start_points[i] + closest_multiples[i, j]*direction_vectors[i]
                                                  for i, j in zip(*lower_triangle)])

    midpoints = np.zeros_like(closest_points)
    # make the matrix upper triangular
    # this will prevent calculations being done twice
    midpoints[upper_triangle] = (closest_points[upper_triangle] + np.swapaxes(closest_points, 0, 1)[upper_triangle])/2
    # if these "midpoints" are more than 30cm from the origin they are no good
    # infact they are likely nonsense
    # two close-to-parallel lines in the most likely expalanation
    displacement2 = np.array([[dot(m, m) for m in row] for row in midpoints])
    too_long = np.where(displacement2 > midpoint2_limit)  # matrix coordinates
    # instead take the mean of each closest approch to the origin
    origin = np.zeros(dimensions)
    row_approch = approch_to_point(origin, start_points[too_long[0]],
                                   direction_vectors[too_long[0]],
                                   dot=dot)
    column_approch = approch_to_point(origin, start_points[too_long[1]],
                                      direction_vectors[too_long[1]],
                                      dot=dot)
    # the upper triangle is the columns
    closest_multiples[too_long] = column_approch
    # the lower triangle is the rows
    closest_multiples[too_long[1], too_long[0]] = row_approch
    # now recalculate the midpoints
    closest_points[too_long] = np.array([start_points[j] + closest_multiples[i, j]*direction_vectors[j]
                                                  for i, j in zip(*too_long)])
    closest_points[too_long[1], too_long[0]] = np.array([start_points[i] + closest_multiples[j, i]*direction_vectors[i]
                                                  for i, j in zip(*too_long)])

    midpoints[upper_triangle] = (closest_points[upper_triangle] + np.swapaxes(closest_points, 0, 1)[upper_triangle])/2
        
    # mos of everything should be within 30cm now
    bridging_vectors = np.full((n_lines, n_lines, 3), np.nan)
    bridging_vectors[upper_triangle] = closest_points[upper_triangle] - np.swapaxes(closest_points, 0, 1)[upper_triangle]
    distances2 = np.sum(bridging_vectors**2, axis=2).T  # fill lower triangle
    distances2[upper_triangle] = distances2.T[upper_triangle]  # place in upper triangle
    midpoints = np.full((n_lines, n_lines, 3), np.nan)
    midpoints[upper_triangle] = (closest_points[upper_triangle] + np.swapaxes(closest_points, 0, 1)[upper_triangle])/2
    midpoints = midpoints.transpose(1, 0, 2)
    midpoints[upper_triangle] = midpoints.transpose(1, 0, 2)[upper_triangle]
    return distances2, midpoints


# for akt deltaR=0.4 this works best with threshold 0.02
def find_vertices(eventWise, jet_name, vertex_name=None, batch_length=np.inf, threshold=0.02):
    """
    

    Parameters
    ----------
    eventWise :
        
    jet_name :
        
    vertex_name :
         (Default value = None)
    batch_length :
         (Default value = np.inf)
    threshold :
         (Default value = 0.02)

    Returns
    -------

    """
    if vertex_name is None:
        vertex_name = ''
    jet_vertex_name = f"{jet_name}_{vertex_name}Vertex"
    assignment_name = f"{jet_name}_{vertex_name}Assignment"
    jet_inputidx_name = jet_name + "_InputIdx"
    eventWise.selected_index = None
    n_events = len(getattr(eventWise, jet_inputidx_name))
    vertex_locations = getattr(eventWise, jet_vertex_name, awkward.fromiter([]))
    vertex_locations = vertex_locations.tolist()
    track_assignment = getattr(eventWise, assignment_name, awkward.fromiter([]))
    track_assignment = track_assignment.tolist()
    start_event = len(vertex_locations)
    end_event = min(batch_length+start_event, n_events)
    for event_n in range(start_event, end_event):
        if event_n % 10 == 0:
            print(f"{100*event_n/n_events}%", end='\r', flush=True)
        eventWise.selected_index = event_n
        input_idx = getattr(eventWise, jet_inputidx_name)
        source_idx = eventWise.JetInputs_SourceIdx
        is_leaf = getattr(eventWise, jet_name + "_Child1") == -1
        start_bar_codes = eventWise.Start_vertex_barcode
        bar_codes = eventWise.Vertex_barcode
        x = eventWise.X
        y = eventWise.Y
        z = eventWise.Z
        ctau = eventWise.Ctau
        px = eventWise.Px
        py = eventWise.Py
        pz = eventWise.Pz
        e = eventWise.Energy
        n_jets = len(input_idx)
        vertex_locations.append([])
        track_assignment.append([])
        for jet_n in range(n_jets):
            # make a set of start points and (unnormalised) direction vectors
            particle_idx = source_idx[input_idx[jet_n, is_leaf[jet_n]]]
            n_tracks = len(particle_idx)
            if n_tracks == 1:
                # cannot have vertices with only one track
                vertex_locations[-1].append(awkward.fromiter([]))
                track_assignment[-1].append(awkward.fromiter([]))
                continue
            vertex_idx = np.fromiter((np.where(bar_codes==code)[0][0]
                                      for code in start_bar_codes[particle_idx]),
                                     dtype=int)
            start_location = np.vstack((ctau[vertex_idx], x[vertex_idx], y[vertex_idx], z[vertex_idx])).T
            direction = np.vstack((e[particle_idx], px[particle_idx], py[particle_idx], pz[particle_idx])).T
            # calculate the multiple of each direction vector that gives 
            # the point of closest approch to every other vector
            multiple = closest_approches(start_location, direction)
            distances2, midpoints = distance2_midpoints(start_location, direction, multiple)
            assignments = np.full(n_tracks, -1, dtype=int)
            selected_midpoints = np.full((int(n_tracks/2), 3), np.nan)
            # used ot prevent already selected columns from being min
            infinities = np.zeros_like(assignments, dtype=float) 
            # merge the shortest distances first
            # if there is an odd number of tracks abandon whatever didn't get merged
            for row_n, row in enumerate(np.argsort(np.min(distances2, axis=1))[:int(n_tracks/2)]):
                min_col = np.argmin(distances2[row, row_n:] + infinities[row_n:]) + row_n
                infinities[min_col] = np.inf
                assignments[[row, min_col]] = row_n
                selected_midpoints[row_n] = midpoints[row, min_col]
            # if there are less than 4 tracks no need to cluster further
            if n_tracks < 4:
                clusters = np.arange(len(selected_midpoints))
            else:  # do a higherarchical clustering
                # TODO need to write a lorentz invarient clustering...
                # or could be using radpidity coordinates
                clusters = hierarchy.fclusterdata(selected_midpoints, threshold, criterion='distance')
            cluster_ids = sorted(set(clusters))
            cluster_midpoints = np.fromiter((np.mean(selected_midpoints[clusters == cid], axis=0)
                                             for cid in cluster_ids),
                                             dtype=float)
            # put whichever midpoint is closest to the front
            primary_idx = np.argmin(np.sum(cluster_midpoints**2, axis=1))
            primary_cid = cluster_ids[primary_idx]
            switch_with = cluster_ids[0]
            switch_with_idx = cluster_ids.index(switch_with)
            cluster_midpoints[[primary_idx, switch_with_idx]] = cluster_midpoints[[switch_with_idx, primary_idx]]
            place_holder = np.max(cluster_ids) + 1
            clusters[clusters == switch_with] = place_holder
            clusters[clusters == primary_cid] = switch_with
            clusters[clusters == place_holder] = primary_cid
            assignments[assignments != -1] = clusters[assignments[assignments != -1]]
            # now the lowest value assignment is the primary vertex
            vertex_locations[-1].append(awkward.fromiter(cluster_midpoints))
            track_assignment[-1].append(awkward.fromiter(assignments))
        vertex_locations[-1] = awkward.fromiter(vertex_locations[-1])
        track_assignment[-1] = awkward.fromiter(track_assignment[-1])
    eventWise.append(**{jet_vertex_name: awkward.fromiter(vertex_locations),
                      assignment_name: awkward.fromiter(track_assignment)})


def vertex_uncertanty(eventWise, jet_name, vertex_name):
    """
    

    Parameters
    ----------
    eventWise :
        
    jet_name :
        
    vertex_name :
        

    Returns
    -------

    """
    if vertex_name is None:
        vertex_name = ''
    jet_vertex_name = f"{jet_name}_{vertex_name}Vertex"
    tag_vertex_name = jet_name + "_TrueSecondaryVertex"
    assignment_name = f"{jet_name}_{vertex_name}Assignment"
    eventWise.selected_index = None
    vertex_locations = getattr(eventWise, jet_vertex_name, awkward.fromiter([]))
    n_events = len(vertex_locations)
    track_assignment = getattr(eventWise, assignment_name)
    tag_locations = getattr(eventWise, tag_vertex_name)
    for event_n in range(n_events):
        if event_n % 10 == 0:
            print(f"{100*event_n/n_events}%", end='\r', flush=True)
        distances_to_tag


# TODO make relativistic from here
def compare_vertices(eventWise, jet_name, vertex_name):
    """
    

    Parameters
    ----------
    eventWise :
        
    jet_name :
        
    vertex_name :
        

    Returns
    -------

    """
    vertex_name = f"{jet_name}_{vertex_name}Vertex"
    tag_vertex_name = jet_name + "_TrueSecondaryVertex"
    eventWise.selected_index = None
    n_events = len(getattr(eventWise, tag_vertex_name))
    imagined_vertices = []
    missed_vertices = []
    primary_displacement = []
    secondary_displacement = []
    for event_n in range(n_events):
        if event_n % 10 == 0:
            print(f"{100*event_n/n_events}%", end='\r', flush=True)
        eventWise.selected_index = event_n
        x = eventWise.X
        y = eventWise.Y
        z = eventWise.Z
        ctau = eventWise.Ctau
        true_vertices = getattr(eventWise, tag_vertex_name)
        reco_vertices = getattr(eventWise, vertex_name)
        for true, reco in zip(true_vertices, reco_vertices):
            if len(reco) > 0:
                primary_displacement.append(np.sum(reco[0]**2))
            assigned = np.full(len(reco), False, dtype=bool)
            found = np.full(len(true), False, dtype=bool)
            #true_pos = np.vstack((ctau[true], x[true], y[true], z[true])).T
            true_pos = np.vstack((x[true], y[true], z[true])).T
            if len(reco) > 1 and len(true) > 0:
                distances2 = scipy.spatial.distance.cdist(np.array(reco[1:].tolist()), true_pos, metric='sqeuclidean')
                for row_n in np.argsort(np.sum(distances2, axis=1)):
                    if np.all(found):
                        break
                    col_n = np.argmin(distances2[row_n] + found*np.inf)
                    assigned[row_n] = True
                    found[col_n] = True
                    reco2 = max(np.sum(reco[row_n]**2), 10**(-12))
                    secondary_displacement.append(distances2[row_n, col_n]/reco2)
            # anything not found is missed
            missed_vertices += true_pos[~found].tolist()
            # anything not assigned is imagined
            imagined_vertices += reco[~assigned].tolist()
    imagined_vertices = np.array(imagined_vertices)
    imagined_displacement = np.sqrt(np.sum(imagined_vertices**2, axis=1))
    missed_vertices = np.array(missed_vertices)
    missed_displacement = np.sqrt(np.sum(missed_vertices**2, axis=1))
    primary_displacement = np.sqrt(np.array(primary_displacement))
    secondary_displacement = np.sqrt(np.array(secondary_displacement))
    return imagined_vertices, imagined_displacement, missed_vertices, missed_displacement, primary_displacement, secondary_displacement



