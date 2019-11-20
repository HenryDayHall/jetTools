""" Create primary and secondary vertices """
import numpy as np
from numpy import testing as tst
import awkward
from ipdb import set_trace as st

def truth_vertices(eventWise, jet_name, start_event=0, end_event=np.inf):
    tag_idx_name = jet_name + "_Tags"
    tag_vertex_name = jet_name + "_TrueSecondaryVertex"
    eventWise.selected_index = None
    n_events = len(getattr(eventWise, tag_idx_name))
    end_event = min(end_event, n_events)
    secondary_vertices = getattr(eventWise, tag_vertex_name, awkward.fromiter([]))
    secondary_vertices = secondary_vertices.tolist()
    if len(secondary_vertices) != start_event:
        secondary_vertices += [np.nan for _ in range(start_event - len(secondary_vertices))]
    for event_n in range(start_event, end_event):
        eventWise.selected_index = event_n
        end_bar_codes = eventWise.End_vertex_barcode
        start_bar_codes = eventWise.Start_vertex_barcode
        bar_codes = eventWise.Vertex_barcode
        tag_idx = getattr(eventWise, tag_idx_name)
        x = eventWise.X
        y = eventWise.Y
        z = eventWise.Z
        vertices_here = []
        for jet in tag_idx:
            # all tag particles should come from the primary vertex
            # almost always at the origin but not quite always
            start_vertex_indices = np.where(bar_codes == start_bar_codes[jet])[0]
            dist_primary2 = (x[start_vertex_indices]**2 +
                             y[start_vertex_indices]**2 +
                             z[start_vertex_indices]**2)

            # secondary vertices should be futher from the origin
            vertex_indices = np.where(bar_codes == end_bar_codes[jet])[0]
            dist_secondary2 = (x[vertex_indices]**2 +
                               y[vertex_indices]**2 +
                               z[vertex_indices]**2)
            assert np.all(dist_secondary2 > dist_primary2)
            vertices_here.append([vertex_indices])
        secondary_vertices.append(awkward.fromiter(vertices_here))
    secondary_vertices = awkward.fromiter(secondary_vertices)
    eventWise.append({tag_vertex_name: secondary_vertices})

def closest_approches(start_points, direction_vectors):
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
    closest_multiples = np.where(denominator!=0, numerator/denominator, parellels)
    return closest_multiples


def distance2_midpoints(start_points, direction_vectors, closest_multiples):
    n_lines = len(start_points)
    upper_triangle = np.triu_indices(n_lines)
    start_matrix = np.tile(start_points, (n_lines, 1, 1))
    #start_matrix[upper_triangle] = np.swapaxes(start_matrix, 0, 1)[upper_triangle]
    direction_matrix = np.tile(direction_vectors, (n_lines, 1, 1))
    #direction_matrix[upper_triangle] = np.swapaxes(direction_matrix, 0, 1)[upper_triangle]
    closest_matrix = start_matrix + np.tile(closest_multiples, (3, 1, 1)).transpose(1, 2, 0)*direction_matrix
    st()
    bridging_vectors = np.full((n_lines, n_lines, 3), np.nan)
    bridging_vectors[upper_triangle] = closest_matrix[upper_triangle] - np.swapaxes(closest_matrix, 0, 1)[upper_triangle]
    distances2 = np.sum(bridging_vectors**2, axis=2).T  # fill lower triangle
    distances2[upper_triangle] = distances2.T[upper_triangle]  # place in upper triangle
    midpoints = np.full((n_lines, n_lines, 3), np.nan)
    midpoints[upper_triangle] = (closest_matrix[upper_triangle] + np.swapaxes(closest_matrix, 0, 1)[upper_triangle])/2
    return distances2, midpoints


def find_vertices(eventWise, jet_name, start_event=0, end_event=np.inf):
    vertex_name = jet_name + "_SecondaryVertex"
    jet_inputidx_name = eventWise, jet_name + "_InputIdx"
    eventWise.selected_index = None
    n_events = len(getattr(eventWise, jet_inputidx_name))
    end_event = min(end_event, n_events)
    secondary_vertices = getattr(eventWise, vertex_name, awkward.fromiter([]))
    secondary_vertices = secondary_vertices.tolist()
    for event_n in range(start_event, end_event):
        eventWise.selected_index = event_n
        input_idx = getattr(eventWise, jet_inputidx_name)
        source_idx = eventWise.JetInputs_SourceIdx
        start_bar_codes = eventWise.Start_vertex_barcode
        bar_codes = eventWise.Vertex_barcode
        x = eventWise.X
        y = eventWise.Y
        z = eventWise.Z
        px = eventWise.Px
        py = eventWise.Py
        pz = eventWise.Pz
        n_jets = len(input_idx)
        for jet_n in range(n_jets):
            # make a set of start points and (unnormalised) direction vectors
            particle_idx = source_idx[input_idx[jet_n]]
            vertex_idx = np.array([np.where([bar_codes==code])[0][0]
                                   for code in start_bar_codes[particle_idx]])
            start_location = np.vstack((x[vertex_idx], y[vertex_idx], z[vertex_idx])).T
            direction = np.vstack((px[particle_idx], py[particle_idx], pz[particle_idx])).T
            # calculate the multiple of each direction vector that gives 
            # the point of closest approch to every other vector








