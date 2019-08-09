""" a collection of scripts to assocate each jet to it's MC truth """

def allocate(tag_particles, jets):
    """
    each tag will be assigned to a jet
    both tag particles and jets must offer rap and phi methods
    """
    eta_distance = np.array([[tag.eta - jet.eta for jet in jets]
                             for tag in tag_particles])
    rap_distance = np.array([[tag.rap - jet.rap for jet in jets]
                             for tag in tag_particles])
    dist2 = np.square(eta_distance) + np.square(rap_distance)
    closest = np.min(dist2, axis=1)
    return closest

def from_root_particles(event_particles, jets):




