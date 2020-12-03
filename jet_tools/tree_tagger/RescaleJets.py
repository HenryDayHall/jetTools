""" The Jets are not expected to catch all the momentum of the tags,
    The jet 4 vectors are rescaled to match the energy of the average tag.
    The scaling factor varies with the jet PT and jet rapidity"""
import numpy as np
import itertools
#from ipdb import set_trace as st

# from https://stackoverflow.com/a/7997925
def polyfit2d(x, y, z, order=3):
    """
    

    Parameters
    ----------
    x :
        param y:
    z :
        param order:  (Default value = 3)
    y :
        
    order :
        (Default value = 3)

    Returns
    -------

    
    """
    # ignore nan
    has_nan = np.logical_and(np.isnan(x), np.logical_and(np.isnan(y), np.isnan(z)))
    x = x[~has_nan]
    y = y[~has_nan]
    z = z[~has_nan]
    ncols = (order + 1)**2
    G = np.zeros((x.size, ncols))
    ij = itertools.product(range(order+1), range(order+1))
    for k, (i,j) in enumerate(ij):
        G[:,k] = x**i * y**j
    m, _, _, _ = np.linalg.lstsq(G, z)
    return m


def polyval2d(x, y, m):
    """
    

    Parameters
    ----------
    x :
        param y:
    m :
        
    y :
        

    Returns
    -------

    
    """
    order = int(np.sqrt(len(m))) - 1
    ij = itertools.product(range(order+1), range(order+1))
    z = np.zeros_like(x)
    for a, (i,j) in zip(m, ij):
        z += a * x**i * y**j
    return z


def energy_poly(eventWise, jet_name, n_degrees=2, overwrite=False, append=True):
    """
    This works by fitting a polynomial to the tags and one to the jets
    then returning the ratio

    Parameters
    ----------
    eventWise :
        param jet_name:
    n_degrees :
        Default value = 2)
    overwrite :
        Default value = False)
    append :
        Default value = True)
    jet_name :
        

    Returns
    -------

    
    """
    append_name = jet_name + "_RescaleEnergy"
    if overwrite is False:
        # check if this has already been done
        if append_name in eventWise.hyperparameter_columns:
            return
    # parts of the jet
    root_name = jet_name + "_RootInputIdx"
    inputidx_name = jet_name + "_InputIdx"
    pt_name = jet_name + "_PT"
    rapidity_name = jet_name + "_Rapidity"
    energy_name = jet_name + "_Energy"
    # check number of events
    eventWise.selected_index = None
    n_events = len(eventWise.TagIndex)
    # lists for output
    tag_pts = []
    jet_pts = []
    tag_rapidities = []
    jet_rapidities = []
    tag_energies = []
    jet_energies = []
    for event_n in range(n_events):
        eventWise.selected_index = event_n
        tag_pts.append(np.mean(eventWise.PT[eventWise.TagIndex]))
        try:
            jet_pts.append(np.mean(eventWise.match_indices(pt_name, root_name, inputidx_name).flatten()))
        except Exception as e:
            st()
            eventWise
        tag_rapidities.append(np.mean(eventWise.Rapidity[eventWise.TagIndex]))
        jet_rapidities.append(np.mean(eventWise.match_indices(rapidity_name, root_name, inputidx_name).flatten()))
        tag_energies.append(np.mean(eventWise.Energy[eventWise.TagIndex]))
        jet_energies.append(np.mean(eventWise.match_indices(energy_name, root_name, inputidx_name).flatten()))
    tag_poly = polyfit2d(np.array(tag_pts), np.array(tag_rapidities), np.array(tag_energies), n_degrees)
    jet_poly = polyfit2d(np.array(jet_pts), np.array(jet_rapidities), np.array(jet_energies), n_degrees)
    energy_poly = tag_poly/jet_poly
    h_dict = {append_name: energy_poly}
    if append:
        eventWise.append_hyperparameters(**h_dict)
    else:
        return h_dict


def rescale(eventWise, jet_name, jet_indices):
    """
    

    Parameters
    ----------
    eventWise :
        param jet_name:
    jet_indices :
        
    jet_name :
        

    Returns
    -------

    
    """
    assert eventWise.selected_index is not None
    root_name = jet_name + "_RootInputIdx"
    input_name = jet_name + "_InputIdx"
    pts = eventWise.match_indices(jet_name + "_PT", input_name, root_name).flatten()[jet_indices]
    rapidities = eventWise.match_indices(jet_name + "_Rapidity", input_name, root_name).flatten()[jet_indices]
    energies = eventWise.match_indices(jet_name + "_Energy", input_name, root_name).flatten()[jet_indices]
    pxs = eventWise.match_indices(jet_name + "_Px", input_name, root_name).flatten()[jet_indices]
    pys = eventWise.match_indices(jet_name + "_Py", input_name, root_name).flatten()[jet_indices]
    pzs = eventWise.match_indices(jet_name + "_Pz", input_name, root_name).flatten()[jet_indices]
    factor = polyval2d(pts, rapidities, getattr(eventWise, jet_name + "_RescaleEnergy"))
    new_pz = pzs * factor
    new_px = pxs * factor
    new_py = pys * factor
    new_e = energies * factor
    # a particle is a tachyon if its energy**2 is < momentum**2
    #tachyons_rescale = new_e**2/(new_px**2 + new_py**2 + new_pz**2)
    #np.clip(tachyons_rescale, None, 1., out=tachyons_rescale)
    #new_px *= tachyons_rescale
    #new_py *= tachyons_rescale
    #new_pz *= tachyons_rescale
    return new_e, new_px, new_py, new_pz
 

