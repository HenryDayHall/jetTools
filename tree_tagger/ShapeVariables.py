"""calculate shape variables """
import scipy.optimize
import scipy.linalg
import numpy as np
#from ipdb import set_trace as st
from tree_tagger.stefano_shapes import shape as stefano
from tree_tagger import TrueTag, Constants
import awkward
from matplotlib import pyplot as plt 


def python_shape(energies, pxs, pys, pzs):
    shape_names = ['heavy_jet_mass2', 'light_jet_mass2',
                   'difference_jet_mass2', 
                   'planarity', 'acoplanarity',
                   'spherocity']
    shapes = {}
    # https://home.fnal.gov/~mrenna/lutp0613man2/node234.html
    # precalculate some quantities
    momentums = np.vstack((pxs, pys, pzs)).T
    birrs2 = np.sum(momentums**2, axis=1)
    sum_birr = np.sqrt(np.sum(birrs2))
    # Thrust
    def to_minimise(theta_phi):
        sin_theta = np.sin(theta_phi[0])
        thrust_axis = [sin_theta*np.cos(theta_phi[1]),
                       sin_theta*np.sin(theta_phi[1]),
                       np.cos(theta_phi[0])]
        return -np.sum(np.abs(momentums*thrust_axis))
    theta_phi_bounds = ((0, np.pi), (-np.pi, np.pi))
    best_theta_phi = scipy.optimize.minimize(to_minimise, np.zeros(2),
                                             bounds=theta_phi_bounds).x
    sin_theta = np.sin(best_theta_phi[0])
    thrust_axis = [sin_theta*np.cos(best_theta_phi[1]),
                   sin_theta*np.sin(best_theta_phi[1]),
                   np.cos(best_theta_phi[0])]
    momentum_dot_thrust = np.dot(momentums, thrust_axis)
    shapes['Thrust'] = np.sum(np.abs(momentum_dot_thrust))/sum_birr
    # the major thrust has an exist in the plane perpendicular to the thrust
    if best_theta_phi[0] in (np.pi, 0):
        # along the z axis
        perp1 = np.array([1, 0, 0])
        perp2 = np.array([0, 1, 0])
    else:
        perp1 = np.cross(np.array([0, 0, 1]), thrust_axis)
        perp1 /= np.sqrt(np.sum(perp1**2))
        perp2 = np.cross(perp1, thrust_axis)
        perp2 /= np.sqrt(np.sum(perp2**2))
    def to_minimise_major(alpha):
        major_thrust_axis = np.cos(alpha)*perp1 + np.sin(alpha)*perp2
        return -np.sum(np.abs(momentums*major_thrust_axis))
    best_alpha = scipy.optimize.minimize_scalar(to_minimise_major, bounds=(-np.pi, np.pi),
                                                method='Bounded')
    major_thrust_axis = np.cos(best_alpha)*perp1 + np.sin(best_alpha)*perp2
    momentum_dot_major = np.dot(momentums, major_thrust_axis)
    shapes['Major'] = np.sum(np.abs(momentum_dot_major))/sum_birr
    minor_direction = np.cross(major_thrust_axis, thrust_axis)
    minor_direction /= np.sqrt(np.sum(minor_direction**2))
    sum_momentum_dot_minor = max(np.sum(np.abs(np.dot(momentums, minor_direction))),
                                 np.sum(np.abs(np.dot(momentums, -minor_direction))))
    shapes['Minor'] = sum_momentum_dot_minor/sum_birr
    shapes['Oblateness'] = shapes['Major'] - shapes['Minor']

    # sphericity
    dimension = 3
    per_mom_tensor = np.ones((dimension, dimension, len(momentums)))
    for i in range(dimension):
        per_mom_tensor[:, i, :] *= momentums.T
        per_mom_tensor[i, :, :] *= momentums.T
    mom_tensor = np.sum(per_mom_tensor, axis=2)
    eigenvalues, _ = scipy.linalg.eig(mom_tensor/sum_birr**2)
    eigenvalues = np.sort(eigenvalues)/np.sum(eigenvalues)
    shapes['Sphericity'] = 1.5*(eigenvalues[0] + eigenvalues[1])
    shapes['Aplanarity'] = 1.5*eigenvalues[0]
    lin_mom_tensor = np.sum(per_mom_tensor/np.sqrt(birrs2).reshape((-1, 1, 1)), axis=2)
    eigenvalues, _ = scipy.linalg.eig(lin_mom_tensor/sum_birr)
    eigenvalues = np.sort(eigenvalues)/np.sum(eigenvalues)
    shapes['Cparameter'] = 3*(eigenvalues[2]*eigenvalues[1] +
                              eigenvalues[2]*eigenvalues[0] +
                              eigenvalues[i]*eigenvalues[0])
    shapes['Dparameter'] = 27*np.product(eigenvalues)


def shape(energies, pxs, pys, pzs):
    """
    

    Parameters
    ----------
    energies :
        param pxs:
    pys :
        param pzs:
    pxs :
        
    pzs :
        

    Returns
    -------

    """
    if len(energies) > 1 and len(energies) < 5:
        return stefano.shape(energies, pxs, pys, pzs, my_dir="./tree_tagger/stefano_shapes/")
    return None


def append_jetshapes(eventWise, jet_name, batch_length=100, silent=False, jet_pt_cut='default', tag_before_pt_cut=True):
    """
    

    Parameters
    ----------
    eventWise :
        param jet_name:
    batch_length :
        Default value = 100)
    silent :
        Default value = False)
    jet_pt_cut :
        Default value = 'default')
    tag_before_pt_cut :
        Default value = True)
    jet_name :
        

    Returns
    -------

    """
    # name the vaiables to be cut on
    inputidx_name = jet_name + "_InputIdx"
    rootinputidx_name = jet_name+"_RootInputIdx"
    # check this jet is tagged in this event
    if jet_pt_cut is None or tag_before_pt_cut:
        name_tags = jet_name + "_Tags"
    else:
        if jet_pt_cut == 'default':
            jet_pt_cut = Constants.min_jetpt
        name_tags = f"{jet_name}_{int(jet_pt_cut)}Tags"
    if name_tags not in eventWise.columns:
        TrueTag.add_tags(eventWise, jet_name, 0.4, np.inf)
    shape_names = ['thrust', 'oblateness', 'sphericity',
                   'heavy_jet_mass2', 'light_jet_mass2',
                   'difference_jet_mass2', 'alpanarity',
                   'planarity', 'acoplanarity', 'minor',
                   'major', 'D parameter', 'C parameter',
                   'spherocity']
    append_names = {shape: jet_name + "_" + shape.replace(' ', '').replace('_', '').capitalize()
                    for shape in shape_names}
    eventWise.selected_index = None
    #content = {name: list(getattr(eventWise, name, []))
    #           for name in append_names.values()}
    content = {name: []
               for name in append_names.values()}
    n_events = len(eventWise.JetInputs_Energy)
    start_point = len(list(content.values())[0])
    if start_point >= n_events:
        if not silent:
            print("Finished")
        return True
    end_point = min(n_events, start_point+batch_length)
    if not silent:
        print(f" Starting at {100*start_point/n_events}%")
        print(f" Will stop at {100*end_point/n_events}%")
    # updated_dict will be replaced in the first batch
    thrust_issues = []
    hjm_issues = []
    for event_n in range(start_point, end_point):
        if event_n % 10 == 0 and not silent:
            print(f"{100*event_n/n_events}%", end='\r', flush=True)
        eventWise.selected_index = event_n
        # select the jets that have been tagged
        jet_idx = [i for i, tags in enumerate(getattr(eventWise, name_tags))
                   if len(tags)>0]
        # we cna ony work with events with at least two but less that 5 jets
        if len(jet_idx) > 1 and len(jet_idx) < 5:
            jet_px = eventWise.match_indices(jet_name + "_Px", rootinputidx_name, inputidx_name).flatten()[jet_idx]
            jet_py = eventWise.match_indices(jet_name + "_Py", rootinputidx_name, inputidx_name).flatten()[jet_idx]
            jet_pz = eventWise.match_indices(jet_name + "_Pz", rootinputidx_name, inputidx_name).flatten()[jet_idx]
            jet_e = eventWise.match_indices(jet_name + "_Energy", rootinputidx_name, inputidx_name).flatten()[jet_idx]
            call, shape_dict = shape(jet_e, jet_px, jet_py, jet_pz)
            report = f"INPUT {call} OUTPUT {shape_dict}\n"
            if shape_dict["thrust"] > 1.:
                thrust_issues.append(report)
            if shape_dict['heavy_jet_mass2'] < 0.0001:
                hjm_issues.append(report)
            for name in shape_dict:
                content[append_names[name]].append(shape_dict[name])
        else:
            for name in content:
                content[name].append(np.nan)
    content = {name: awkward.fromiter(content[name]) for name in content}
    eventWise.append(**content)
    with open("jet_thrust_problems.dat", 'w') as out_file:
        out_file.writelines(thrust_issues)
    with open("jet_heavymass_problems.dat", 'w') as out_file:
        out_file.writelines(hjm_issues)
    

def append_tagshapes(eventWise, batch_length=100, silent=False, jet_pt_cut='default', tag_before_pt_cut=True):
    """
    

    Parameters
    ----------
    eventWise :
        param batch_length:  (Default value = 100)
    silent :
        Default value = False)
    jet_pt_cut :
        Default value = 'default')
    tag_before_pt_cut :
        Default value = True)
    batch_length :
         (Default value = 100)

    Returns
    -------

    """
    # name the vaiables to be cut on
    shape_names = ['thrust', 'oblateness', 'sphericity',
                   'heavy_jet_mass2', 'light_jet_mass2',
                   'difference_jet_mass2', 'alpanarity',
                   'planarity', 'acoplanarity', 'minor',
                   'major', 'D parameter', 'C parameter',
                   'spherocity']
    if jet_pt_cut is None or tag_before_pt_cut:
        name_tag = "Tag"
    else:
        if jet_pt_cut == 'default':
            jet_pt_cut = Constants.min_jetpt
        name_tag = f"Tag{int(jet_pt_cut)}"
    append_names = {shape: name_tag + shape.replace(' ', '').replace('_', '').capitalize() 
                              for shape in shape_names}
    eventWise.selected_index = None
    content = {name: list(getattr(eventWise, name, []))
               for name in append_names.values()}
    n_events = len(eventWise.JetInputs_Energy)
    start_point = len(list(content.values())[0])
    if start_point >= n_events:
        if not silent:
            print("Finished")
        return True
    end_point = min(n_events, start_point+batch_length)
    if not silent:
        print(f" Starting at {100*start_point/n_events}%")
        print(f" Will stop at {100*end_point/n_events}%")
    # updated_dict will be replaced in the first batch
    thrust_issues = []
    hjm_issues = []
    for event_n in range(start_point, end_point):
        if event_n % 10 == 0 and not silent:
            print(f"{100*event_n/n_events}%", end='\r', flush=True)
        eventWise.selected_index = event_n
        tag_idx = eventWise.TagIndex
        # there shoudl always be 4 tags
        if len(tag_idx) != 4:
            for name in append_names.values():
                content[name].append(np.nan)
            continue  # skip the shape dict
        tag_px = eventWise.Px[tag_idx]
        tag_py = eventWise.Py[tag_idx]
        tag_pz = eventWise.Pz[tag_idx]
        tag_e = eventWise.Energy[tag_idx]
        call, shape_dict = shape(tag_e, tag_px, tag_py, tag_pz)
        report = f"INPUT {call} OUTPUT {shape_dict}\n"
        if shape_dict["thrust"] > 1.:
            thrust_issues.append(report)
        if shape_dict['heavy_jet_mass2'] < 0.0001:
            hjm_issues.append(report)
        for name in shape_dict:
            content[append_names[name]].append(shape_dict[name])
    content = {name: awkward.fromiter(content[name]) for name in content}
    eventWise.append(**content)
    with open("tag_thrust_problems.dat", 'w') as out_file:
        out_file.writelines(thrust_issues)
    with open("tag_heavymass_problems.dat", 'w') as out_file:
        out_file.writelines(hjm_issues)


def test():
    """ """
    vectors = np.random.rand(4, 3)
    vectors[0] *= 20  # add some energy....
    print(shape(*vectors))


def plot_shapevars(eventWise, jet_name=None, jet_pt_cut=None):
    """
    

    Parameters
    ----------
    eventWise :
        param jet_name:  (Default value = None)
    jet_pt_cut :
        Default value = None)
    jet_name :
         (Default value = None)

    Returns
    -------

    """
    eventWise.selected_index = None
    shape_names = ['thrust', 'oblateness', 'sphericity',
                   'heavy_jet_mass2', 'light_jet_mass2',
                   'difference_jet_mass2', 'alpanarity',
                   'planarity', 'acoplanarity', 'minor',
                   'major', 'D parameter', 'C parameter',
                   'spherocity']
    append_names = [shape.replace(' ', '').replace('_', '').capitalize() 
                    for shape in shape_names]
    has_jets = jet_name is not None
    n_plots = len(append_names) + has_jets  # last plot for the jet params
    n_cols = int(np.sqrt(len(append_names)))
    n_rows = int(np.ceil(len(append_names)/n_cols))
    fig, axarry = plt.subplots(n_rows, n_cols)
    if jet_pt_cut is None:
        name_tag = "Tag"
    else:
        if jet_pt_cut == 'default':
            jet_pt_cut = Constants.min_jetpt
        name_tag = f"Tag{int(jet_pt_cut)}"
    if has_jets:
        fig.suptitle(jet_name)
    for name, ax in zip(append_names, axarry.flatten()):
        tag_data = getattr(eventWise, name_tag + name)
        ax.hist(tag_data, density=True, histtype='step', label=f'{name_tag} ({len(tag_data)})')
        if has_jets:
            jet_data = getattr(eventWise, '_'.join((jet_name, name)))
            #jet_data[np.isnan(jet_data)] = np.min(jet_data)  - 1.
            jet_data = jet_data[~np.isnan(jet_data)]
            jet_data = jet_data[jet_data != 0.]
            assert not np.any(np.isnan(jet_data))
            try:
                ax.hist(jet_data, density=True, histtype='step', label=f'Jets ({len(jet_data)})')
            except IndexError:
                print(f"Couldn't histogram {name} for {jet_name} in {eventWise.save_name}")
        ax.set_ylabel("Density")
        ax.set_xlabel(name)
        ax.legend()
    fig.set_size_inches(10, 11)
    plt.tight_layout()
    dir_name = 'images/shape/'
    if has_jets:
        plt.savefig(dir_name + jet_name + '_shapevars.png')
    else:
        plt.savefig(dir_name + 'tags_shapevars.png')


def pairplot():
    """ """
    pass

if __name__ == '__main__':
    test()
