"""calculate shape variables """
import scipy.optimize
import numpy as np
#from ipdb import set_trace as st
from tree_tagger.stefano_shapes import shape as stefano
from tree_tagger import TrueTag, Constants
import awkward
from matplotlib import pyplot as plt 


def D_parameter(energies, pxs, pys, pzs):
    raise NotImplementedError

def C_parameter(energies, pxs, pys, pzs):
    raise NotImplementedError

def spherocity(energies, pxs, pys, pzs):
    momentums = np.vstack((pxs, pys)).T
    def to_minimise(phi):
        transverse_thrust_axis = [np.cos(phi), np.sin(phi)]
        return np.sum(np.abs(np.cross(momentums, transverse_thrust_axis)))
    best_phi = scipy.optimize.minimize_scalar(to_minimise, bounds=(-np.pi, np.pi),
                                              method='Bounded').x
    transverse_thrust_axis = [np.cos(best_phi), np.sin(best_phi)]
    fraction = np.sum(np.abs(np.cross(momentums, transverse_thrust_axis))) \
               /np.sum(np.abs(momentums))
    spherocity = 0.25*np.pi**2*fraction**2
    return spherocity


def python_shape(energies, pxs, pys, pzs):
    shape_names = ['thrust', 'oblateness', 'sphericity',
                   'heavy_jet_mass2', 'light_jet_mass2',
                   'difference_jet_mass2', 'alpanarity',
                   'planarity', 'acoplanarity', 'minor',
                   'major', 'D parameter', 'C parameter',
                   'spherocity']
    shapes = {}
    momentums = np.vstack((pxs, pys)).T
    birrs = np.sum(np.abs(momentums), axis=1)
    sum_birr = np.sum(birrs)
    def to_minimise(phi):
        transverse_thrust_axis = [np.cos(phi), np.sin(phi)]
        return -np.sum(np.abs(momentums*transverse_thrust_axis))
    best_phi = scipy.optimize.minimize_scalar(to_minimise, bounds=(-np.pi, np.pi),
                                              method='Bounded').x
    transverse_thrust_axis = [np.cos(best_phi), np.sin(best_phi)]
    shapes['Thrust'] = np.sum(np.abs(momentums*transverse_thrust_axis))/sum_birr
    shapes['Minor'] = np.sum(np.abs(np.cross(momentums, transverse_thrust_axis)))/sum_birr


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
