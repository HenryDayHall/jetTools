""" compare two jet clustering techniques """
import tabulate
import awkward
import ast
import csv
import os
import pickle
import matplotlib
import awkward
from ipdb import set_trace as st
from tree_tagger import Components, TrueTag, InputTools, FormJets, Constants, RescaleJets, FormShower, JetQuality
import sklearn.metrics
import sklearn.preprocessing
from matplotlib import pyplot as plt
import numpy as np
import scipy.stats
import bokeh, bokeh.palettes, bokeh.models, bokeh.plotting, bokeh.transform
import socket

def parameter_step(records, jet_class, ignore_parameteres=None):
    """
    Select a varient of the best jet in class that has not yet been tried

    Parameters
    ----------
    records :
        param jet_class:
    ignore_parameteres :
        Default value = None)
    jet_class :
        

    Returns
    -------

    
    """
    array = records.typed_array()
    # get the jets parameter list 
    names = FormJets.cluster_classes
    jet_parameters = list(names[jet_class].param_list.keys())
    all_parameters = jet_parameters + ['jet_class']
    parameter_idxs = [(name, records.indices[name]) for name in all_parameters]
    all_parameters, parameter_indices = zip(*sorted(parameter_idxs, key=lambda x: x[1]))
    print(f"All parameters; {all_parameters}")
    if ignore_parameteres is None:
        ignore_parameteres = []
    # sets of parameters to vary
    parameter_sets = {name: set(array[:, records.indices[name]]) for name in jet_parameters
                      if name not in ignore_parameteres}
    # for some parameter sets, fix the values
    parameter_sets['DeltaR'] = np.linspace(0.1, 1.5, 15)
    parameter_sets['ExponentMultiplier'] = np.linspace(-1., 1., 21)
    if 'AffinityCutoff' in parameter_sets:
        knn = [('knn', c) for c in np.arange(1, 6)]
        distance = [('distance', c) for c in np.linspace(0., 10., 11)]
        parameter_sets['AffinityCutoff'] = knn + distance + [None]
    # clip the array to the parameters themselves
    to_compare = [i for i, name in enumerate(all_parameters)
                  if name not in ignore_parameteres]
    array = array[:, parameter_indices][:, to_compare]
    # also only consider scored rows (the rest may not really exist)
    array = array[records.scored]
    steps_taken = 1
    while steps_taken < 20:
        current_best = records.best(jet_class=jet_class, num_items=steps_taken, return_cols=parameter_indices)[0]
        print(f"Searching round {current_best}")
        new_step = np.copy(current_best)
        # go through all the parameters looking for a substitution that hasn't been tried
        names = list(parameter_sets.keys())
        np.random.shuffle(names)
        for name in names:
            avalible = list(parameter_sets[name])
            np.random.shuffle(avalible)
            idx = all_parameters.index(name)
            for value in avalible:
                new_step[idx] = value
                matching = np.all(new_step[to_compare] == array,
                                  axis=1)
                if not np.any(matching):
                    parameters = {name: new_step[all_parameters.index(name)] for name in jet_parameters}
                    # check it's not an illegal combination
                    if parameters_valid(parameters, jet_class):
                        return parameters
            # reset this parameter
            new_step[idx] = current_best[idx]
        # if we get here, no luck in the first round
        # try the next best entry
        steps_taken += 1
    print("Cannot find free space round the 20 best points")


# code for making scores
def calculate_fpr_tpr(eventWise, jet_name, jet_idxs, ctag):
    """
    

    Parameters
    ----------
    eventWise :
        
    jet_name :
        
    jet_idxs :
        
    ctag :
        

    Returns
    -------

    """
    assert eventWise.selected_index is not None
    is_root = getattr(eventWise, jet_name+"_Parent")[jet_idxs] == -1
    # consider the 4 jets with highest b_energy
    # and at least 50% b energy to be b_jets
    # not that ctag already includes jet energy as it's weighting
    ctag = ctag[jet_idxs]
    b_idx = np.argsort(ctag[is_root])[-4:]
    b_idx = b_idx[ctag[is_root][b_idx].flatten() > 0.5]
    n_positives = np.sum(ctag.flatten())
    n_negatives = np.sum(1-ctag.flatten())
    true_positives = np.sum(ctag[b_idx].flatten())
    false_positives = np.sum(1-ctag[b_idx].flatten())
    if n_positives == 0:
        tpr = 1.  # no positives avalible...
    else:
        tpr = true_positives / n_positives
        # probably happens
    if n_negatives == 0:
        fpr = 0.
    else:
        fpr = false_positives / n_negatives
    return tpr, fpr


# code for plotting scores
def parameter_comparison(records, c_name="percentfound", cuts=True):
    """
    

    Parameters
    ----------
    records :
        
    cuts :
        ( Default value = True)
    c_name :
        (Default value = "percentfound")

    Returns
    -------

    
    """
    array = records.typed_array()[records.scored]
    col_names = ["s_distance", "bdecendant_tpr", "bdecendant_fpr", "distance_to_HiggsMass", "distance_to_LightMass", "quality_width", "quality_fraction"]
    # filter anything that scored too close to zero in pt or rapidity
    # these are soem kind of bug
    small = 0.001
    for name in col_names:
        if "score" in name:
            array = array[array[:, records.indices[name]] > small]
    # now we have a scored valid selection
    y_cols = {name: array[:, records.indices[name]].astype(float) for name in col_names
              if name !='TagAngle'}
    sufficient_jets = array[:, records.indices["njets"]].astype(float) > 0.5
    if cuts:
        angles = array[:, records.indices["symmetric_diff(Phi)"]].astype(float)
        good_angle = angles < 1.
        #good_distance = y_cols["s_distance"] < 100000
        mask = np.logical_and(good_angle, sufficient_jets)
        #mask = np.logical_and(good_distance, mask)
        array = array[mask]
        y_cols = {name: y_cols[name][mask] for name in y_cols}
        sufficient_jets = sufficient_jets[mask]
    c_col = array[:, records.indices[c_name]].astype(float)
    # make the data dict
    data_dict = {}
    data_name = {}
    name_data = {}
    max_len = 15
    # some work on the names is needed for for hover to work
    for name, i in records.indices.items():
        if 'uncert' in name or 'std' in name:
            continue
        data_dict[str(i)] = np.array([str(x)[:max_len] for x in array[:, i]])
        data_name[str(i)] = name # somr of the names have probem characters in them
        name_data[name] = str(i) # need to be able to invert this
    use_names = ["jet_class", "DeltaR", "PhyDistance", "ExponentMultiplier",
                 "TagAngle", "AffinityCutoff", "Laplacien",
                 "NumEigenvectors", "AffinityType", "WithLaplacienScaling"]
    hover = bokeh.models.HoverTool(tooltips=[(data_name[i], "@" + i) for i in data_dict
                                             if data_name[i] in use_names])
    data_dict['colour'] = c_col
    data_dict['alpha'] = [0.6 if suf else 0.2 for suf in sufficient_jets]
    # now we are done altering the data dict, make a copy
    orignal_data_dict = {name: np.copy(data_dict[name]) for name in data_dict}
    # fix the colours
    mapper = bokeh.models.LinearColorMapper(palette=bokeh.palettes.Plasma10,
                                            low=min(c_col), high=max(c_col))
    # fix the markers
    marker_name = 'jet_class'
    marker_key = str(records.indices[marker_name])
    marker_shapes = ['triangle', 'hex', 'circle_x', 'square']
    marker_matches = sorted(set(data_dict[marker_key]))
    markers = bokeh.transform.factor_mark(marker_key, marker_shapes[:len(marker_matches)],
                                          marker_matches)
    plots = []
    sources = {}
    original_positions = {}
    ignore_cols = list(records.evaluation_columns) + ["jet_id", "match_error"]
    parameter_order = [name for name in sorted(records.indices.keys()) if name not in ignore_cols if len(set(array[:, records.indices[name]])) > 1]
    mask_dict = {}
    for name in parameter_order:
        print(name + "~"*5)
        # the data_dict contains strings, we need real values
        parameter_values = array[:, records.indices[name]]
        catigorical = np.any([hasattr(value, '__iter__') for value in set(parameter_values)])
        if catigorical:
            deciding_value = next(value for value in set(parameter_values) if hasattr(value, '__iter__'))
            scale, positions, label_dict, label_catigories = make_ordinal_scale(parameter_values)
            print(f"label_dict = {label_dict}, scale={scale}, set(positions) = {set(positions)}")
            labels = [label_dict[key] for key in scale]  # to fix the order
            boxes = bokeh.models.widgets.CheckboxGroup(labels=labels, active=list(range(len(labels))))
        else:
            scale, positions, label_dict, bottom, top = make_float_scale(parameter_values)
            has_slider = bottom != top
            if has_slider:
                slider = bokeh.models.RangeSlider(title=name, start=bottom, end=top,
                                                 value=(bottom, top), step=(top-bottom)/20,
                                                 orientation='vertical')
        original_positions[name] = np.copy(positions)
        plots.append([])
        sources[name] = {}
        for col_name, y_col in y_cols.items():
            # first the intresting jets
            data_dict['x'] = positions
            data_dict['y'] = y_col
            source = bokeh.models.ColumnDataSource(data=data_dict)
            sources[name][col_name] = source
            p = bokeh.plotting.figure(tools=[hover, "crosshair", "pan", "reset", "save", "wheel_zoom"], title=name)
            # p.xaxis.axis_label_text_font_size = "30pt" not working
            p.xaxis.ticker = scale
            p.xaxis.major_label_overrides = label_dict
            p.xaxis.major_label_orientation = "vertical"
            p.yaxis.axis_label = col_name
            if "symmetric_diff" in col_name or "distance" in col_name:
                p.y_range.flipped = True
            y_lims = chose_ylims(y_col, invert=p.y_range.flipped)
            p.y_range = bokeh.models.Range1d(*y_lims)
            p.scatter('x', 'y', size=10, source=source,
                     fill_color=bokeh.transform.transform('colour', mapper),
                     fill_alpha='alpha', line_alpha='alpha',
                     marker=markers, legend_group=marker_key)
            #p.legend.click_policy = "hide"
            #p.legend.location = "bottom_left"
            p.legend.visible = False
            plots[-1].append(p)
        # add a colour bar to the last plot only
        colour_bar = bokeh.models.ColorBar(color_mapper=mapper, location=(0, 0),
                                           title=c_name)
        p.add_layout(colour_bar, 'right')
        # now make the selector features
        if catigorical:
            def update(attrname, old, new,
                       name=name, labels=labels, label_catigories=label_catigories):
                """
                

                Parameters
                ----------
                attrname :
                    
                old :
                    
                new :
                    
                name :
                     (Default value = name)
                labels :
                     (Default value = labels)
                label_catigories :
                     (Default value = label_catigories)

                Returns
                -------

                """
                keep_catigories = [label_catigories[label] for label in np.array(labels)[new]]
                original_values = array[:, records.indices[name]]
                new_mask = np.fromiter((value in keep_catigories for value in original_values),
                                       dtype=bool)
                mask_dict[name] = new_mask
                mask = np.vstack(tuple(mask_dict.values()))
                mask = np.all(np.vstack(tuple(mask_dict.values())), axis=0)
                new_dict = {name: orignal_data_dict[name][mask] for name in orignal_data_dict}
                for change_name in parameter_order:
                    for col_name, y_col in y_cols.items():
                        new_dict['x'] = original_positions[change_name][mask]
                        new_dict['y'] = y_col[mask]
                        sources[change_name][col_name].data = new_dict
            boxes.on_change('active', update)
            plots[-1].append(boxes)
        elif has_slider:
            def update(attrname, old, new, name=name, top=top, bottom=bottom):
                """
                

                Parameters
                ----------
                attrname :
                    
                old :
                    
                new :
                    
                name :
                     (Default value = name)
                top :
                     (Default value = top)
                bottom :
                     (Default value = bottom)

                Returns
                -------

                """
                original_values = array[:, records.indices[name]]
                epsilon = (top-bottom)/50
                new_mask = np.full_like(original_values, True, dtype=bool)
                if new[0] + epsilon > bottom:
                    new_mask = np.logical_and(new_mask, original_values>new[0])
                if new[1] - epsilon < top:
                    new_mask = np.logical_and(new_mask, original_values<new[1])
                mask_dict[name] = new_mask
                mask = np.vstack(tuple(mask_dict.values()))
                mask = np.all(np.vstack(tuple(mask_dict.values())), axis=0)
                new_dict = {name: orignal_data_dict[name][mask] for name in orignal_data_dict}
                for change_name in parameter_order:
                    for col_name, y_col in y_cols.items():
                        new_dict['x'] = original_positions[change_name][mask]
                        new_dict['y'] = y_col[mask]
                        sources[change_name][col_name].data = new_dict
            slider.on_change('value', update)
            plots[-1].append(slider)
    all_p = bokeh.layouts.gridplot(plots, plot_width=400, plot_height=400)
    return all_p


def make_float_scale(col_content):
    """
    

    Parameters
    ----------
    col_content :
        

    Returns
    -------

    """
    catigories = set(col_content)
    #print(f"Unordered catigories {catigories}")
    has_none = None in catigories
    # None prevents propper sorting
    catigories.discard(None)
    catigories = sorted(catigories)
    min_val, max_val = catigories[0], catigories[-1]
    if has_none:
        catigories += [None]
    #print(f"Ordered catigories {catigories}")
    scale = np.array(catigories)
    # the last one is skipped as it is the None value
    try:
        real = np.fromiter((x for x in catigories[:-1] if np.isfinite(x)),
                           dtype=float)
    except:
        print(catigories)
        real = np.fromiter((x for x in catigories[:-1] if np.isfinite(x)),
                           dtype=float)
    if len(real) > 1:
        ave_gap = np.mean(real[1:] - real[:-1])
    else:
        ave_gap = 1.
    scale[scale == -np.inf] = np.min(real) - ave_gap
    scale[scale == np.inf] = np.max(real) + ave_gap
    scale[scale == None] = np.min(real) - 2*ave_gap
    scale = scale.astype(float)
    positions = np.fromiter((scale[catigories.index(x)] for x in col_content),
                            dtype=float)
    # now check if the scale is too tight and if so drop a few values
    if len(scale) > 2:
        length = np.max(scale) - np.min(scale)
        min_gap = length/30
        new_scale = [scale[0]]
        new_catigories = [catigories[0]]
        last_point = scale[0]
        for s, c in zip(scale[1:], catigories[1:]):
            if s - new_scale[-1] > min_gap:
                new_scale.append(s)
                new_catigories.append(c)
        scale = np.array(new_scale)
        catigories = new_catigories
    # trim the labels
    label_dict = {tick: str(cat)[:5] for cat, tick in zip(catigories, scale)}
    return scale, positions, label_dict, min_val, max_val


def make_ordinal_scale(col_content):
    """
    

    Parameters
    ----------
    col_content :
        

    Returns
    -------

    """
    # x positions of each point
    positions = np.empty(len(col_content), dtype=int)
    # dictionary of name : example value
    label_catigories = {}
    # dictionary of scale position : name
    label_dict = {}
    # occasionaly equality comparison on the col_content is not possible
    # make a translation to names
    content_names = []
    for con in col_content:
        if isinstance(con, tuple):
            name = ', '.join((con[0], str(con[1])[:1]))
        else:
            name = str(con)
        content_names.append(name)
    # None prevents propper sorting
    catigories = [(name, col_content[content_names.index(name)]) for name in set(content_names)
                  if name != str(None)]
    catigories = sorted(catigories, key=lambda x: x[1])
    if str(None) in content_names:
        catigories.append((str(None), None))
    # set of possible x positions
    scale = list(range(len(catigories)))
    content_names = np.array(content_names)  # need to use this as a mask
    for pos, (name, cat) in enumerate(catigories):
        label_catigories[name] = cat
        label_dict[pos] = name
        positions[content_names == name] = pos
    return scale, positions, label_dict, label_catigories


def chose_ylims(y_data, invert=False):
    """
    

    Parameters
    ----------
    y_data :
        param invert:  (Default value = False)
    invert :
        (Default value = False)

    Returns
    -------

    
    """
    std = np.std(y_data)
    if invert:
        top = np.min(y_data) - 0.3*std
        bottom = np.min((np.max(y_data), np.mean(y_data) + 1.*std))
    else:
        top = np.max(y_data) + 0.3*std
        bottom = np.max((np.min(y_data), np.mean(y_data) - 1.*std))
    return bottom, top

# code for sorting finished clusters
def group_discreet(records, only_scored=True):
    """
    group the records by their discreet parameters

    Parameters
    ----------
    records :
        param only_scored:  (Default value = True)
    only_scored :
        (Default value = True)

    Returns
    -------

    
    """
    array = records.typed_array()
    if only_scored:
        array = array[records.scored]
    pure_numeric = ['DeltaR', 'ExponentMultiplier']
    numeric_tuples = ['AffinityCutoff']
    numeric_tuples = [(name, records.indices[name]) for name in numeric_tuples]
    not_discreet = pure_numeric + numeric_tuples + list(records.evaluation_columns) + ['jet_id']
    discreet_attributes = [(name, records.indices[name]) for name in 
                           sorted(set(records.indices.keys()) - set(not_discreet))]
    groups = {}
    for row in array:
        tuple_start = [f"{name}_None" if row[idx] is None else f"{name}_{row[idx][0]}"
                       for name, idx in numeric_tuples]
        discreet = [f"{name}_{row[idx]}" for name, idx in discreet_attributes]
        key = ','.join(discreet + tuple_start)
        groups.setdefault(key, []).append(row.tolist())
    return groups


def thin_clusters(records, min_mean_jets=0.5, proximity=0.10):
    """
    Identify some clusters to remove in order to save space

    Parameters
    ----------
    records :
        param min_mean_jets:  (Default value = 0.5)
    proximity :
        Default value = 0.10)
    min_mean_jets :
        (Default value = 0.5)

    Returns
    -------

    
    """
    removed_name = "removed_records.csv"
    removed_records = Records(removed_name)
    # first remove anything that fails a hard cut
    array = records.typed_array()
    mask = array[records.scored, records.indices['njets']] < min_mean_jets
    to_remove = array[records.scored][mask, 0]
    print(f"Removing {len(to_remove)} based on mean jets")
    remove_clusters(records, to_remove)
    print("clusteres removed from awkd files")
    removed_records.transfer(records, to_remove)
    print("cluster records moved to removed_records.csv")
    # now sift through the records looking for things that
    # share all discreet attributes and 
    # are closer than proximity
    pure_numeric_indices = [records.indices[name] for name in ('DeltaR', 'ExponentMultiplier')]
    numeric_tuples_indices = [records.indices['AffinityCutoff']]
    groups = group_discreet(records)
    to_remove = []
    # within each group
    for key in groups:
        group = np.array(groups[key])
        if len(group) < 3:
            continue
        # be sure to preseve the one with the best score
        cumulative = cumulative_score(records, group)
        best_idx = np.argmax(cumulative)
        # now filter on numeric values
        for col in pure_numeric_indices:
            min_step = proximity * (np.max(group[:, col]) - np.min(group[:, col]))
            order = np.argsort(group[:, col])
            best_idx = np.where(order == best_idx)[0][0]
            group = group[order]
            # start by working up from best_idx
            i = best_idx + 1
            while i < len(group):
                if group[i, col] - group[i-1, col] < min_step:
                    to_remove.append(group[i, 0])
                    group = np.delete(group, i, 0)
                i += 1
            # now go down, adjusting the position of the best idx everytime something is deleted
            i = best_idx - 1
            while i >= 0:
                if group[i+1, col] - group[i, col] < min_step:
                    to_remove.append(group[i, 0])
                    group = np.delete(group, i, 0)
                    best_idx -= 1
                i -= 1
        # now filter on numeric tuples
        for col in numeric_tuples_indices:
            values = np.fromiter((np.nan if item is None else item[1] for item in group[:, col]), dtype=float)
            found_none = False
            min_step = proximity * (np.max(values) - np.min(values))
            order = np.argsort(values)
            best_idx = np.where(order == best_idx)[0][0]
            group = group[order]
            values = values[order].tolist()
            # start by working up from best_idx
            i = best_idx + 1
            while i < len(group):
                if np.nan in [values[i], values[i-1]]:
                    if found_none:
                        to_remove.append(group[i, 0])
                        group = np.delete(group, i, 0)
                        del values[i]
                    else:
                        found_none = True
                        continue
                if values[i] - values[i-1] < min_step:
                    to_remove.append(group[i, 0])
                    group = np.delete(group, i, 0)
                    del values[i]
                i += 1
            # now go down, adjusting the position of the best idx everytime something is deleted
            i = best_idx - 1
            while i >= 0:
                if np.nan in [values[i], values[i-1]]:
                    if found_none:
                        to_remove.append(group[i, 0])
                        group = np.delete(group, i, 0)
                        del values[i]
                    else:
                        found_none = True
                        continue
                if values[i+1] - values[i] < min_step:
                    to_remove.append(group[i, 0])
                    group = np.delete(group, i, 0)
                    del values[i]
                    best_idx -= 1
                i -= 1
    print(f"Removing {len(to_remove)} based on proximity")
    remove_clusters(records, to_remove)
    print("clusteres removed from awkd files")
    removed_records.transfer(records, to_remove)
    print("cluster records moved to removed_records.csv")


def remove_clusters(records, jet_ids):
    """
    

    Parameters
    ----------
    records :
        param jet_ids:
    jet_ids :
        

    Returns
    -------

    
    """
    # probably need to deal with one eventwise at a time
    # and not seek too many jets int he first place
    batch_size = 50
    by_eventWise = {}
    for i in range(0, len(jet_ids), batch_size):
        batch_ids = jet_ids[i: i + batch_size]
        for eventWise, jet_name in seek_clusters(records, batch_ids):
            path = os.path.join(eventWise.dir_name, eventWise.save_name)
            by_eventWise.setdefault(path, []).append(jet_name)
    for path in by_eventWise:
        print(f"Removing {len(by_eventWise[path])} jets from {path}")
        eventWise = Components.EventWise.from_file(path)
        for jet_name in by_eventWise[path]:
            eventWise.remove_prefix(jet_name)
        eventWise.write()


def consolidate_clusters(length=2000, dir_name="megaIgnore", max_size=300):
    """
    Sort clusters by hyperparameter groups and
    put them into new eventWise awkd files

    Parameters
    ----------
    length :
        Default value = 2000)
    dir_name :
        Default value = "megaIgnore")
    max_size :
        Default value = 300)

    Returns
    -------

    
    """
    # start by learning what's in the directory
    records_name = "tmp{}_records.csv"
    i=0
    while os.path.exists(records_name.format(i)):
        i += 1
    records_name = records_name.format(i)
    records = Records(records_name)
    pottential_names = [os.path.join(dir_name, name) for name in os.listdir(dir_name)
                        if name.endswith(".awkd")]
    ew_names = []
    for ew_name in pottential_names:
        try:
            ew = Components.EventWise.from_file(ew_name)
            ew_names.append(ew_name)
        except Exception:
            print(f"Couldn't read {ew_name}")
            continue
        print(ew_name)
        ew.selected_index = None
        if len(ew.JetInputs_Energy) == length:
            records.scan(ew)
        del ew
    # now all jets in the sample should be located
    groups = group_discreet(records, only_scored=False)
    finalised = []
    priorties = [(name, records.indices[name]) for name in 
                 ["jet_class", "Laplacien", "WithLaplacienScaling", "AffinityType"]]
    for name, col in priorties:
        if not groups:
            break  # everything is finalised
        gathered = {}  # new gathering baskets
        for key, group in groups.items():
            try:
                assignment = str(group[0][col])
            except IndexError:
                group
                col
            if assignment in gathered and len(gathered[assignment]) + len(group) > max_size:
                finalised.append(gathered.pop(assignment))
                gathered[assignment] = group
            elif assignment in gathered:
                gathered[assignment] += group
            else:
                gathered[assignment] = group
        # now if there are still things left in gathered assign them back to groups
        if gathered:
            groups = gathered
        else:
            break
    # if we end the loop with things in groups move them to finalised
    finalised += list(groups.values())
    print(f"Found {len(finalised)} groups")
    # make a directory for the new eventwise objects
    new_dir = os.path.join(dir_name, "consolidated")
    os.mkdir(new_dir)
    consolodated_names = set()
    new_names = []
    # walk over the groups saving them
    for group in finalised:
        group = np.array(group)
        consistant_components = [str(group[0, int(idx)]).capitalize() for name, idx in priorties
                                 if len(set(group[:, int(idx)])) == 1]
        save_name_form = f"Group{{}}{''.join(consistant_components)}_{int(length/1000)}k.awkd"
        save_name = save_name_form.format('')
        i=1
        while save_name in new_names:
            save_name = save_name_form.format(i)
            i += 1
        new_names.append(save_name)
        print(f"Writing group {save_name}, length {len(group)}")
        jet_ids = group[:, records.indices['jet_id']]
        cluster = seek_clusters(records, [jet_ids[0]])
        print(f"Adding non-jet colummns")
        first_eventWise = cluster[0][0]
        jet_names = FormJets.jet_names(first_eventWise)
        non_jet_colunms = [name for name in first_eventWise.columns if
                           not any(name.startswith(jname) for jname in jet_names)]
        non_jet_hcolunms = [name for name in first_eventWise.hyperparameter_columns if
                           not any(name.startswith(jname) for jname in jet_names)]
        non_jet_content = {name: getattr(first_eventWise, name) for name in non_jet_colunms + non_jet_hcolunms}
        new_eventWise = Components.EventWise(new_dir, save_name, contents=non_jet_content,
                                             columns=non_jet_colunms, hyperparameter_columns=non_jet_hcolunms)
        del cluster  # get rid of the ew to save memory
        print("Adding jet content")
        # probably need to deal with one eventwise at a time
        # and not seek too many jets int he first place
        batch_size = 10
        by_eventWise = {}
        for i in range(0, len(jet_ids), batch_size):
            batch_ids = jet_ids[i: i + batch_size]
            for eventWise, jet_name in seek_clusters(records, batch_ids):
                path = os.path.join(eventWise.dir_name, eventWise.save_name)
                by_eventWise.setdefault(path, []).append(jet_name)
        print("Located jet content")
        for path in by_eventWise:
            print(f"Copying {len(by_eventWise[path])} jets from {path}", end='')
            eventWise = Components.EventWise.from_file(path)
            # if there are a lot of jets here break this into chunks
            for i in range(0, len(by_eventWise[path]), batch_size):
                for jet_name in by_eventWise[path][i:i+batch_size]:
                    consolodated_names.add(eventWise.save_name)
                    jet_content = {name: getattr(eventWise, name) for name in eventWise.columns if name.startswith(jet_name)}
                    new_eventWise.append(**jet_content)
                    jet_hcontent = {name: getattr(eventWise, name) for name in eventWise.hyperparameter_columns if name.startswith(jet_name)}
                    new_eventWise.append_hyperparameters(**jet_hcontent)
        print("Added all content, saving...")
        new_eventWise.write()
        print("Saved. removing from ram")
        del new_eventWise
    with open(os.path.join(new_dir, "included.txt"), 'w') as included:
        included.write(', '.join(consolodated_names))
            

if __name__ == '__main__':
    records_name = InputTools.get_file_name("Records file? (new or existing) ")
    records = Records(records_name)
    #for i in range(1, 10):
    #    ew_name = f"megaIgnore/MC{i}.awkd"
    #    print(ew_name)
    #    records.score(ew_name.strip())
    
    ew_name = InputTools.get_file_name("EventWise file? (existing) ")
    records.score(ew_name.strip())

# serve this with
# bokeh serve --show tree_tagger/CompareClusters.py
if 'bk_script' in __name__:
    records_name = InputTools.get_file_name("Records file? (new or existing) ")
    records = Records(records_name)
    plots = parameter_comparison(records, cuts=True)
    bokeh.plotting.curdoc().add_root(plots)

