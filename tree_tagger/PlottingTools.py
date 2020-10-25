from matplotlib import pyplot as plt
from ipdb import set_trace as st
import numpy as np
import os

def discribe_jet(eventWise=None, jet_name=None, properties_dict=None, ax=None, font_size=12, additional_text=None, skip_scores=True):
    """
    

    Parameters
    ----------
    eventWise :
         (Default value = None)
    jet_name :
         (Default value = None)
    properties_dict :
         (Default value = None)
    ax :
         (Default value = None)
    font_size :
         (Default value = 12)
    additional_text :
         (Default value = None)

    Returns
    -------

    """
    if eventWise is not None:
        eventWise.selected_index = None
    create_ax = ax is None
    if create_ax:
        fig, (ax, ax2) = plt.subplots(1, 2)
    hide_axis(ax)
    # write out the jet properties
    text = ''
    if eventWise is not None:
        prefix = jet_name + "_"
        trim = len(prefix)
        properties_dict = {name[trim:]: getattr(eventWise, name) for name in eventWise.hyperparameter_columns
                           if name.startswith(prefix)}
        if skip_scores:
            score_starts = ["Ave", "Seperate", "Quality"]
            properties_dict = {name: value for name, value in properties_dict.items()
                               if not np.any([name.startswith(s) for s in score_starts])}
        text += f"In file {eventWise.save_name}\n"
    elif properties_dict is not None:
        jet_name = properties_dict.get('jet_name', "Unnamed")
    else:
        raise ValueError("Must give eventWise or a properties_dict")
    text += f"Jet properties for {jet_name}\n"
    if eventWise is not None:
        n_events = len(getattr(eventWise, jet_name + "_PT"))
        text += f"# events; {n_events}\n"
    for name, value in properties_dict.items():
        if name in ["RescaleEnergy", "jet_name"]:
            continue
        text += f"{name}; {value}\n"
    if additional_text is not None:
        text += additional_text
    ax.text(0, 0, text, fontsize=font_size)
    if create_ax:
        return fig, ax2
        

def find_crossing_point(x_start, y_start, x_end, y_end, y_max=np.pi, y_min=-np.pi):
    """
    

    Parameters
    ----------
    x_start :
        
    y_start :
        
    x_end :
        
    y_end :
        
    y_max :
         (Default value = np.pi)
    y_min :
         (Default value = -np.pi)

    Returns
    -------

    """
    y_range = y_max - y_min
    if 2*abs(y_end - y_start) < y_range:
        return None, None
    # work out the x coord of the axis cross
    # leg/body closes to np.pi
    sign = 1 - 2*(y_end > y_start)
    if sign < 0:
        top_y, bottom_y, top_x, bottom_x = y_start, y_end, x_start, x_end
    else:
        top_y, bottom_y, top_x, bottom_x = y_end, y_start, x_end, x_start
    #                   | y_to_top/ range - (top y - bottom y) |
    percent_to_top = np.abs((y_max - top_y)/(y_range + bottom_y - top_y))
    #          top x     +  x seperation * percent to top
    x_cross = top_x + (bottom_x - top_x)*percent_to_top
    return x_cross, sign


def hide_axis(ax):
    ax.set_frame_on(False)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)


def text_table(ax, content, cell_fmt=None):
    table_sep = '|'
    table = []
    if cell_fmt is None:
        cell_fmt = "17.17"
    for row in content:
        table.append([])
        for x in row:
            try:
                table[-1].append(f"{x:{cell_fmt}}")
            except ValueError:
                table[-1].append(f"{x:{cell_fmt[0]}}")
            except TypeError:
                table[-1].append(f"{str(x):{cell_fmt}}")
        table[-1] = table_sep.join(table[-1])
    table = os.linesep.join(table)
    ax.text(0, 0, table, fontdict={"family": 'monospace'})


def make_inputs_table(eventWise, jet_names, table_ax, jet_inputs=None):
    if jet_inputs is None:
        jet_inputs = ["PhyDistance", "AffinityType", "AffinityCutoff",
                      "Laplacien", "ExpOfPTMultiplier"]
    # construct a text table
    table_content = [[" "] + jet_inputs]
    table_content += [[name] + [getattr(eventWise, name+'_'+inp,  "not found") for inp in jet_inputs]
                      for name in jet_names]
    text_table(table_ax, table_content)
    hide_axis(table_ax)
    return table_content, jet_inputs


def label_scatter(x, y, labels, s=None, c=None, **kwargs):
    if c is None:
        c = [[0., 0., 0.]]*len(x)
    # make an array to hold the y pos of the text labels
    text_y = np.copy(y)
    # going to scan in the x coordinate moving oyjects that are within 20%
    # of the x range appart in the y range
    scan_order = np.argsort(x)
    x_width = x[scan_order[-1]] - x[scan_order[0]]
    x_spacing = 0.2*x_width
    # chose a y range in which we will consider proximity
    y_width = np.max(y) - np.min(y)
    y_proximity = 0.05*y_width
    # keep track of points in the scan window
    current_y = []
    current_x = []
    for next_idx in scan_order:
        next_x = x[next_idx]
        next_y = y[next_idx]
        window_start = next_x - x_spacing
        while current_x and current_x[0] < window_start:
            del current_x[0]
            del current_y[0]
        above = ceiling = next_y + y_proximity
        below = floor = next_y - y_proximity
        for c_y in current_y:
            if c_y >= next_y and c_y < ceiling:
                ceiling = c_y
            elif c_y < next_y and c_y > floor:
                floor = c_y
        # now relocate the y coord
        next_y = 0.5 * (ceiling + floor)
        text_y[next_idx] = next_y
        current_y.append(next_y)
        current_x.append(next_x)
    ax = kwargs.get('ax', plt.gca())
    for t_x, t_y, t_c, label in zip(x, text_y, c, labels):
        ax.text(t_x, t_y, label, color=t_c)
    if kwargs.get('add_points', False):
        ax.scatter(x, y, s=s, c=c, **kwargs)




