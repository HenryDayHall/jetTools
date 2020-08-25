from matplotlib import pyplot as plt
import numpy as np

def discribe_jet(eventWise=None, jet_name=None, properties_dict=None, ax=None, font_size=12, additional_text=None):
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
