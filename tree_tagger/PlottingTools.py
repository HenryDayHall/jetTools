from matplotlib import pyplot as plt
from tree_tagger import FormJets

def discribe_jet(eventWise=None, jet_name=None, properties_dict=None, ax=None, font_size=12, additional_text=None):
    if eventWise is not None:
        eventWise.selected_index = None
    create_ax = ax is None
    if create_ax:
        fig, (ax, ax2) = plt.subplots(1, 2)
    ax.set_frame_on(False)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    # write out the jet properties
    text = ''
    if eventWise is not None:
        properties_dict = FormJets.get_jet_params(eventWise, jet_name, True)
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
        
