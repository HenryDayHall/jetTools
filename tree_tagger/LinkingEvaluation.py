""" evaluate the decisions of the linking NN """
from ipdb import set_trace as st
import numpy as np
from matplotlib import pyplot as plt
from tree_tagger import LinkingFramework, LinkingNN, Constants

def apply_linking_net(run, use_test=True):
    tower_net, track_net = run.best_nets
    dataset = run.dataset
    if use_test:
        events = dataset.test_events
    else:
        events = dataset
    output_events = []
    for event_data in events:
        towers_data, tracks_data, proximities, MC_truth = event_data
        towers_projection = np.empty((len(towers_data), run.settings['latent_dimension']))
        tracks_projection = np.empty((len(tracks_data), run.settings['latent_dimension']))
        for i, tower_d in enumerate(towers_data):
            towers_projection[i] = tower_net(tower_d).detach().numpy()
        for i, track_d in enumerate(tracks_data):
            tracks_projection[i] = track_net(track_d).detach().numpy()
        output_events.append([towers_projection, tracks_projection, proximities, MC_truth])
    return output_events

def get_distance_to_neighbor(output_events):
    match_status = []  # was it correctly matched?
    match_distance = []  # how far to the closest match?
    true_distance = []  # how far to what it should have matched to?
    for event in output_events:
        towers_projection, tracks_projection, proximities, MC_truth = event
        nearest_neighbor = LinkingFramework.high_dim_proximity(tracks_projection, towers_projection)
        assert len(nearest_neighbor) == len(tracks_projection)
        # get the match distance for eveything
        matched_towers = towers_projection[nearest_neighbor]
        this_matched_distance = np.sum(np.sqrt((tracks_projection - matched_towers)**2),
                                       axis=1)
        match_distance += this_matched_distance.tolist()
        # process each track
        for track_n, nearest_tower in enumerate(nearest_neighbor):
            true_tower = MC_truth[track_n]
            if true_tower is None:
                match_status.append(Constants.NO_TOWER)
            elif true_tower == nearest_tower:
                match_status.append(Constants.CORRECT_MATCH)
                true_distance.append(this_matched_distance[track_n])
            else:
                match_status.append(Constants.INCORRECT_MATCH)
                true_distance.append(np.sum(np.sqrt((tracks_projection[track_n] -
                                                     towers_projection[true_tower])**2)))
    return match_status, match_distance, true_distance


def plot_distances(output_events):
    match_status, match_distance, true_distance = get_distance_to_neighbor(output_events)
    num_true = len(true_distance)
    total = len(match_distance)
    # split the match distance by the match_status
    match_status = np.array(match_status)
    match_distance = np.array(match_distance)
    no_tower_distance = match_distance[match_status==Constants.NO_TOWER]
    num_no = len(no_tower_distance)
    correct_distance = match_distance[match_status==Constants.CORRECT_MATCH]
    num_corr = len(correct_distance)
    incorrect_distance = match_distance[match_status==Constants.INCORRECT_MATCH]
    num_inc = len(incorrect_distance)

    # plot
    n_bins = 50
    hist_type = 'step'
    true_hist_type = 'bar'
    distance_range = [0, max(np.max(match_distance), max(true_distance))]
    plt.hist(true_distance, n_bins, distance_range, histtype=true_hist_type, color='gray',
             label=f"distance to correct tower ({num_true})")
    plt.hist(correct_distance, n_bins, distance_range, histtype=hist_type, color='green',
             label=f"{num_corr} correct matches") 
    plt.hist(incorrect_distance, n_bins, distance_range, histtype=hist_type, color='red',
             label=f"{num_inc} incorrect matches") 
    plt.hist(no_tower_distance, n_bins, distance_range, histtype=hist_type, color='blue',
             label=f"{num_no} no real match") 

    # label
    plt.legend()
    plt.title("Euclidean seperation between projections of tracks and towers")
    plt.xlabel("Seperation in latent space")
    plt.ylabel(f"Frequency from {total} tracks")
    
    

