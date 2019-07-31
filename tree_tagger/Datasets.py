import h5py
import random
from torch.utils.data import Dataset
from torch import nn
from ipdb import set_trace as st
import numpy as np
from tree_tagger import Components

class TracksTowersDataset(Dataset):
    def __init__(self, folder_name):
        observables = Components.Observables.from_file(folder_name)
        tracks_list = observables.tracks_list
        all_events = []
        for tracks_list, towers_list in zip(tracks_list_per_event, towers_list_per_event):
            event_info = gen_overall_data(tracks_list, towers_list)
            track_data = gen_track_data(tracks_list, event_info)
            tower_data = gen_tower_data(towers_list, event_info)
            all_events.append([track_data, tower_data])
        total_events = len(all_events)
        # shuffle the events
        random.shuffle(all_events)
        # then take the first section as test data
        test_percent = 0.2
        num_test = int(total_events*test_percent)
        self._test_events = all_events[:num_test]
        self._events = all_events[num_test:]
        self._len = total_events - num_test
        

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        """ return a single event """
        return self._events[idx]

    @property
    def test_events(self):
        return self._test_events

def gen_overall_data(tracks_list, towers_list):
    info = []
    track_energies = [t.e for t in tracks_list]
    info.append(np.mean(track_energies))
    info.append(np.std(track_energies))
    tower_energies = [t.e for t in towers_list]
    info.append(np.mean(tower_energies))
    info.append(np.std(tower_energies))
    track_etas = [t.eta for t in tracks_list]
    info.append(np.mean(track_etas))
    info.append(np.std(track_etas))
    tower_etas = [t.eta for t in towers_list]
    info.append(np.mean(tower_etas))
    info.append(np.std(tower_etas))
    track_pts = [t.pt for t in tracks_list]
    info.append(np.mean(track_pts))
    info.append(np.std(track_pts))
    tower_pts = [t.pt for t in towers_list]
    info.append(np.mean(tower_pts))
    info.append(np.std(tower_pts))
    track_phis = [t.phi for t in tracks_list]
    info.append(np.std(track_phis))
    tower_phis = [t.phi for t in towers_list]
    info.append(np.std(tower_phis))
    return info
    

def gen_track_data(tracks_list, event_info):
    data = np.empty((len(tracks_list), len(event_info) + 21))
    for i, track in enumerate(tracks_list):
        data[i] = [track._p,
                   track._pT,
                   track._eta,
                   track._phi,
                   track.ctgTheta,
                   track.etaOuter,
                   track.phiOuter,
                   track._t,
                   track._x,
                   track._y,
                   track._z,
                   track.xd,
                   track.yd,
                   track.zd,
                   track.l,
                   track.d0,
                   track.dZ,
                   track.x,
                   track.y,
                   track.z,
                   track.t] + event_info
    return data
    

def gen_tower_data(towers_list, event_info):
    data = np.empty((len(towers_list), len(event_info) + 12))
    for i, tower in enumerate(towers_list):
        data[i] = [tower.nTimeHits,
                   tower.eem,
                   tower.ehad,
                   tower.edges[0],
                   tower.edges[1],
                   tower.edges[2],
                   tower.edges[3],
                   tower._t,
                   tower.et,
                   tower.eta,
                   tower.phi,
                   tower.e] + event_info
    return data


