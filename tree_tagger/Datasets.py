import os
import csv
import random
from torch.utils.data import Dataset
from torch import nn
from ipdb import set_trace as st
import numpy as np
from tree_tagger import Components, ReadSQL, ReadHepmc, LinkingFramework
import torch

TEST_DS="/home/henry/lazy/h1bBatch2.db"
TEST_HEPMC="/home/henry/lazy/h1bBatch2.hepmc"

class TracksTowersDataset(Dataset):
    def __init__(self, folder_name=None, database_name=TEST_DS, hepmc_name=TEST_HEPMC, num_events=100, shuffle=True, all_events=None):
        torch.set_default_tensor_type('torch.DoubleTensor')
        # read in from file
        if folder_name is not None:
            self.folder_name = folder_name
        # check for preprocessed events
        if all_events is not None:  # given as an argument
            # check the number of events and all event match up
            if num_events != len(all_events):
                assert num_events == 100, f"num_events={num_events} (not the default) but the length of events={len(all_events)}"
                num_events = len(all_events)
        elif folder_name is not None:  # could be in the folder
            # try reading a dataset from the folder
            try:
                num_events, all_events = self._read_file(folder_name)
                assert num_events == len(all_events)
                self.num_events = num_events
            except FileNotFoundError:
                pass  # it's fine the next method will sorth things out
            assert isinstance(all_events, (list, np.ndarray))
        if all_events is None:  # no preprocessed events
            print("No existing dataset found, reading new.")
            all_events = []
            tracks_list_per_event, towers_list_per_event = [], []
            start_event = 0
            hepmc_events = ReadHepmc.read_file(hepmc_name, start_event, start_event+num_events)
            for event_n, event in enumerate(hepmc_events):
                track_list, tower_list = ReadSQL.read_tracks_towers(event, database_name, event_n)
                tracks_list_per_event.append(track_list)
                towers_list_per_event.append(tower_list)
            # process the events
            for tracks_list, towers_list in zip(tracks_list_per_event, towers_list_per_event):
                tracks_near_tower, towers_near_track = LinkingFramework.tower_track_proximity(towers_list, tracks_list)
                MCtruth = LinkingFramework.MC_truth_links(towers_list, tracks_list)
                event_info = gen_overall_data(tracks_list, towers_list)
                track_data = gen_track_data(tracks_list, event_info)
                track_data = torch.from_numpy(track_data)
                tower_data = gen_tower_data(towers_list, event_info)
                tower_data = torch.from_numpy(tower_data)
                all_events.append([tower_data, track_data, towers_near_track, MCtruth])
        # convert to array if needed
        if isinstance(all_events, list):
            all_events = np.array(all_events)
        total_events = len(all_events)
        if shuffle:
            # shuffle the events
            random.shuffle(all_events)
        # then take the first section as test data
        test_percent = 0.2
        self.num_test = int(total_events*test_percent)
        self._test_events = all_events[:self.num_test]
        self._events = all_events[self.num_test:]
        self._len = total_events - self.num_test
        

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        """ return events in numpy style """
        return self._events[idx]

    @property
    def test_events(self):
        return self._test_events

    @property
    def track_dimensions(self):
        _, tracks, _, _ = self[0]
        return tracks.shape[1]

    @property
    def tower_dimensions(self):
        towers, _,  _, _ = self[0]
        return towers.shape[1]

    # define the file names
    numpy_save_name = "input_arrays.npz"
    proximities_save_name = "input_proximities.csv"

    def write(self, folder_name=None):
        if folder_name is None:
            folder_name = self.folder_name
        all_events = np.vstack((self._test_events, self._events))
        # list the hieght of all the arrays
        cumulative_tower_entries = np.empty(len(all_events))
        cumulative_track_entries = np.empty(len(all_events))
        tower_reached = 0
        track_reached = 0
        all_towers = []
        all_tracks = []
        all_prox = []
        all_truth = []
        for i, (towers, tracks, prox, MCtruth) in enumerate(all_events):
            tower_reached += len(towers)
            cumulative_tower_entries[i] = tower_reached
            track_reached += len(tracks)
            cumulative_track_entries[i] = track_reached
            all_towers.append(towers)
            all_tracks.append(tracks)
            all_prox += prox
            all_truth += list(MCtruth.items())
        all_tracks = np.vstack(all_tracks)
        all_towers = np.vstack(all_towers)
        # save all the numpy arrays into a npz format
        numpy_save_name = os.path.join(folder_name, self.numpy_save_name)
        np.savez(numpy_save_name, cumulative_tower_entries=cumulative_tower_entries,
                 cumulative_track_entries=cumulative_track_entries, all_towers=all_towers,
                 all_tracks=all_tracks, all_MC_truth=np.array(all_truth))
        # prox has variable numbers of items in each row,
        # so save with csv_writer
        all_prox_name = os.path.join(folder_name, self.proximities_save_name)
        with open(all_prox_name, 'w') as prox_file:
            writer = csv.writer(prox_file)
            for track_prox in all_prox:
                string_row = [str(x) for x in track_prox]
                writer.writerow(string_row)

    @classmethod
    def from_file(cls, folder_name):
        num_events, events = cls._read_file(folder_name)
        return cls(folder_name=folder_name, num_events=num_events,
                   shuffle=False,  # so that the test events remain consistant
                   all_events=events)

    @classmethod
    def _read_file(cls, folder_name):
        # read all the numpy arrays into a npz format
        numpy_save_name = os.path.join(folder_name, cls.numpy_save_name)
        array_dict = np.load(numpy_save_name)
        cumulative_tower_entries = array_dict["cumulative_tower_entries"].astype(int)
        cumulative_track_entries = array_dict["cumulative_track_entries"].astype(int)
        all_towers = array_dict["all_towers"]
        assert len(all_towers) == cumulative_tower_entries[-1]
        all_tracks = array_dict["all_tracks"]
        assert len(all_tracks) == cumulative_track_entries[-1]
        all_MC_truth = array_dict["all_MC_truth"]
        assert len(all_MC_truth) == cumulative_track_entries[-1]
        # split them by event
        towers_by_event = np.split(all_towers, cumulative_tower_entries[:-1])
        tracks_by_event = np.split(all_tracks, cumulative_track_entries[:-1])
        MC_truth_by_event = [{int(a):None if b is None else int(b)
                              for a, b in event_truth}
                             for event_truth in 
                             np.split(all_MC_truth, cumulative_track_entries[:-1])]
        # prox has variable numbers of items in each row,
        # so it is stored as a csv
        all_prox_name = os.path.join(folder_name, cls.proximities_save_name)
        prox_by_event = [[]]
        with open(all_prox_name, 'r') as prox_file:
            reader = csv.reader(prox_file)
            event_num = 0
            for i, row in enumerate(reader):
                if i >= cumulative_track_entries[event_num]:
                    prox_by_event.append([])
                    event_num += 1
                prox_by_event[-1].append([int(x) for x in row])
        num_events = len(cumulative_tower_entries)
        assert len(prox_by_event) == num_events
        events = []
        for event_n in range(num_events):
            tower_data = torch.from_numpy(towers_by_event[event_n])
            track_data = torch.from_numpy(tracks_by_event[event_n])
            events.append([tower_data,
                           track_data,
                           prox_by_event[event_n],
                           MC_truth_by_event[event_n]])
        return num_events, events


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
    track_phis = [t.phi() for t in tracks_list]
    info.append(np.std(track_phis))
    tower_phis = [t.phi() for t in towers_list]
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
    data = np.empty((len(towers_list), len(event_info) + 11))
    for i, tower in enumerate(towers_list):
        data[i] = [# tower.nTimeHits,  # appears to be blank
                   tower.eem,
                   tower.ehad,
                   tower.edges[0],
                   tower.edges[1],
                   tower.edges[2],
                   tower.edges[3],
                   tower._t,
                   tower.et,
                   tower.eta,
                   tower.phi(),
                   tower.e] + event_info
    return data


