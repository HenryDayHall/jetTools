import os
import csv
import random
from torch.utils.data import Dataset
from torch import nn
from ipdb import set_trace as st
import numpy as np
from tree_tagger import Components, ReadSQL, ReadHepmc, LinkingFramework, FormJets, TreeWalker
from sklearn import preprocessing
import torch

TEST_DS="/home/henry/lazy/h1bBatch2.db"
TEST_HEPMC="/home/henry/lazy/h1bBatch2.hepmc"

class TracksTowersDataset(Dataset):
    def __init__(self, folder_name=None, database_name=None, hepmc_name=None, num_events=100, shuffle=True, all_events=None):
        torch.set_default_tensor_type('torch.DoubleTensor')
        self.hepmc_name = hepmc_name
        self.database_name = database_name
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
                assert isinstance(all_events, (list, np.ndarray))
            except FileNotFoundError:
                pass  # it's fine the next method will sorth things out
        if all_events is None:  # no preprocessed events
            if self.hepmc_name is None:
                print(f"No data files given, using {TEST_HEPMC}, {TEST_DS}")
                self.hepmc_name = TEST_HEPMC
                self.database_name = TEST_DS
            print("No existing dataset found, reading new.")
            all_events = self._process_databaseHepmc(num_events)
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

    def _process_databaseHepmc(self, num_events):
        all_events = []
        towers_list_per_event, tracks_list_per_event = [], []
        start_event = 0
        hepmc_events = ReadHepmc.read_file(self.hepmc_name, start_event, start_event+num_events)
        dud_events = []
        for event_n, event in enumerate(hepmc_events):
            try:
                track_list, tower_list = ReadSQL.read_tracks_towers(event, self.database_name, event_n)
                # make the simplifying assumption that there is at least one track and one tower
                assert len(track_list) > 0
                assert len(tower_list) > 0
                tracks_list_per_event.append(track_list)
                towers_list_per_event.append(tower_list)
            except AssertionError:
                dud_events.append(event_n)
        if len(dud_events) > 0:
            print(f"issues in events {dud_events}, removing")
            # sort any dud events and remove them
            for dud in sorted(dud_events, reverse=True):
                del hepmc_events[dud]
        # process the events
        for tracks_list, towers_list in zip(tracks_list_per_event, towers_list_per_event):
            tracks_near_tower, towers_near_track = LinkingFramework.tower_track_proximity(towers_list, tracks_list)
            MCtruth = LinkingFramework.MC_truth_links(towers_list, tracks_list)
            event_info = gen_overall_data(tracks_list, towers_list)
            track_data = gen_track_data(tracks_list, event_info)
            tower_data = gen_tower_data(towers_list, event_info)
            all_events.append([tower_data, track_data, towers_near_track, MCtruth])
        st()
        # now flatten the events to apply normalisation
        (cumulative_tower_entries, cumulative_track_entries,
         all_towers, all_tracks, all_prox, all_truth) = self._per_event_to_flat(all_events)
        rescaled_towers = preprocessing.robust_scale(all_towers, axis=0)
        rescaled_tracks = preprocessing.robust_scale(all_tracks, axis=0)
        # done, repackage them
        _, all_events = self._flat_to_per_event(cumulative_tower_entries, cumulative_track_entries,
                                                rescaled_towers, rescaled_tracks, all_prox, all_truth)
        return all_events


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

    @classmethod
    def _per_event_to_flat(cls, all_events):
        # list the hieght of all the arrays
        cumulative_tower_entries = np.empty(len(all_events), dtype=int)
        cumulative_track_entries = np.empty(len(all_events), dtype=int)
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
            # towers are tracks are going to be torch vectors,
            # this dosn't seem to matter
            # the become numpy on calling vstack
            all_towers.append(towers)
            all_tracks.append(tracks)
            all_prox += prox
            all_truth += list(MCtruth.items())
        all_tracks = np.vstack(all_tracks)
        all_towers = np.vstack(all_towers)
        return (cumulative_tower_entries, cumulative_track_entries,
                all_towers, all_tracks, all_prox, all_truth)

    def write(self, folder_name=None):
        if folder_name is None:
            folder_name = self.folder_name
        all_events = np.vstack((self._test_events, self._events))
        (cumulative_tower_entries, cumulative_track_entries,
         all_towers, all_tracks, all_prox, all_truth) = self._per_event_to_flat(all_events)
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
    def _flat_to_per_event(cls, cumulative_tower_entries, cumulative_track_entries,
                           all_towers, all_tracks, all_prox, all_truth):
        n_events = len(cumulative_tower_entries)
        # split them by event
        towers_by_event = np.split(all_towers, cumulative_tower_entries[:-1])
        tracks_by_event = np.split(all_tracks, cumulative_track_entries[:-1])
        MC_truth_by_event = [{int(a):None if b is None else int(b)
                              for a, b in event_truth}
                             for event_truth in 
                             np.split(all_truth, cumulative_track_entries[:-1])]
        prox_by_event = [[]]
        event_num = 0
        for i, row in enumerate(all_prox):
            if i >= cumulative_track_entries[event_num]:
                prox_by_event.append([])
                event_num += 1
            prox_by_event[-1].append(row)
        assert event_num == n_events - 1
        events = []
        for event_n in range(n_events):
            tower_data = torch.from_numpy(towers_by_event[event_n])
            track_data = torch.from_numpy(tracks_by_event[event_n])
            events.append([tower_data, track_data,
                           prox_by_event[event_n], MC_truth_by_event[event_n]])
        return n_events, events


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
        # prox has variable numbers of items in each row,
        # so it is stored as a csv
        all_prox_name = os.path.join(folder_name, cls.proximities_save_name)
        all_prox = []
        with open(all_prox_name, 'r') as prox_file:
            reader = csv.reader(prox_file)
            for i, row in enumerate(reader):
                all_prox.append([int(x) for x in row])
        # split them by event
        num_events, events = cls._flat_to_per_event(cumulative_tower_entries, cumulative_track_entries,
                                                     all_towers, all_tracks, all_prox, all_MC_truth)
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


# TODO no reason to nly take one pair from each set
def simplifier(dataset, num_pairs=1):
    """ simplify the problem """
    all_events = np.vstack((dataset.test_events, dataset._events))
    reduced_events = []
    for tower_data, track_data, proximites, MC_truth in all_events:
        chosen_tracks = []
        chosen_towers = []
        for track, tower in MC_truth.items():
            if tower is not None:
                assert tower in proximites[track]
                chosen_towers.append(tower)
                chosen_tracks.append(track)
                if len(chosen_tracks) >= num_pairs:
                    break
        new_MC_truth = {chosen_tracks.index(track): chosen_towers.index(MC_truth[track])
                        for track in chosen_tracks}
        new_proximites = []
        for track in chosen_tracks:
            old_prox = proximites[track]
            new_proximites.append([chosen_towers.index(tower)
                                   for tower in old_prox
                                   if tower in chosen_towers])
        new_track_data = track_data[chosen_tracks]
        new_tower_data = tower_data[chosen_towers]
        reduced_events.append([new_tower_data, new_track_data,
                               new_proximites, new_MC_truth])
    new_dataset = TracksTowersDataset("reduced", dataset.num_events,
                                      shuffle=False, all_events=reduced_events)
    return new_dataset


class ParticlesDataset(Dataset):
    def __init__(self, folder_name=None, database_name=None, hepmc_name=None, num_events=100, shuffle=True, all_events=None):
        torch.set_default_tensor_type('torch.DoubleTensor')
        self.hepmc_name = hepmc_name
        self.database_name = database_name
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
                assert isinstance(all_events, (list, np.ndarray))
            except FileNotFoundError:
                pass  # it's fine the next method will sorth things out
        if all_events is None:  # no preprocessed events
            if self.hepmc_name is None:
                print(f"No data files given, using {TEST_HEPMC}, {TEST_DS}")
                self.hepmc_name = TEST_HEPMC
                self.database_name = TEST_DS
            print("No existing dataset found, reading new.")
            all_events = self._process_databaseHepmc(num_events)
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

    def _process_databaseHepmc(self, num_events):
        all_events = []
        towers_list_per_event, tracks_list_per_event = [], []
        start_event = 0
        hepmc_events = ReadHepmc.read_file(self.hepmc_name, start_event, start_event+num_events)
        dud_events = []
        for event_n, event in enumerate(hepmc_events):
            try:
                track_list, tower_list = ReadSQL.read_tracks_towers(event, self.database_name, event_n)
                # make the simplifying assumption that there is at least one track and one tower
                assert len(track_list) > 0
                assert len(tower_list) > 0
                tracks_list_per_event.append(track_list)
                towers_list_per_event.append(tower_list)
            except AssertionError:
                dud_events.append(event_n)
        if len(dud_events) > 0:
            print(f"issues in events {dud_events}, removing")
            # sort any dud events and remove them
            for dud in sorted(dud_events, reverse=True):
                del hepmc_events[dud]
        # process the events
        for tracks_list, towers_list in zip(tracks_list_per_event, towers_list_per_event):
            tracks_near_tower, towers_near_track = LinkingFramework.tower_track_proximity(towers_list, tracks_list)
            MCtruth = LinkingFramework.MC_truth_links(towers_list, tracks_list)
            event_info = gen_overall_data(tracks_list, towers_list)
            track_data = gen_track_data(tracks_list, event_info)
            tower_data = gen_tower_data(towers_list, event_info)
            all_events.append([tower_data, track_data, towers_near_track, MCtruth])
        st()
        # now flatten the events to apply normalisation
        (cumulative_tower_entries, cumulative_track_entries,
         all_towers, all_tracks, all_prox, all_truth) = self._per_event_to_flat(all_events)
        rescaled_towers = preprocessing.robust_scale(all_towers, axis=0)
        rescaled_tracks = preprocessing.robust_scale(all_tracks, axis=0)
        # done, repackage them
        _, all_events = self._flat_to_per_event(cumulative_tower_entries, cumulative_track_entries,
                                                rescaled_towers, rescaled_tracks, all_prox, all_truth)
        return all_events


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

    @classmethod
    def _per_event_to_flat(cls, all_events):
        # list the hieght of all the arrays
        cumulative_tower_entries = np.empty(len(all_events), dtype=int)
        cumulative_track_entries = np.empty(len(all_events), dtype=int)
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
            # towers are tracks are going to be torch vectors,
            # this dosn't seem to matter
            # the become numpy on calling vstack
            all_towers.append(towers)
            all_tracks.append(tracks)
            all_prox += prox
            all_truth += list(MCtruth.items())
        all_tracks = np.vstack(all_tracks)
        all_towers = np.vstack(all_towers)
        return (cumulative_tower_entries, cumulative_track_entries,
                all_towers, all_tracks, all_prox, all_truth)

    def write(self, folder_name=None):
        if folder_name is None:
            folder_name = self.folder_name
        all_events = np.vstack((self._test_events, self._events))
        (cumulative_tower_entries, cumulative_track_entries,
         all_towers, all_tracks, all_prox, all_truth) = self._per_event_to_flat(all_events)
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
    def _flat_to_per_event(cls, cumulative_tower_entries, cumulative_track_entries,
                           all_towers, all_tracks, all_prox, all_truth):
        n_events = len(cumulative_tower_entries)
        # split them by event
        towers_by_event = np.split(all_towers, cumulative_tower_entries[:-1])
        tracks_by_event = np.split(all_tracks, cumulative_track_entries[:-1])
        MC_truth_by_event = [{int(a):None if b is None else int(b)
                              for a, b in event_truth}
                             for event_truth in 
                             np.split(all_truth, cumulative_track_entries[:-1])]
        prox_by_event = [[]]
        event_num = 0
        for i, row in enumerate(all_prox):
            if i >= cumulative_track_entries[event_num]:
                prox_by_event.append([])
                event_num += 1
            prox_by_event[-1].append(row)
        assert event_num == n_events - 1
        events = []
        for event_n in range(n_events):
            tower_data = torch.from_numpy(towers_by_event[event_n])
            track_data = torch.from_numpy(tracks_by_event[event_n])
            events.append([tower_data, track_data,
                           prox_by_event[event_n], MC_truth_by_event[event_n]])
        return n_events, events


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
        # prox has variable numbers of items in each row,
        # so it is stored as a csv
        all_prox_name = os.path.join(folder_name, cls.proximities_save_name)
        all_prox = []
        with open(all_prox_name, 'r') as prox_file:
            reader = csv.reader(prox_file)
            for i, row in enumerate(reader):
                all_prox.append([int(x) for x in row])
        # split them by event
        num_events, events = cls._flat_to_per_event(cumulative_tower_entries, cumulative_track_entries,
                                                     all_towers, all_tracks, all_prox, all_MC_truth)
        return num_events, events



class JetTreesDataset(Dataset):
    def __init__(self, multijet_filename, n_jets=None):
        torch.set_default_tensor_type('torch.DoubleTensor')
        # read in from file
        self.multijet_filename = multijet_filename
        jets_by_event = FormJets.PsudoJets.multi_from_file(multijet_filename)
        jets = []
        for j_event in jets_by_event:
            jets += j_event.split()
        # get truth tags
        data = np.load(multijet_filename)
        self.truth_tags = data['truth_tags']
        assert len(self.truth_tags) == len(jets)
        if n_jets is None:
            full_event = True
            self.n_jets = len(jets)
        else:
            full_event = n_jets >= len(jets)
            self.n_jets = min(n_jets, len(jets))
        # check in the file for test allocation
        if 'test_allocation' in data:
            test_allocation = data['test_allocation']
        else:
            test_allocation = np.full(len(jets), True)
            test_percent = 0.2
            test_allocation[int(len(jets)*test_percent):] = False
            np.random.shuffle(test_allocation)
            np.savez(multijet_filename, **data, test_allocation=test_allocation)
        # assign test and train
        if full_event:
            _test_jets = np.array(jets)[test_allocation]
            self._jets = np.array(jets)[~test_allocation]
        else:
            test_allocation = test_allocation[:self.n_jets]
            _test_jets = np.array(jets)[:self.n_jets][test_allocation]
            self._jets = np.array(jets)[:self.n_jets][~test_allocation]
        # this will cause all the test walkers to be stored in memory
        # if memory becomes an issue might need to move to he glloupe data reading anyhow
        self._test_walkers = np.array([TreeWalker.TreeWalker(jet, jet.root_psudojetIDs[0])
                                       for jet in _test_jets])
        self._len = len(self._jets)
        # work out the dimensions
        valid_jet = next(j for j in jets if len(j._ints)>0)
        walker = TreeWalker.TreeWalker(valid_jet, valid_jet.root_psudojetIDs[0])
        self.dimensions = len(walker.leaf_val)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        jet = self._jets[idx]
        walker = TreeWalker.TreeWalker(jet, jet.root_psudojetIDs[0])
        return self.truth_tags[idx], walker

    @property
    def test_jets(self):
        return self._test_walkers

