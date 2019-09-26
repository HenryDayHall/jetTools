# TODO Needs updates for uproot!
import os
import csv
import random
from torch.utils.data import Dataset
from torch import nn
from ipdb import set_trace as st
import numpy as np
from tree_tagger import Components, LinkingFramework, FormJets, TreeWalker, InputTools
from sklearn import preprocessing
import torch

TEST_DS="/home/henry/lazy/dataset2/h1bBatch2_particles.awkd"


class EventWiseDataset(Dataset):
    def __init__(self, database_name=None, num_events=None, test_percent=0.2, shuffle=True, all_events=None):
        torch.set_default_tensor_type('torch.DoubleTensor')
        self.database_name = database_name
        self.eventWise = Components.EventWise.from_file(database_name)
        total_avalible = len(self.eventWise.Energy)
        if num_events is None:
            num_events = total_avalible
        elif num_events > total_avalible:
            num_events = total_avalible
        test_percent = 0.2
        self.num_test = int(num_events*test_percent)
        all_indices = np.arange(total_avalible)
        if "TestEventIdx" in self.eventWise.columns:
            self._test_indices = self.eventWise.TestEventIdx
            assert len(self._test_indices) >= self.num_test
        else:
            np.random.shuffle(all_indices)
            self._test_indices = all_indices[:self.num_test]
            self.eventWise.append(["TestEventIdx"], {"TestEventIdx", self._test_indices})
        all_indices = np.sort(all_indices)
        self.all_indices = all_indices
        train_mask = np.ones_like(all_indices)
        train_mask[self._test_indices] = 0
        self._train_indices = all_indices[train_mask]
        if shuffle:
            # shuffle the events
            random.shuffle(self._train_indices)
        # then take the first section as test data
        self._len = len(self._train_indices)
        if all_events is None:
            all_events = np.array(self._process_events())
        self._events = all_events[self._train_indices]
        self._test_events = all_events[self._test_indices]
        
    def _process_events(self):
        raise NotImplementedError


    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        """ return events in numpy style """
        return self._events[idx]

    @property
    def test_events(self):
        return self._test_events

    def write(self, folder_name=None):
        raise NotImplementedError

    @classmethod
    def from_file(cls, database_name):
        folder_name = os.path.split(database_name)[0]
        num_events, events = cls._read_file(folder_name)
        return cls(database_name=database_name, num_events=num_events,
                   all_events=events)

    @classmethod
    def _read_file(cls, folder_name):
        raise NotImplementedError


class TracksTowersDataset(EventWiseDataset):
    def _process_databaseHepmc(self):
        all_events = []
        n_events = len(self.all_indices)
        eventWise = self.eventWise
        for i, event_n in enumerate(self.all_indices):
            if i%100 == 0:
                print(f"{100*i/n_events}%", end='\r')
            tracks_near_tower, towers_near_track = LinkingFramework.tower_track_proximity(eventWise)
            MCtruth = LinkingFramework.MC_truth_links(eventWise)
            event_info = gen_overall_data(eventWise)
            track_data = gen_track_data(eventWise, event_info)
            tower_data = gen_tower_data(eventWise, event_info)
            all_events.append([tower_data, track_data, towers_near_track, MCtruth])
        return all_events

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
            folder_name = os.path.split(self.database_name)[0]
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
    def from_file(cls, database_name):
        folder_name = os.path.split(database_name)[0]
        num_events, events = cls._read_file(folder_name)
        return cls(database_name=database_name, num_events=num_events,
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


def gen_overall_data(eventWise):
    assert eventWise.selected_index != None
    info = []
    track_energies = eventWise.Track_Energy
    info.append(np.mean(track_energies))
    info.append(np.std(track_energies))
    tower_energies = eventWise.Tower_Energy
    info.append(np.mean(tower_energies))
    info.append(np.std(tower_energies))
    track_etas = eventWise.Track_Eta
    info.append(np.mean(track_etas))
    info.append(np.std(track_etas))
    tower_etas = eventWise.Tower_Eta
    info.append(np.mean(tower_etas))
    info.append(np.std(tower_etas))
    track_pts = eventWise.Track_PT
    info.append(np.mean(track_pts))
    info.append(np.std(track_pts))
    tower_pts = eventWise.Tower_PT
    info.append(np.mean(tower_pts))
    info.append(np.std(tower_pts))
    track_phis = eventWise.Track_Phi
    info.append(np.std(track_phis))
    tower_phis = eventWise.Tower_Phi
    info.append(np.std(tower_phis))
    return info


def gen_track_data(eventWise, event_info):
    assert eventWise.selected_index != None
    num_tracks = len(eventWise.Track_Energy)
    track_components = ["Birr", "PT", "Eta", "Phi",
                        "CtgTheta", "OuterEta", "OuterPhi",
                        "T", "X", "Y", "Z", "DX", "DY", "DZ",
                        "L", "D0"]
    data = np.empty((num_tracks, len(event_info) + len(track_components)))
    track_data = [getattr(eventWise, "Track_" + comp).reshape((-1, 1))
                  for comp in track_components]
    track_data += event_info
    track_data = np.hstack(track_data)
    return track_data


def gen_tower_data(eventWise, event_info):
    assert eventWise.selected_index != None
    num_towers = len(eventWise.Track_Energy)
    tower_components = ["Energy", "Eem", "Ehad",
                        "T", "ET", "Eta", "Phi",
                        "Edges0", "Edges1", "Edges2", "Edges3"]
    data = np.empty((num_towers, len(event_info) + len(tower_components)))
    tower_data = [getattr(eventWise, "Track_" + comp).reshape((-1, 1))
                  for comp in tower_components]
    tower_data += event_info
    tower_data = np.hstack(tower_data)
    return tower_data


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


class JetWiseDataset(Dataset):
    def __init__(self, database_name=None, jet_name="FastJet", n_jets=None, shuffle=True, all_truth_jets=None):
        torch.set_default_tensor_type('torch.DoubleTensor')
        self.database_name = database_name
        self.eventWise = Components.EventWise.from_file(database_name)
        jet_energies = getattr(self.eventWise, jet_name + "_Energy")
        jets_per_event = np.array([len(e) for e in jet_energies])
        self._cumulative_jets = np.cumsum(jets_per_event)
        total_avalible = self.cumulative_jets[-1]
        if num_events is None:
            num_events = total_avalible
        elif num_events > total_avalible:
            num_events = total_avalible
        test_percent = 0.2
        self.num_test = int(num_events*test_percent)
        all_indices = np.arange(total_avalible)
        if "TestEventIdx" in self.eventWise.columns:
            test_events = self.eventWise.TestEventIdx
        else:
            all_event_indices = np.arange(len(jet_energies))
            np.random.shuffle(all_event_indices)
            test_events = all_event_indices[:self.num_test]
            self.eventWise.append(["TestEventIdx"], {"TestEventIdx", test_events})
        self._test_indices = self._event_idx_to_jet(test_events)
        assert len(self._test_indices) >= self.num_test
        all_indices = np.sort(all_indices)
        self.all_indices = all_indices
        train_mask = np.ones_like(all_indices)
        train_mask[self._test_indices] = 0
        self._train_indices = all_indices[train_mask]
        if shuffle:
            # shuffle the events
            random.shuffle(self._train_indices)
        # then take the first section as test data
        self._len = len(self._train_indices)
        if all_truth_jets is None:
            all_truth, all_jets = np.array(self._process_jets())
        else:
            all_truth, all_jets = all_truth_jets
        self._jets = all_jets[self._train_indices]
        self._truth = all_truth[self._train_indices]
        test_jets = all_jets[self._test_indices]
        test_truth = all_truth[self._test_indices]
        # may cause memry issues
        self._test = np.array(list(zip(all_truth[self._test_indices], 
                                       [TreeWalker.TreeWalker(jet, jet.root_psudojetIDs[0])
                                        for jet in all_jets[self._test_indices]])))

    def _event_idx_to_jet(self, idx_list):
        jet_idx = [np.arange(self._cumulative_jets[idx-1],
                             self._cumulative_jets[idx])
                   for idx in idx_list]
        return np.concatenate(jet_idx)

    def _process_jets(self):
        raise NotImplementedError

    def find_multijet_file(self, dir_name):
        in_dir = os.listdir(dir_name)
        possibles = [f for f in in_dir if f.startswith("pros_") and f.endswith(".npz")]
        if len(possibles) == 1:
            chosen = possibles[0]
        else:
            chosen = InputTools.list_complete("Which jet file? ", possibles)
        return os.path.join(dir_name, chosen)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        if isinstance(idx, (int, np.int, np.int64)):  # becuase there a lots of int-like things....
            jet = self._jets[idx]
            walker = TreeWalker.TreeWalker(jet, jet.root_psudojetIDs[0])
            return self._truth[idx], walker
        # it's a slice or a list
        jets = self._jets[idx]
        walkers = [TreeWalker.TreeWalker(jet, jet.root_psudojetIDs[0])
                  for jet in jets]
        return list(zip(self._truth[idx], walkers))

    @property
    def test_events(self):
        return self._test

    @property
    def num_targets(self):
        return self._truth.shape[1]


# TODO point reached
class JetTreesDataset(JetWiseDataset):
    def __process_jets(self):
        eventWise = self.eventWise
        eventWise.selected_index = None
        truth_tags = eventWise
        # truth_tags to taget form 
        if len(truth_tags.shape) == 1:
            truth_tags = truth_tags.reshape((-1, 1))
        truth_tags = torch.DoubleTensor(truth_tags)

