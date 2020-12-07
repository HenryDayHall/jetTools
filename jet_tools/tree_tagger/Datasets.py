import os
import csv
import random
from torch.utils.data import Dataset
from torch import nn
#from ipdb import set_trace as st
import numpy as np
from jet_tools.tree_tagger import Components, LinkingFramework, FormJets, TreeWalker, InputTools, Constants
from sklearn import preprocessing
import torch

TEST_DS="/home/henry/lazy/dataset2/h1bBatch2_particles.awkd"

class EventWiseDataset(Dataset):
    """ """
    def __init__(self, database_name=None, num_events=-1, test_percent=0.2, shuffle=True, all_events=None):
        torch.set_default_tensor_type('torch.DoubleTensor')
        self.database_name = database_name
        self.eventWise = Components.EventWise.from_file(database_name)
        total_avalible = len(self.eventWise.Energy)
        test_percent = 0.2
        if num_events is -1 or num_events * (1+test_percent) > total_avalible:
            num_events = int(total_avalible/(1+test_percent))
        self._len = num_events
        if "IsTestEvent" in self.eventWise.columns:
            self._test_mask = self.eventWise.IsTestEvent
            assert len(self._test_mask) == total_avalible
        else:
            num_test_events = int(total_avalible*test_percent)
            test_mask = np.full(total_avalible, False, dtype=bool)
            test_mask[:num_test_events] = True
            np.random.shuffle(test_mask)
            self.eventWise.append(IsTestEvent=test_mask)
        self.num_test = num_events*test_percent
        self._test_indices = np.random.shuffle(np.where(test_mask)[0])[:self.num_test]
        self._train_indices = np.random.shuffle(np.where(~test_mask)[0])[:num_events]
        self.all_indices = np.concatenate((self._test_indices, self._train_indices))
        self._events = all_events[self._train_indices]
        self._test_events = all_events[self._test_indices]
        
    def _process_events(self):
        """ """
        raise NotImplementedError


    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        """ return events in numpy style """
        return self._events[idx]

    @property
    def test_events(self):
        """ """
        return self._test_events

    def write(self, folder_name=None):
        """
        

        Parameters
        ----------
        folder_name :
            Default value = None)

        Returns
        -------

        
        """
        raise NotImplementedError

    @classmethod
    def from_file(cls, database_name):
        """
        

        Parameters
        ----------
        database_name :
            

        Returns
        -------

        
        """
        folder_name = os.path.split(database_name)[0]
        num_events, events = cls._read_file(folder_name)
        return cls(database_name=database_name, num_events=num_events,
                   all_events=events)

    @classmethod
    def _read_file(cls, folder_name):
        """
        

        Parameters
        ----------
        folder_name :
            

        Returns
        -------

        
        """
        raise NotImplementedError


class TracksTowersDataset(EventWiseDataset):
    """ """
    def _process_databaseHepmc(self):
        """ """
        all_events = []
        n_events = len(self.all_indices)
        eventWise = self.eventWise
        for i, event_n in enumerate(self.all_indices):
            if i%100 == 0:
                print(f"{i/n_events:.1%}", end='\r')
            tracks_near_tower, towers_near_track = LinkingFramework.tower_track_proximity(eventWise)
            MCtruth = LinkingFramework.MC_truth_links(eventWise)
            event_info = gen_overall_data(eventWise)
            track_data = gen_track_data(eventWise, event_info)
            tower_data = gen_tower_data(eventWise, event_info)
            all_events.append([tower_data, track_data, towers_near_track, MCtruth])
        return all_events

    @property
    def track_dimensions(self):
        """ """
        _, tracks, _, _ = self[0]
        return tracks.shape[1]

    @property
    def tower_dimensions(self):
        """ """
        towers, _,  _, _ = self[0]
        return towers.shape[1]

    # define the file names
    numpy_save_name = "input_arrays.npz"
    proximities_save_name = "input_proximities.csv"

    @classmethod
    def _per_event_to_flat(cls, all_events):
        """
        

        Parameters
        ----------
        all_events :
            

        Returns
        -------

        
        """
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
        """
        

        Parameters
        ----------
        folder_name :
            Default value = None)

        Returns
        -------

        
        """
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
        """
        

        Parameters
        ----------
        database_name :
            

        Returns
        -------

        
        """
        folder_name = os.path.split(database_name)[0]
        num_events, events = cls._read_file(folder_name)
        return cls(database_name=database_name, num_events=num_events,
                   all_events=events)

    @classmethod
    def _flat_to_per_event(cls, cumulative_tower_entries, cumulative_track_entries,
                           all_towers, all_tracks, all_prox, all_truth):
        """
        

        Parameters
        ----------
        cumulative_tower_entries :
            param cumulative_track_entries:
        all_towers :
            param all_tracks:
        all_prox :
            param all_truth:
        cumulative_track_entries :
            
        all_tracks :
            
        all_truth :
            

        Returns
        -------

        
        """
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
        """
        

        Parameters
        ----------
        folder_name :
            

        Returns
        -------

        
        """
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
    """
    

    Parameters
    ----------
    eventWise :
        

    Returns
    -------

    
    """
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
    """
    

    Parameters
    ----------
    eventWise :
        param event_info:
    event_info :
        

    Returns
    -------

    
    """
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
    """
    

    Parameters
    ----------
    eventWise :
        param event_info:
    event_info :
        

    Returns
    -------

    
    """
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
    """
    simplify the problem

    Parameters
    ----------
    dataset :
        param num_pairs: (Default value = 1)
    num_pairs :
        (Default value = 1)

    Returns
    -------

    
    """
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
    """ """
    def __init__(self, database_name, jet_name="FastJet", n_train_jets=-1, shuffle=True, all_truth_jets=None):
        self.is_torch = False
        #print(f"Creating Dataset for {jet_name}")
        torch.set_default_tensor_type('torch.DoubleTensor')
        self.database_name = database_name
        self.eventWise = Components.EventWise.from_file(database_name)
        self.jet_name = jet_name
        test_percent = 0.2
        if all_truth_jets is None:
            jet_energies = getattr(self.eventWise, jet_name + "_Energy")
            total_events = len(jet_energies)
            print(f"Total events on disk {total_events}")
            jets_per_event = np.array([len(e) for e in jet_energies])
            self._cumulative_jets = np.cumsum(jets_per_event)
            total_jets = self._cumulative_jets[-1]
            print(f"Total jets on disk {total_jets}")
            if "IsTestEvent" in self.eventWise.columns:
                print("Test allocation exists")
                test_mask = self.eventWise.IsTestEvent
                assert len(test_mask) == total_events
            else:
                print("Assigning test events")
                num_test_events = int(total_events*test_percent)
                test_mask = np.full(total_events, False, dtype=bool)
                test_mask[:num_test_events] = True
                np.random.shuffle(test_mask)
                self.eventWise.append(IsTestEvent=test_mask)
            if n_train_jets is -1 or n_train_jets > int(total_jets* (1-test_percent)):
                n_train_jets = int(total_jets*(1-test_percent))
            test_events = np.where(test_mask)[0]
            np.random.shuffle(test_events)
            self._test_indices = self._event_idx_to_jet(test_events)
            self.num_test = min(len(self._test_indices),
                                int((test_percent*n_train_jets)/(1-test_percent)))
            print(f"Number of test events to be used {self.num_test}")
            train_events = np.where(~test_mask)[0]
            np.random.shuffle(train_events)
            self._train_indices = self._event_idx_to_jet(train_events)
            self._len = min(len(self._train_indices), n_train_jets)
            print(f"Number of train events to be used {len(self)}")
            self.all_indices = np.concatenate((self._test_indices, self._train_indices))
            print("Processing the jets")
            all_truth, all_jets = self._process_jets()
        else:
            #print("Jets are provided")
            all_truth, all_jets = all_truth_jets
            total_jets = len(all_truth)
            if n_train_jets is -1 or n_train_jets > int(total_jets * (1-test_percent)):
                n_train_jets = int(total_jets*(1-test_percent))
            self._len = n_train_jets
            self.num_test = total_jets - n_train_jets
        self.jets = all_jets[self.num_test:]
        self.truth = all_truth[self.num_test:]
        self.test_jets = all_jets[:self.num_test]
        self.test_truth = all_truth[:self.num_test]
        self._test = None
        #print("Dataset initilised")

    def to_torch(self, device):
        """
        

        Parameters
        ----------
        device :
            

        Returns
        -------

        
        """
        self.is_torch = True
        torch.set_default_tensor_type('torch.DoubleTensor')
        self.jets = torch.from_numpy(self.jets).to(device)
        #self.truth = torch.from_numpy(self.truth)
        self.test_jets = torch.from_numpy(self.test_jets).to(device)
        #self.test_truth = torch.from_numpy(self.test_truth)

    def _event_idx_to_jet(self, idx_list):
        """
        

        Parameters
        ----------
        idx_list :
            

        Returns
        -------

        
        """
        jet_idx = [np.arange(self._cumulative_jets[idx-1],
                             self._cumulative_jets[idx])
                   for idx in idx_list]
        return np.concatenate(jet_idx)

    def _process_jets(self):
        """ """
        raise NotImplementedError

    def find_multijet_file(self, dir_name):
        """
        

        Parameters
        ----------
        dir_name :
            

        Returns
        -------

        
        """
        in_dir = os.listdir(dir_name)
        possibles = [f for f in in_dir if f.startswith("pros_") and f.endswith(".npz")]
        if len(possibles) == 1:
            chosen = possibles[0]
        else:
            chosen = InputTools.list_complete("Which jet file? ", possibles)
        return os.path.join(dir_name, chosen)

    def __len__(self):
        return self._len

    @property
    def num_targets(self):
        """ """
        return self.truth.shape[1]

    @property
    def num_inputs(self):
        """ """
        raise NotImplementedError

    @classmethod
    def save_name(cls, database_name, folder_name=None):
        """
        

        Parameters
        ----------
        database_name :
            param folder_name: (Default value = None)
        folder_name :
            (Default value = None)

        Returns
        -------

        
        """
        ds_components = os.path.split(database_name)
        if folder_name is None:
            folder_name = ds_components[0]
        elif folder_name.endswith('/'):
            folder_name = folder_name[:-1]
        ds_base = ds_components[1].split('.', 1)[0]
        self_name = cls.__name__
        save_name = os.path.join(folder_name, f"{ds_base}_{self_name}.npz")
        return save_name

    def write(self, folder_name=None):
        """
        

        Parameters
        ----------
        folder_name :
            Default value = None)

        Returns
        -------

        
        """
        params = {"database_name": self.database_name,
                  "jet_name": self.jet_name,
                  "n_train_jets": len(self),
                  "shuffle": True}
        all_jets = np.vstack((self.test_jets, self.jets))
        all_truth = np.vstack((self.test_truth, self.truth))
        np.savez(self.save_name(self.database_name, folder_name), params=[params],
                 all_jets=all_jets, all_truth=all_truth)

    @classmethod
    def from_file(cls, dataset_name, folder_name=None):
        """
        

        Parameters
        ----------
        dataset_name :
            param folder_name: (Default value = None)
        folder_name :
            (Default value = None)

        Returns
        -------

        
        """
        params = cls._read_file(dataset_name, folder_name)
        return cls(**params)

    @classmethod
    def _read_file(cls, dataset_name, folder_name):
        """
        

        Parameters
        ----------
        dataset_name :
            param folder_name:
        folder_name :
            

        Returns
        -------

        
        """
        content = np.load(cls.save_name(dataset_name, folder_name))
        params = content['params'][0]
        params['all_truth_jets'] = (content['all_truth'], content['all_jets'])
        return params


class FlatJetDataset(JetWiseDataset):
    """ """
    def _process_jets(self):
        """ """
        per_event_columns = [c for c in self.eventWise.columns
                             if c.startswith("Event_")]
        per_jet_columns = [c for c in self.eventWise.columns if
                           c.startswith(self.jet_name + "_Std") or
                           c.startswith(self.jet_name + "_Ave") or
                           c.startswith(self.jet_name + "_Sum")]
        self.column_names = per_event_columns + per_jet_columns
        self.column_names.remove("Event_n")
        jets = np.empty((len(self.all_indices),
                         len(per_event_columns)+len(per_jet_columns)))
        truths = np.empty((len(self.all_indices), 1))
        n_train_jets = len(truths)
        event_num = 0
        event_start = 0
        eventWise = self.eventWise
        eventWise.selected_index = 0
        event_vars = [getattr(eventWise, c) for c in per_event_columns]
        jet_vars = [getattr(eventWise, c) for c in per_jet_columns]
        jet_pt_cut = Constants.min_jetpt
        tags = getattr(eventWise, self.jet_name + f"_{int(jet_pt_cut)}Tags")
        update=False
        for jet_num, idx in enumerate(sorted(self.all_indices)):
            if jet_num %100 == 0:
                print(f"{jet_num/n_train_jets:.2f:.1%}", end='\r', flush=True)
            while idx >= self._cumulative_jets[event_num]:
                event_num += 1
                update=True
            if update:
                event_start = self._cumulative_jets[event_num-1]
                eventWise.selected_index = event_num
                event_vars = [getattr(eventWise, c) for c in per_event_columns]
                jet_vars = [getattr(eventWise, c) for c in per_jet_columns]
                tags = getattr(eventWise, self.jet_name + f"_{int(jet_pt_cut)}Tags")
                update=False
            jet_in_event = idx - event_start
            jets[jet_num] = event_vars + [jv[jet_in_event] for jv in jet_vars]
            # anything with any tag is called signal
            truths[jet_num][0] = len(tags[jet_in_event]) > 0
        eventWise.selected_index = None
        return truths, jets

    def __getitem__(self, idx):
        if isinstance(idx, (int, np.int, np.int64)):  # becuase there a lots of int-like things....
            return self.truth[idx], self.jets[idx]
        # it's a slice or a list
        return list(zip(self.truth[idx], self.jets[idx]))

    @property
    def test_events(self):
        """ """
        if self._test is None:
            self._test = np.array(list(zip(self.test_truth, self.test_jets)))
        return self._test

    @property
    def num_inputs(self):
        """ """
        return self.jets.shape[1]


class JetTreesDataset(JetWiseDataset):
    """ """
    def _process_jets(self):
        """ """
        jets = np.empty((len(self.all_indices), 3))
        truths = np.empty((len(self.all_indices), 1))
        event_num = 0
        event_start = 0
        eventWise = self.eventWise
        eventWise.selected_index = 0
        parents = getattr(eventWise, self.jet_name + "_Parent")
        jet_pt_cut = Constants.min_jetpt
        tags = getattr(eventWise, self.jet_name + f"_{int(jet_pt_cut)}Tags")
        for jet_num, idx in enumerate(sorted(self.all_indices)):
            while idx >= self._cumulative_jets[event_num]:
                event_start = self._cumulative_jets[event_num]
                event_num += 1
                eventWise.selected_index = event_num
                parents = getattr(eventWise, self.jet_name + "_Parent")
                tags = getattr(eventWise, self.jet_name + "_{int(jet_pt_cut)}Tags")
            jet_in_event = idx - event_start
            root = np.where(parents[jet_in_event] == -1)[0][0]
            jets[jet_num] = [event_num, jet_in_event, root]
            # anything with any tag is called signal
            truths[jet_num][0] = len(tags[jet_in_event]) > 0
        eventWise.selected_index = None
        return truths, jets

    def __getitem__(self, idx):
        if isinstance(idx, (int, np.int, np.int64)):  # becuase there a lots of int-like things....
            jet = self.jets[idx]
            walker = TreeWalker.TreeWalker(self.eventWise, self.jet_name, 
                                           jet[0], jet[1], jet[3])
            return self.truth[idx], walker
        # it's a slice or a list
        jets = self.jets[idx]
        walkers = [TreeWalker.TreeWalker(self.eventWise, self.jet_name, 
                                         jet[0], jet[1], jet[3])
                  for jet in jets]
        return list(zip(self.truth[idx], walkers))

    @property
    def test_events(self):
        """ """
        if self._test is None:
            self._test = np.array(list(zip(self.test_truth,
                                           [TreeWalker.TreeWalker(self.eventWise, self.jet_name, 
                                                                  jet[0], jet[1], jet[3])
                                            for jet in self.test_jets])))
        return self._test

