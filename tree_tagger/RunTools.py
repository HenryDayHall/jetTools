''' module for data reading tools '''
import os
import sys
import time
import datetime
import csv
from ast import literal_eval
from pathlib import Path
from copy import deepcopy
from ipdb import set_trace as st
import torch
import numpy as np
from tree_tagger import Constants


def remove_file_extention(path):
    suffix = Path(path).suffix
    if len(suffix) == 0:
        return path
    return path[:-len(suffix)]


def str_is_type(s_type, s, accept_none=False):
    if s is None and accept_none:
        return True
    if isinstance(s, s_type) or s is None: # the conversion already happened
        return True
    try:
        s_type(s)
        return True
    except ValueError:
        return False


class Run:
    # the list arguments in the info line
    # given inorder of precidence when performing comparison
    setting_names = ["net_type", "data_folder",  # nature of the net itself
                     "weight_decay", # minor net parameters
                     "batch_size", "loss_type", "inital_lr",  # training parameters
                     "auc", "lowest_loss", "notes"] # results
    # the permissable values
    net_types = ['trackTower_projectors']
    net_names = {'trackTower_projectors': ['track', 'tower']}
    loss_functions = ["BCE"]
    # the tests for identifying the arguments
    arg_tests = {"data_folder"   : os.path.exists,
                 "batch_size"  : lambda s: str_is_type(int, s),
                 "inital_lr"   : lambda s: str_is_type(float, s),
                 "weight_decay": lambda s: str_is_type(float, s),
                 "net_type"    : lambda s: s.lower() in Run.net_types,
                 "loss_type"   : lambda s: s.upper() in Run.loss_functions,
                 "auc"         : lambda s: str_is_type(float, s) or (s is None),
                 "lowest_loss" : lambda s: str_is_type(float, s) or (s is None),
                 "notes"       : lambda s: True}

    arg_convert = {"data_folder"   : lambda s: s,
                   "batch_size"  : lambda s: int(s),
                   "inital_lr"   : lambda s: float(s),
                   "weight_decay": lambda s: float(s),
                   "net_type"    : lambda s: s.lower(),
                   "loss_type"   : lambda s: s.upper(),
                   "auc"         : lambda s: None if s is None else float(s),
                   "lowest_loss" : lambda s: None if s is None else float(s),
                   "notes"       : lambda s: s}

    arg_defaults = {"data_folder"   : "CSVv2_all_jets.h5",
                    "time"        : 3000,
                    "batch_size"  : 1000,
                    "inital_lr"   : 0.1,
                    "weight_decay": 0.01,
                    "net_type"    : net_types[0],
                    "loss_type"   : loss_functions[0],
                    "auc"         : None,
                    "lowest_loss" : None,
                    "notes"       : ""}
                          
    def __init__(self, folder_name, run_name, accept_empty=False):
        self.folder_name = folder_name
        self.base_name = os.path.join(folder_name, run_name)
        self.progress_file_name = self.base_name + ".txt"
        with open(self.progress_file_name, 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=' ')
            # the first line in the file is info
            info_line = next(csv_reader)
            # the line under this is the columns
            try:
                self.column_headings = next(csv_reader)
                # if it exists the run is not empty
                self.empty_run = False
            except StopIteration:
                if accept_empty:
                    self.empty_run = True
                else:
                    # not a full run
                    raise ValueError
            if not self.empty_run:
                # the first column is always time_stamps
                if self.column_headings[0] != 'time_stamps':
                    raise ValueError
                # then tehre is the table
                table = []
                for row in csv_reader:
                    table.append(row)
                self.table = np.array(table, dtype=np.float)
                # if the table is smaller than the column headings drop those columns
                if self.table.shape[1] < len(self.column_headings):
                    self.column_headings = self.column_headings[:self.table.shape[1]]
                assert len(self.column_headings) == self.table.shape[1], "Number columns not equal to number column headdings"
        # now process the info line
        self.settings = self.process_info_line(info_line)
        # if the net is empty these things should not exist
        if self.empty_run:
            if self.settings['auc'] is not None:
                print("Warning, found auc {} in empty run".format(self.settings['auc']))
                self.settings['auc'] = None
            if self.settings['lowest_loss'] is not None:
                print("Warning, found lowest_loss {} in empty run".format(self.settings['lowest_loss']))
                self.settings['lowest_loss'] = None
        self.settings["pretty_name"] = run_name
        # also make file names for the best and last nets
        net_extention = ".torch"
        self.last_net_files = []
        self.best_net_files = []
        for name in self.net_names[self.settings['net_type']]:
            self.last_net_files.append(self.base_name + "_last_" + name + net_extention)
            self.best_net_files.append(self.base_name + "_best_" + name + net_extention)
        # check to see if either of these exist and load if so
        try:
            self.last_nets = [torch.load(name, map_location='cpu')
                              for name in self.last_net_files]
        except FileNotFoundError:
            # if the run is not empty there should be a last net
            if not self.empty_run:
                print("Warning; not an empty run, but no last net found!")
        try:
            self.best_nets = [torch.load(name, map_location='cpu')
                              for name in self.best_net_files]
            self.set_best_nets(best_nets, self.settings['lowest_loss'])
        except FileNotFoundError:
            if not self.empty_run:
                print("Warning; not an empty run, but no best net found!")

    def __process_idx(self, idx):
        # if there is only one item make the second item a total slice
        if not isinstance(idx, tuple):
            idx = (idx, slice(None))
        elif len(idx) == 1:
            idx = (idx[0], slice(None))
        else:
            assert len(idx) < 3, "Only two dimensions to the run record!"
        processed_idx = [None, None]
        for n in range(2):
            i = idx[n]
            # put numpy arrays to lists
            if isinstance(i, np.ndarray):
                i = list(i)
            # deal with the column headidings case
            if n == 1:
                if isinstance(i, str):
                    assert i in self.column_headings, "index {} not in column_headings"
                    i = self.column_headings.index(i)
                elif isinstance(i, list) and set(i).issubset(self.column_headings):
                    # if it is a list of coumn headings convert them
                    i = [self.column_headings.index(ie) for ie in i]
            # now the whole thing should be ints or a slice
            if isinstance(i, list):
                i = [int(ie) for ie in i]
            elif not isinstance(i, slice): # dont need to convert slices
                i = int(i)
            processed_idx[n] = i
        return tuple(processed_idx)

    def __getitem__(self, idx):
        processed_idx = self.__process_idx(idx)
        return self.table[processed_idx]

    def __setitem__(self, idx, value):
        processed_idx = self.__process_idx(idx)
        self.table[idx] = value


    def __len__(self):
        return self.table.shape[0]

    def __str__(self):
        return "{}; net {}, batch {}"\
                .format(self.base_name,
                        self.settings["net_type"],
                        self.settings["batch_size"])

    # inequlity comparisons
    def __eq__(self, other):
        tables_eq = np.allclose(self.table, other.table)
        names_eq = self.setting_names == other.setting_names
        settings_eq = np.all([self.settings[name] == other.settings[name]
                        for name in self.setting_names])
        return tables_eq and names_eq and settings_eq


    def __ne__(self, other):
        return not self == other


    def __gt__(self, other):
        for name in self.setting_names:
            seset = self.settings[name]
            otset = other.settings[name]
            if seset is not None and otset is not None:
                if seset != otset:
                    return seset > otset
        # if we get here they are all either none or equal
        return False

    def __lt__(self, other):
        for name in self.setting_names:
            seset = self.settings[name]
            otset = other.settings[name]
            if seset is not None and otset is not None:
                if seset != otset:
                    return seset < otset
        # if we get here they are all either none or equal
        return False

    def __le__(self, other):
        return not self > other
    
    def __ge__(self, other):
        return not self < other

    @property
    def column_headings(self):
        return self.__column_headings

    @column_headings.setter
    def column_headings(self, column_headings):
        # we can only set column headdings if there arnt ay already
        assert not hasattr(self, 'column_headings'), "Column headdings already set!"
        # there should definetly be column headdings before data in the table
        assert not hasattr(self, 'table'), "Data in table before column headdings chosen!"
        self.__column_headings = column_headings
        self.table = np.array([]).reshape((0, len(self.column_headings)))

    def append(self, line):
        # the line to append must have a value for every column
        assert len(line) == len(self.column_headings), "tried to append values not equal to number of columns"
        self.table = np.vstack((self.table, line))


    def process_info_line(self, info_line):
        # kill empty strings
        info_line = list(filter(None, info_line))
        # if the info line looks like a dict
        # then we can just read it
        info_line = ' '.join(info_line)
        args = literal_eval(info_line)
        assert isinstance(args, dict)
        # and convert he values
        for key in (set(args.keys() & set(Run.arg_tests.keys()))):
            # check the input is valid
            assert Run.arg_tests[key](args[key]), "problem with {}".format(key)
            # convert and store
            args[key] = Run.arg_convert[key](args[key])
        # find out if anything didn't get added
        for arg_name in self.setting_names:
            if arg_name not in args.keys():
                print("Missing {}, assuming default value {}"
                      .format(arg_name, self.arg_defaults[arg_name]))
                args[arg_name] = self.arg_defaults[arg_name]
        return args

    def load_net(self, version='best', name=''):
        # TODO update for new nets
        raise NotImplementedError()
        name_list = self.net_names[self.settings['net_type']]
        if name not in name_list:
            raise ValueError(f"Net {name} not in this run. This run contains {name_list}.")
        attribute_name = version + '_nets'
        if hasattr(self, attribute_name):
            param_dict = getattr(self, attribute_name)
        else:
            # then this net was never created
            return None
        with_sigmoid = self.settings['loss_type'] != "BCE"
        # load the net
        #if (self.settings["net_type"].lower() == 'csvv2' 
        #        and self.settings["backend"] == 'pytorch'):
        #    net = CSVv2_Net(n_variables, n_targets, with_sigmoid)
        #    net.load_state_dict(param_dict)
        #elif (self.settings["net_type"].lower() == 'deepcsv'
        #        and self.settings["backend"] == 'pytorch'):
        #    net = DeepCSV_Net(n_variables, n_targets, with_sigmoid)
        #    net.load_state_dict(param_dict)
        #elif (self.settings["net_type"].lower() == 'old'
        #        and self.settings["backend"] == 'pytorch'):
        #    # work on this later
        #    # net = Old_net(22, 3)
        #    # state_dict = torch.load(net_name, map_location='cpu')["net"][-1]
        #    # net.load_state_dict(state_dict)
        #elif self.settings["backend"] == 'keras':
        #    # work on this later
        #    raise NotImplementedError()
        #    # net = load_model(net_name)
        #else:
        #    print("Error; self.settings['net_type'] {} not recognised, expecting 'csvv2' or 'deepcsv'".format(self.settings["net_type"]))
        #    sys.exit(1)
        #return net

    def get_time(self):
        # pick the first timestamp
        t_0 = self['time_stamps', 0]
        time_0 = time.gmtime(t_0)
        self.time = datetime.datetime(*time_0[:6])
        return self.time

    def add_auc(self):
        _, _, auc = calculate_roc(self, self.settings['data_folder'])
        self.settings['auc'] = auc

    def to_file(self):
        # just clear the file and start from scratch
        with open(self.progress_file_name, 'w') as pf:
            writer = csv.writer(pf, delimiter=' ')
            writer.writerow([str(self.settings)])
            writer.writerow(self.column_headings)
            for row in self.table:
                writer.writerow(row)
        # save the best and last nets, they should exist by now
        for net, file_name in zip(self.best_nets, self.best_net_files):
            torch.save(net, file_name)
        try:
            for net, file_name in zip(self.last_nets, self.last_net_files):
                torch.save(net, file_name)
        except AttributeError:
            print("Didn't find a last_net")

    @property
    def best_nets(self):
        return self.__best_nets

    @best_nets.setter
    def best_nets(self, param_dicts):
        # don't allow direct setting of the best net
        # must use set method to ensure we also get the lowest loss
        raise AttributeError("Do not set this directly, use set_best_net")

    def set_best_nets(self, param_dicts, lowest_loss):
        # could make an assertion about this loss being lower thant previous ones,
        # but as I expect this code to be called many times I will omit it
        self.settings['lowest_loss'] = float(lowest_loss)
        self.__best_nets = param_dicts

    @property
    def last_nets(self):
        return self.__last_nets

    @last_nets.setter
    def last_nets(self, param_dicts):
        self.__last_nets= deepcopy(param_dicts)

            

def calculate_roc(run, focus=0, target_flavour='b', ddict_name=None):
    raise NotImplementedError

if __name__ == '__main__':
    pass
    
