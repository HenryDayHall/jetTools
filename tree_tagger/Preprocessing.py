# TODO Needs updates for uproot!
""" Preprocessig module takes prepared datafiles and performed generic processing tasks requird beofre data can be run as a net """
import numpy as np
from sklearn import preprocessing
import os


class Process:
    def __init__(self, start_filename, add_to=True):
        if start_filename.startswith("pros_"):
            # the naem is of a file that has already began processing
            self.processed_filename = start_filename
        else: # then make the processed name
            dir_name, f_name = os.path.split(start_filename)
            self.processed_filename = os.path.join(dir_name, "pros_" + f_name)
        if add_to:  # then we continue adding to an existing processed verison
            if os.path.exists(self.processed_filename):
                try:
                    self.existing_data = np.load(self.processed_filename)
                except OSError:
                    # the processed file is not redable
                    self.existing_data = np.load(start_filename)
            else: # go back to the start_name
                self.existing_data = np.load(start_filename)
        else: # go back to the start_name
            self.existing_data = np.load(start_filename)
        self.new_data = {}

    def __enter__(self):
        return self.existing_data, self.new_data

    def __exit__(self, exception_type, exception_value, traceback):
        existing_data = {k: v for k, v in self.existing_data.items() if k not in self.new_data}
        np.savez(self.processed_filename, **self.new_data, **existing_data)


def normalize_jets(multijet_filename):
    with Process(multijet_filename) as (existing_data, new_data):
        new_data['floats'] = preprocessing.robust_scale(existing_data['floats'])


def make_targets(multijet_filename):
    """ anything with at least one tag is considered signal """
    with Process(multijet_filename) as (existing_data, new_data):
        truth_tags = existing_data['truth_tags']
        new_data['truth_tags'] = np.array([bool(np.count_nonzero(tags)) for tags in truth_tags],
                                      dtype=int)


