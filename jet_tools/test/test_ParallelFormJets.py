""" Module to test the code in ParallelFormJets"""
import psutil
import pytest
import numpy.testing as tst
import shutil
import numpy as np
from ipdb import set_trace as st
from tree_tagger import ParallelFormJets, Components, FormJets
from test.tools import TempTestDir
import unittest.mock
import time
import os
import awkward

def test_make_jet_name():
    expected = "DogJet3"
    found = ParallelFormJets.make_jet_name("Dog", 3.)
    assert found == expected


def test_name_generator():
    # should work fine with no existing jets
    jet_class = "Horse"
    gen = ParallelFormJets.name_generator(jet_class, [])
    assert next(gen) == "HorseJet1"
    assert next(gen) == "HorseJet2"
    # and should work if there are jets
    jet_class = "Horse"
    existing_jets = ["HorseJet1", "HorseJet3", "CatJet2", "CatJet0"]
    gen = ParallelFormJets.name_generator(jet_class, existing_jets)
    assert next(gen) == "HorseJet2"
    assert next(gen) == "HorseJet4"


fake_clusters = []
return_required = False
time_delay = 1
def fake_cluster_multiapply(eventWise, cluster_algorithm, dict_jet_params={},
                            jet_name=None, batch_length=100, silent=False):
    time.sleep(time_delay)
    total_dict = {'eventWise': eventWise, 'cluster_algorithm': cluster_algorithm,
                  'dict_jet_params': dict_jet_params, 'jet_name': jet_name,
                  'batch_length': batch_length, 'silent':silent}
    fake_clusters.append(total_dict)
    return return_required
    

def test_worker():
    # mock multiapply - it has been tested elsewhere
    with unittest.mock.patch('tree_tagger.FormJets.cluster_multiapply',
                             new=fake_cluster_multiapply):
        with TempTestDir("tst") as temp_dir:
            jet_name = "GoodJet"
            ew = Components.EventWise(temp_dir, "file.awkd")
            ew.write()
            eventWise_path = os.path.join(temp_dir, ew.save_name)
            # get rid of a continue file if there is one or we will get stuck
            try:
                os.remove('continue')
            except FileNotFoundError:
                pass
            # running with the continue condition should resutl in instant shutdown
            ParallelFormJets._worker(eventWise_path, 'continue', 'SpectralMean', jet_name, {}, 13)
            assert len(fake_clusters) == 0
            # running with progfiling should strill resilt in a profile being made
            ParallelFormJets.worker(eventWise_path, 'continue', 'SpectralMean', jet_name, {}, 13)
            assert len(fake_clusters) == 0
            assert os.path.exists(eventWise_path.replace('.awkd', '.prof'))
            # now if we run for 10 seconds we expect less that 11 but more than 7
            ParallelFormJets._worker(eventWise_path, time.time()+10, 'SpectralMean', jet_name, {}, 13)
            assert len(fake_clusters) > 7
            assert len(fake_clusters) < 11
            for fake in fake_clusters:
                assert fake['eventWise'] == ew
                assert fake['cluster_algorithm'] == FormJets.SpectralMean
                assert fake['batch_length'] == 13
                assert len(fake['dict_jet_params']) == 0
            with pytest.raises(ValueError):
                ParallelFormJets._worker(eventWise_path, None, 'Spectral', jet_name, {}, 10)

    

def test_make_n_working_fragments():
    with TempTestDir("tst") as dir_name:
        # calling this one a directory that dosn't contain any eventWise objects
        # should raise a file not found erroj
        with pytest.raises(FileNotFoundError):
            ParallelFormJets.make_n_working_fragments(dir_name, 3, "squibble")
        # trying to split something that isn't an eventwise shoudl raise a FileNotFoundError
        wrong_path = os.path.join(dir_name, "wrong.ods")
        open(wrong_path, 'w').close()  # equivalent of touch
        with pytest.raises(FileNotFoundError):
            ParallelFormJets.make_n_working_fragments(wrong_path, 3, "flob")
        os.remove(wrong_path)
        # now make a real eventWise to play with
        save_name = "test.awkd"
        ew = Components.EventWise(dir_name, save_name)
        ew_path = os.path.join(dir_name, save_name)
        # try with 12 events
        n_events = 12
        params = {}
        params['Event_n'] = awkward.fromiter(np.arange(n_events))
        params['JetInputs_Energy'] = awkward.fromiter(np.arange(n_events))
        unfinished_jet = 'DogJet'
        n_unfinished = 6
        params[unfinished_jet + '_Energy'] = awkward.fromiter(np.random.rand(n_events-n_unfinished))
        params[unfinished_jet + '_Food'] = awkward.fromiter([np.random.rand(np.random.randint(5)) for _ in range(n_events-n_unfinished)])
        finished_jet = 'CatJet'
        params[finished_jet + '_Energy'] = awkward.fromiter([[awkward.fromiter(np.random.rand(np.random.randint(5)))
                                       for _ in range(np.random.randint(5))]
                                      for _ in range(n_events)])
        ew.append(**params)
        # making fragments of the finished jet should result in no change
        paths = ParallelFormJets.make_n_working_fragments(ew_path, 3, finished_jet)
        assert isinstance(paths, bool)
        assert paths
        # check nothing else has been made in the dir
        assert len(os.listdir(dir_name)) == 1
        # now split the unfinished jet
        paths = ParallelFormJets.make_n_working_fragments(ew_path, 3, unfinished_jet)
        assert len(paths) == 3
        ew.selected_index = None
        # there is no garentee that the split events will hold the same order
        expected_indices = list(range(n_events - n_unfinished, n_events))
        for path in paths:
            ew_part = Components.EventWise.from_file(path)
            indices_here = ew_part.JetInputs_Energy.tolist()
            for i in indices_here:
                expected_indices.remove(i)
            tst.assert_allclose(ew_part.CatJet_Energy.flatten().flatten().tolist(),
                                ew.CatJet_Energy[indices_here].flatten().flatten().tolist())
        assert not expected_indices, "Didn't find all the expected indices"
        # if we ask for the same split again it would not do anything
        paths2 = ParallelFormJets.make_n_working_fragments(ew_path, 3, unfinished_jet)
        assert set(paths2) == set(paths)
        paths2 = ParallelFormJets.make_n_working_fragments(os.path.split(paths[0])[0],
                                                           3, unfinished_jet)
        assert set(paths2) == set(paths)
        # if we ask for a diferent number of paths it should recluster and then split
        paths3 = ParallelFormJets.make_n_working_fragments(ew_path, 2, unfinished_jet)
        assert len(paths3) == 2
        # there is no garentee that the split events will hold the same order
        expected_indices = list(range(n_events - n_unfinished, n_events))
        for path in paths3:
            ew_part = Components.EventWise.from_file(path)
            indices_here = ew_part.JetInputs_Energy.tolist()
            for i in indices_here:
                expected_indices.remove(i)
            tst.assert_allclose(ew_part.CatJet_Energy.flatten().flatten().tolist(),
                                ew.CatJet_Energy[indices_here].flatten().flatten().tolist())
        assert not expected_indices, "Didn't find all the expected indices"
        # remove any existing directories
        [shutil.rmtree(os.path.join(dir_name, name))
                for name in os.listdir(dir_name) if '.' not in name]
        # try fragmenting and then joining
        ew.fragment("JetInputs_Energy", n_fragments=3)
        fragment_dir = next(os.path.join(dir_name, name) for name in os.listdir(dir_name)
                            if "fragment" in name)
        Components.EventWise.combine(fragment_dir, ew.save_name.split('.', 1)[0])
        ParallelFormJets.make_n_working_fragments(ew_path, 4, finished_jet)
        # check it has reconstructed the original eventWise
        found = os.listdir(fragment_dir)
        assert len(found) == 1
        new_ew = Components.EventWise.from_file(os.path.join(fragment_dir, found[0]))
        order = np.argsort(new_ew.JetInputs_Energy)
        dog_order = order[order<n_events-n_unfinished]
        tst.assert_allclose(ew.JetInputs_Energy, new_ew.JetInputs_Energy[order])
        tst.assert_allclose(ew.DogJet_Energy, new_ew.DogJet_Energy[dog_order])
        for i, j in enumerate(dog_order):
            tst.assert_allclose(ew.DogJet_Food[i], new_ew.DogJet_Food[j])
        for i, j in enumerate(order):
            for evt, new_evt in zip(ew.CatJet_Energy[i], new_ew.CatJet_Energy[j]):
                tst.assert_allclose(evt, new_evt)

            
def fake_worker(eventWise_path, run_condition, jet_class, jet_name, cluster_parameters, batch_size):
    eventWise = Components.EventWise.from_file(eventWise_path)
    eventWise.append(Catto=[2, 3])
    eventWise.append_hyperparameters(run_condition=run_condition,
                                     jet_class=jet_class,
                                     cluster_parameters=cluster_parameters,
                                     batch_size=batch_size)


def test_generate_pool():
    # mock _worker, this is tested above
    with unittest.mock.patch('tree_tagger.ParallelFormJets._worker', new=fake_worker):
        with TempTestDir("tst") as temp_dir:
            ew = Components.EventWise(temp_dir, "file.awkd")
            ew.append(JetInputs_Energy = np.arange(100))
            eventWise_path = os.path.join(temp_dir, ew.save_name)
            # get rid of a continue file if there is one or we will get stuck
            try:
                os.remove('continue')
            except FileNotFoundError:
                pass
            end_time = time.time() + 100
            jet_class = "Dog"
            jet_params = {"Bark": 3}
            n_cores = psutil.cpu_count()
            found = ParallelFormJets.generate_pool(eventWise_path, jet_class, jet_params, "Bali",
                                                   leave_one_free=True, end_time=end_time)
            assert found, "One of the processes crashed"
            # now we expect to find a subdirectory containing n_cores - 1 eventWise
            subdir = next(os.path.join(temp_dir, name) for name in os.listdir(temp_dir)
                          if ".awkd" not in name)
            paths = [os.path.join(subdir, name) for name in os.listdir(subdir)]
            assert len(paths) == n_cores - 1
            # check the ake worker has been run on all of them
            for path in paths:
                ew = Components.EventWise.from_file(path)
                tst.assert_allclose(ew.Catto, [2, 3])
                assert ew.run_condition == end_time
                assert ew.jet_class == jet_class
                assert ew.cluster_parameters["Bark"] == 3
                assert ew.batch_size == 500
            # tidy up
            shutil.rmtree(os.path.split(path)[0])
            # it should work fine with a continue file too
            open('continue', 'w').close()
            found = ParallelFormJets.generate_pool(eventWise_path, jet_class, jet_params, "Bali",
                                                   leave_one_free=False)
            assert found, "One of the processes crashed"
            # now we expect to find a subdirectory containing n_cores eventWise
            subdir = next(os.path.join(temp_dir, name) for name in os.listdir(temp_dir)
                          if ".awkd" not in name)
            paths = [os.path.join(subdir, name) for name in os.listdir(subdir)]
            assert len(paths) == n_cores
            # check the ake worker has been run on all of them
            for path in paths:
                ew = Components.EventWise.from_file(path)
                tst.assert_allclose(ew.Catto, [2, 3])
                assert ew.run_condition == 'continue'
                assert ew.jet_class == jet_class
                assert ew.cluster_parameters["Bark"] == 3
                assert ew.batch_size == 500



def test_remove_partial():
    paths = []
    with TempTestDir("tst") as dir_name:
        # calling it on no paths should run fine
        ParallelFormJets.remove_partial([])
        # now make a real eventWise to play with
        save_name = "test1.awkd"
        ew = Components.EventWise(dir_name, save_name)
        ew_path1 = os.path.join(dir_name, save_name)
        paths.append(ew_path1)
        n_events = 12
        params = {}
        params['Event_n'] = awkward.fromiter(np.arange(n_events))
        params['JetInputs_InputIdx'] = awkward.fromiter(np.arange(n_events))
        unfinished_jet = 'CatJet'
        n_unfinished = 6
        params[unfinished_jet + '_InputIdx'] = awkward.fromiter(np.random.rand(n_events-n_unfinished))
        finished_jet = 'CatJetJet'
        params[finished_jet + '_InputIdx'] = awkward.fromiter(np.random.rand(n_events))
        ew.append(**params)
        # calling it with the wrong total length should yeild an error
        with pytest.raises(AssertionError):
            ParallelFormJets.remove_partial(paths, n_events+1)
        # calling it with the right length should remove the right jet
        ParallelFormJets.remove_partial(paths, n_events)
        ew = Components.EventWise.from_file(paths[0])
        for name in ["Event_n", finished_jet+"_InputIdx", "JetInputs_InputIdx"]:
            tst.assert_allclose(params[name], getattr(ew, name))
        assert unfinished_jet+"_InputIdx" not in ew.columns
        # doing it again should have no effect
        ParallelFormJets.remove_partial(paths, n_events)
        ew = Components.EventWise.from_file(paths[0])
        for name in ["Event_n", finished_jet+"_InputIdx", "JetInputs_InputIdx"]:
            tst.assert_allclose(params[name], getattr(ew, name))
        assert unfinished_jet+"_InputIdx" not in ew.columns



def test_recombine_eventWise():
    with TempTestDir("tst") as dir_name:
        # make a real eventWise to play with
        save_name = "test.awkd"
        ew = Components.EventWise(dir_name, save_name)
        ew_path = os.path.join(dir_name, save_name)
        n_events = 12
        ew.append(Event_n=awkward.fromiter(np.arange(n_events)))
        # calling it on an eventWise that hasn't been split should return the same eventWise
        found = ParallelFormJets.recombine_eventWise(ew_path)
        assert len(found.columns) == 1
        tst.assert_allclose(found.Event_n, ew.Event_n)
        # fragment the eventWise and delete the last fragment
        paths = ew.fragment("Event_n", n_fragments=3)
        fragment_dir = os.path.split(paths[0])[0]
        os.remove(paths[2])
        partial = ParallelFormJets.recombine_eventWise(ew_path)
        # there is no garuntee on which part was removed
        preserved = [i in partial.Event_n for i in ew.Event_n]
        assert sum(preserved) == 8
        # there should now be a joined component in fragment_dir
        assert next(name for name in os.listdir(fragment_dir) if "joined.awkd" in name)
        # repeating the exercize should ignore this joined component
        partial = ParallelFormJets.recombine_eventWise(ew_path)
        preserved = [i in partial.Event_n for i in ew.Event_n]
        assert sum(preserved) == 8



