""" tests to test the OptimiseCluster module """
from jet_tools import Components, FormJets, TrueTag, FormShower, OptimiseCluster
import numpy as np
import numpy.testing as tst
import unittest.mock


def throws_LinAlgError(*args, **kwargs):
    raise np.linalg.LinAlgError

def test_event_batch_loss():
    # use some of the sample data to have an eventWise to wor with
    eventWise = Components.EventWise.from_file("mini_data/mini.hepmc.awkd")
    # set up the required information
    Components.add_all(eventWise)
    FormJets.create_jetInputs(eventWise, filter_functions=[FormJets.filter_ends, FormJets.filter_pt_eta],
                              batch_length=np.inf)
    FormShower.append_b_idxs(eventWise)
    TrueTag.add_detectable_fourvector(eventWise, silent=True)
    Components.add_mass(eventWise, "DetectableTag")
    # then get loss
    n_events = len(eventWise.JetInputs_SourceIdx)
    jet_class = FormJets.SpectralFull
    jet_params = {}
    other_hyperparams = {'min_tracks': 2, 'max_angle2': 0.8, 'min_jetpt': 15}
    generic_data = {}
    # get the loss of the first 4 event
    num_losses = min(4, n_events)
    losses = np.empty(num_losses)
    for event_n in range(num_losses):
        eventWise.selected_index = event_n
        loss = OptimiseCluster.event_loss(eventWise,
                                          jet_class,
                                          jet_params,
                                          other_hyperparams,
                                          generic_data)
        losses[event_n] = loss
    assert np.all(losses >= 0), "Loss should never be negative"
    assert len(losses) == len(set(losses)), "Each event should produce a diferent loss"
    # now try them all together as a batch
    generic_data = {"SuccessCount": np.zeros(num_losses),
                    "FailCount": np.zeros(num_losses)}
    batchloss = OptimiseCluster.batch_loss(range(num_losses), eventWise,
                                           jet_class, jet_params, other_hyperparams,
                                            generic_data) 
    tst.assert_allclose(generic_data["SuccessCount"], np.ones(num_losses))
    tst.assert_allclose(batchloss, np.sum(losses)/num_losses)
    # now mock eventloss so it throws an error
    with unittest.mock.patch('jet_tools.OptimiseCluster.event_loss',
                             new=throws_LinAlgError):
        batchloss2 = OptimiseCluster.batch_loss(range(num_losses), eventWise,
                                               jet_class, jet_params, other_hyperparams,
                                                generic_data) 
        tst.assert_allclose(batchloss2, min(1e5, (2**num_losses-1)/num_losses))

def test_get_usable_events():
    # try when there is no counts yet
    simple_ew = lambda : None
    n_events = 5
    simple_ew.JetInputs_PT = np.empty(n_events)
    usable = OptimiseCluster.get_usable_events(simple_ew)
    tst.assert_allclose(usable, list(range(n_events)))
    # now with existing success and fails
    simple_ew.SuccessCount = np.array([0., 1., 6., 100., 1.])
    simple_ew.FailCount = np.array([0., 99., 100., 0., 100.])
    expected = [0, 1, 2, 3]
    usable = OptimiseCluster.get_usable_events(simple_ew)
    tst.assert_allclose(usable, expected)


def test_BatchSampler():
    batch_size = 1
    data = list(range(3))
    num_samples = 4
    sampler = OptimiseCluster.BatchSampler(data, batch_size, num_samples)
    # the length should be the nuber of unique batches posible in the data
    assert len(sampler) == 3
    # shouldb able to get 4 samples from this
    samples = list(iter(sampler))
    assert len(samples) == 4
    assert {x for sample in samples for x in sample}.issubset(data)
    # try again with a larger batch
    batch_size = 3
    sampler = OptimiseCluster.BatchSampler(data, batch_size)
    # the length should be the nuber of unique batches posible in the data
    assert len(sampler) == 1
    # shouldb able to get 4 samples from this
    sampler = iter(sampler)
    samples = [next(sampler) for _ in range(3)]
    assert {x for sample in samples for x in sample} == set(data)



