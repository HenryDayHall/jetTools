"""tools for training the NN """
from ipdb import set_trace as st
import os
import numpy as np
import pickle
import time
import torch
import torch.nn.functional

def single_pass(nets, run, dataloader, validation_events, test_events, device, train_losser, validation_losser, test_losser, weight_decay):
    # make initial measures
    for net in nets:
        net.eval()
    last_time = time.time()
    epoch_reached = len(run)
    # need one batch to get an loss
    event_data = next(dataloader.__iter__())[0] # only want one event
    # find loss
    training_loss = train_losser(event_data, nets, device).item()
    # validation loss
    validation_loss = 0
    for v_event in validation_events:
        validation_loss += validation_losser(v_event, nets, device).item()
    # test loss
    test_loss = 0
    for t_event in test_events:
        test_loss += test_losser(t_event, nets, device).item()
    # if the run starts empty record a new best and last net
    # the pickle stuff is to make a clone (NB deepcopy not sufficient for this)
    state_dicts = [pickle.loads(pickle.dumps(net.state_dict()))
                   for net in nets]
    if run.empty_run:
        run.set_best_state_dicts(state_dicts, test_loss)
        run.last_net = state_dicts
    elif test_loss < run.settings['lowest_loss']:
        run.set_best_state_dicts(state_dicts, test_loss)
    # weights
    mag_weights = 0
    for net in nets:
        mag_weights += np.sum([torch.sum(torch.abs(w)).item()
                               for w in net.get_weights()])
    # learning rate
    if epoch_reached == 0:
        learning_rate = run.settings['inital_lr']
    elif 'learning_rates' in run.column_headings:
        learning_rate = run[0, 'learning_rates', -1]
    else:
        learning_rate = 'Unknown'
    sampler = dataloader.batch_sampler
    batch_size = sampler.batch_size
    print("Begining epochs:")
    column_names = ["epoch_reached", "time", "train_loss", "validaton_loss", "test_loss",
                    "mag_weights", "current_batch_size", "current_learning_rate", "weight_decay"]
    print(column_names)
    progress = [last_time, training_loss, validation_loss, test_loss, mag_weights, batch_size, learning_rate, weight_decay]
    print("Epoch; {}, {}".format(epoch_reached, progress))
    run.append(progress)
    return epoch_reached


def train(nets, run, dataloader, dataset, validation_sampler, device, train_losser, batch_losser, test_losser, optimiser, end_time, dataset_inv_size, val_schedulers, viewer=None):
    weight_decay = run.settings['weight_decay']
    sampler = dataloader.batch_sampler
    val_i = validation_sampler.validation_indices
    validation_events = [dataset[i] for i in val_i]
    test_events = dataset.test_events
    validation_losser = test_losser
    run.column_headings = ["time_stamps", "training_loss", "validation_loss", "test_loss", "mag_weights", "batch_size", "learning_rates", "weight_decay"]
    # Start working through the training epochs
    weight_decay = optimiser.param_groups[0]['weight_decay']
    epoch_reached = single_pass(nets, run, dataloader, validation_events, test_events, device, train_losser, validation_losser, test_losser, weight_decay)
    last_time = time.time()
    while last_time < end_time and os.path.exists("continue"):
        for net in nets:
            net.train()
        # start an epoch
        sum_loss = 0.
        # get a new iterator
        dataloader.reset()
        for i_batch, sample_batched in enumerate(dataloader):
            # reset the optimiser
            optimiser.zero_grad()  # zero the gradient buffer
            # forward pass
            loss = batch_losser(sample_batched, nets, device, train_losser)
            # backwards pass
            loss.backward()
            sum_loss += loss.item()
            # optimise weights according to gradient found by backpropigation
            optimiser.step()
            # remove from fast version ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
            if i_batch % 2 == 0:                                            #
                print('.', end='', flush=True)                              #
                if viewer is not None:                                      #
                    viewer(run, nets)                                       #
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        for net in nets:
            net.eval()
        # get the test training and validation loss
        validation_loss = 0
        for v_event in validation_events:
            validation_loss += validation_losser(v_event, nets, device).item()
        test_loss = 0
        for t_event in test_events:
            test_loss += test_losser(t_event, nets, device).item()
        training_loss = float(sum_loss) * dataset_inv_size
        # get new validation setup
        val_i = validation_sampler.validation_indices
        validation_events = [dataset[i] for i in val_i]
        # if this is the best seen so far, save it
        if test_loss < run.settings['lowest_loss']:
            stat_dicts = [pickle.loads(pickle.dumps(n.state_dict()))
                          for n in nets]
            run.set_best_state_dicts(stat_dicts, test_loss)
        # look at current learning rate
        learning_rate = float(optimiser.param_groups[0]['lr'])
        batch_size = sampler.batch_size
        weight_decay = float(optimiser.param_groups[0]['weight_decay'])
        last_time = time.time()
        # weights
        mag_weights = 0
        for net in nets:
            mag_weights += np.sum([torch.sum(torch.abs(w)).item()
                                   for w in net.get_weights()])
        progress = [last_time, training_loss, validation_loss, test_loss, mag_weights, batch_size, learning_rate, weight_decay]
        # remove from fast version ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        print("Epoch; {}, {}".format(epoch_reached, progress))               #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        epoch_reached += 1 
        run.append(progress)
        # SHEDULERS updating ~~~~~~~~~~~~~~~~
        for scheduler in val_schedulers:
            # check to see if the weight decay should scale
            scheduler.step(validation_loss, epoch_reached)
    print("Finishing...")
    return nets, run

