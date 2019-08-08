import sys
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, BatchSampler, RandomSampler
import numpy as np
from ipdb import set_trace as st
from tree_tagger import CustomSampler, Datasets, CustomScheduler, LinkingFramework, CustomDataloader, InputTools


class Latent_projector(nn.Sequential):
    """A neural network to project an object into the latent space"""
    def __init__(self, input_size, latent_dimension, leaky_gradient=0.5):
        """Initilzation for the net. Creates the layer that will be used.

        Parameters
        ----------
        input_size : int
            the number of nodes at the input layer. This must equal the number
            of variables in each data point.
        latent_dimension : int
            number of nodes at the output layer.

        """
        hidden_size = latent_dimension
        layers = [nn.Linear(input_size, hidden_size),
                  nn.LeakyReLU(leaky_gradient),
                  nn.Linear(hidden_size, hidden_size),
                  nn.LeakyReLU(leaky_gradient),
                  nn.Linear(hidden_size, latent_dimension),
                  nn.LeakyReLU(leaky_gradient)]
        super(Latent_projector, self).__init__(*layers)
        self.layers = [l for l in layers if hasattr(l, 'weight')]
    
    def get_weights(self):
        weights = []
        for layer in self.layers:
            weights.append(layer.weight.data)
        return weights
    
    def get_bias(self):
        bias = []
        for layer in self.layers:
            bias.append(layer.bias.data)
        return bias


def forward(event_data, nets, criterion, device):
    towers_data, tracks_data, proximities, MC_truth = event_data
    tower_net, track_net = nets
    towers_data = towers_data.to(device)
    towers_projection = tower_net(towers_data)
    tracks_data = tracks_data.to(device)
    tracks_projection = track_net(tracks_data)
    # towers_projection, tracks_projection = [], []
    # for tower_d in towers_data:
    #     tower_d = tower_d.to(device)
    #     towers_projection.append(tower_net(tower_d))
    # for track_d in tracks_data:
    #     track_d = track_d.to(device)
    #     tracks_projection.append(track_net(track_d))
    loss = criterion(towers_projection, tracks_projection, proximities, MC_truth)
    return loss


def batch_forward(events_data, nets, criterion, device):
    losses = [forward(e_data, nets, criterion, device) for e_data in events_data]
    return sum(losses)


def old_truth_criterion(towers_projection, tracks_projection, proximities, MC_truth):
    loss = 0
    # get the closeast tower fro each track
    np_tracks_proj = np.array([t.cpu().detach().numpy() for t in tracks_projection])
    np_towers_proj = np.array([t.cpu().detach().numpy() for t in towers_projection])
    closest_tower = LinkingFramework.high_dim_proximity(np_tracks_proj, np_towers_proj)
    for closest_n, (track_n, tower_n) in zip(closest_tower, MC_truth.items()):
        track_p = tracks_projection[track_n]
        if tower_n is not None:
            # penalise for distance to mc truth partner
            tower_p = towers_projection[tower_n]
            loss += torch.sum((track_p - tower_p)**2)
        # add a penalty if the closes tower is not the MC truth partner
        if closest_n != tower_n:
            close_p = towers_projection[closest_n]
            loss += torch.sum((track_p - close_p)**2)
            #loss += 1./torch.clamp(torch.sum((track_p - close_p)**2),
            #                                  0., 1.) # but we dont care is more than 1 away
    return loss

# this should allow for some of the tracks never making towers?
def old_prox_criterion(towers_projection, tracks_projection, proximities, MC_truth):
    loss = 0
    mask = np.ones(len(towers_projection), dtype=int)
    for track_n, tower_indices in enumerate(proximities):
        track_p = tracks_projection[track_n]
        # pull towards towers in the sector
        distances2 = []
        for tower_n in tower_indices:
            tower_p = towers_projection[tower_n]
            loss_here = (track_p - tower_p)**2
            static_sum = np.sum(loss_here.cpu().detach().numpy())
            distances2.append(static_sum)
            loss += torch.sum(loss_here)/static_sum
        # push away from other towers
        mask[:] = 1
        mask[tower_indices] = 0
        # this distane defind the standadrd behavior of other tracks
        distance = float(np.sqrt(np.median(distances2)))
        factor = float(np.sum(mask)) * distance
        loss += factor/torch.sum(
                       torch.clamp(
                       torch.sum((track_p - towers_projection[mask])**2, dim=1),
                                   0, distance))  # dont care when more than distance away
    return loss


def truth_criterion(towers_projection, tracks_projection, proximities, MC_truth):
    loss = 0
    # get the closeast tower fro each track
    np_tracks_proj = np.array([t.cpu().detach().numpy() for t in tracks_projection])
    np_towers_proj = np.array([t.cpu().detach().numpy() for t in towers_projection])
    closest_tower = LinkingFramework.high_dim_proximity(np_tracks_proj, np_towers_proj)
    for closest_n, (track_n, tower_n) in zip(closest_tower, MC_truth.items()):
        track_p = tracks_projection[track_n]
        if tower_n is not None:
            # penalise for distance to mc truth partner
            tower_p = towers_projection[tower_n]
            loss += torch.sum((track_p - tower_p)**2)
        # add a penalty if the closes tower is not the MC truth partner
        if closest_n != tower_n:
            close_p = towers_projection[closest_n]
            loss += 1./torch.clamp(torch.sum((track_p - close_p)**2),
                                              0.001, 1.) # but we dont care is more than 1 away
    return loss

# this should allow for some of the tracks never making towers?
def prox_criterion(towers_projection, tracks_projection, proximities, MC_truth):
    loss = 0
    track_distance = 0
    tower_mask = np.ones(len(towers_projection), dtype=int)
    track_mask = np.ones(len(tracks_projection), dtype=int)
    for track_n, tower_indices in enumerate(proximities):
        track_p = tracks_projection[track_n]
        # pull towards towers in the sector
        distances2 = []
        for tower_n in tower_indices:
            tower_p = towers_projection[tower_n]
            loss_here = (track_p - tower_p)**2
            static_sum = np.sum(loss_here.cpu().detach().numpy())
            distances2.append(static_sum)
            loss += torch.sum(loss_here)/static_sum
        # push away from other towers
        tower_mask[:] = 1
        tower_mask[tower_indices] = 0
        # this distane defind the standadrd behavior of other tracks
        median_distance2 = float(np.median(distances2))
        factor = float(np.sum(tower_mask)) * median_distance2
        loss += factor*torch.sum(
                       torch.rsqrt(  # inverse square root
                       torch.clamp(
                       torch.sum((track_p - towers_projection[tower_mask])**2, dim=1),
                                   0, median_distance2)))  # dont care when more than distance away
        # push away from other tracks
        track_mask[:] = 1
        track_mask[track_n] = 0
        track_distance += 0.5*torch.sum(
                           torch.rsqrt(  # inverse square root
                           torch.clamp(
                           torch.sum((track_p - tracks_projection[track_mask])**2, dim=1),
                                   0, median_distance2)))  # dont care when more than distance away
    print(f"loss = {float(loss.cpu().detach().numpy())}")
    print(f"track_distance = {float(track_distance.cpu().detach().numpy())}")
    time.sleep(0.1)
    return loss


def single_pass(nets, run, dataloader, validation_events, test_events, device, criterion, validation_criterion, test_criterion, weight_decay):
    # make initial measures
    for net in nets:
        net.eval()
    last_time = time.time()
    epoch_reached = len(run)
    # need one batch to get an loss
    event_data = next(dataloader.__iter__())[0] # only want one event
    # find loss
    training_loss = forward(event_data, nets, criterion, device).item()
    # validation loss
    validation_loss = 0
    for v_event in validation_events:
        validation_loss += forward(v_event, nets, validation_criterion, device).item()
    # test loss
    test_loss = 0
    for t_event in test_events:
        test_loss += forward(t_event, nets, test_criterion, device).item()
    # if the run starts empty record a new best and last net
    state_dicts = [net.state_dict() for net in nets]
    if run.empty_run:
        run.set_best_state_dicts(state_dicts, test_loss)
        run.last_net = state_dicts
    elif test_loss < run.settings['lowest_loss']:
        run.set_best_state_dicts(state_dicts, test_loss)
    # weights
    mag_weights = 0
    for net in nets:
        mag_weights += np.sum([float(torch.sum(torch.abs(w)))
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


def train(nets, run, dataloader, dataset, validation_sampler, device, criterion, test_criterion, optimiser, end_time, dataset_inv_size, val_schedulers, viewer=None):
    weight_decay = run.settings['weight_decay']
    sampler = dataloader.batch_sampler
    val_i = validation_sampler.validation_indices
    validation_events = [dataset[i] for i in val_i]
    test_events = dataset.test_events
    validation_criterion = test_criterion
    run.column_headings = ["time_stamps", "training_loss", "validation_loss", "test_loss", "mag_weights", "batch_size", "learning_rates", "weight_decay"]
    # Start working through the training epochs
    weight_decay = optimiser.param_groups[0]['weight_decay']
    epoch_reached = single_pass(nets, run, dataloader, validation_events, test_events, device, criterion, validation_criterion, test_criterion, weight_decay)
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
            loss = batch_forward(sample_batched, nets, criterion, device)
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
            validation_loss += forward(v_event, nets, validation_criterion, device).item()
        test_loss = 0
        for t_event in test_events:
            test_loss += forward(t_event, nets, test_criterion, device).item()
        training_loss = float(sum_loss) * dataset_inv_size
        # get new validation setup
        val_i = validation_sampler.validation_indices
        validation_events = [dataset[i] for i in val_i]
        # if this is the best seen so far, save it
        if test_loss < run.settings['lowest_loss']:
            stat_dicts = [n.state_dict() for n in nets]
            run.set_best_state_dicts(stat_dicts, test_loss)
        # look at current learning rate
        learning_rate = float(optimiser.param_groups[0]['lr'])
        batch_size = sampler.batch_size
        weight_decay = float(optimiser.param_groups[0]['weight_decay'])
        last_time = time.time()
        # weights
        mag_weights = 0
        for net in nets:
            mag_weights += np.sum([float(torch.sum(torch.abs(w)))
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


def begin_training(run, viewer=None):
    torch.set_default_tensor_type('torch.DoubleTensor')
    end_time = run.settings['time'] + time.time()
    # Device configuration
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    assert run.settings['net_type'] == "tracktower_projectors"
    # create the dataset
    dataset = Datasets.TracksTowersDataset(folder_name=run.settings['data_folder'])
    criterion = prox_criterion
    latent_dimension = run.settings['latent_dimension']
    nets = [Latent_projector(dataset.tower_dimensions, latent_dimension),
            Latent_projector(dataset.track_dimensions, latent_dimension)]
    # if the run is not empty there should be a previous net, load that
    if not run.empty_run:
        nets = run.last_nets
    # finish initilising the nets
    for net in nets:
        net = net.to(device)
        # Experimental!
        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform_(m.weight, gain=0.5)
                m.bias.data.fill_(0.01)
        net.apply(init_weights)
    test_criterion = truth_criterion
    # the nature of the data loader depends if we need to reweight
    sampler = RandomSampler(dataset)
    # this sampler then gets put inside a sampler that can split the data into a validation set
    validation_sampler = CustomSampler.ValidationRandomSampler(sampler, n_folds=3)
    # this gets wrapped in a batch sampler so the sample size can change
    batch_sampler = BatchSampler(validation_sampler, run.settings['batch_size'], False)
    dataloader = CustomDataloader.ArbitaryDataloader(dataset, batch_sampler=batch_sampler)
    dataset_inv_size = 1./len(dataset)
    # create optimisers and an adaptive learning rate
    optimisers = []
    for net in nets:
        optimiser = torch.optim.SGD(net.parameters(), lr=run.settings['inital_lr'],
                                     weight_decay=run.settings['weight_decay'])
        #lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, cooldown=3)
        bs_scheduler = CustomScheduler.ReduceBatchSizeOnPlateau(batch_sampler, cooldown=3)
        #val_schedulers = [lr_scheduler, bs_scheduler]
        val_schedulers = [bs_scheduler]

    nets, run = train(nets, run, dataloader, dataset, validation_sampler, device, criterion, test_criterion, optimiser, end_time, dataset_inv_size, val_schedulers, viewer)
    run.last_nets = [net.state_dict() for net in nets]
    return run
