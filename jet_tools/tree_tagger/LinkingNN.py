import sys
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, BatchSampler, RandomSampler
import numpy as np
#from ipdb import set_trace as st
from tree_tagger import CustomSampler, Datasets, CustomScheduler, LinkingFramework, CustomDataloader, InputTools, TrainingTools


class Latent_projector(nn.Sequential):
    """A neural network to project an object into the latent space"""
    def __init__(self, input_size, latent_dimension, leaky_gradient=0.1):
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
                  nn.Sigmoid()]
        super(Latent_projector, self).__init__(*layers)
        self.layers = [l for l in layers if hasattr(l, 'weight')]
    
    def get_weights(self):
        """ """
        weights = []
        for layer in self.layers:
            weights.append(layer.weight.data)
        return weights
    
    def get_bias(self):
        """ """
        bias = []
        for layer in self.layers:
            bias.append(layer.bias.data)
        return bias

def soft_truth_criterion(towers_projection, tracks_projection, proximities, MC_truth):
    """
    

    Parameters
    ----------
    towers_projection :
        param tracks_projection:
    proximities :
        param MC_truth:
    tracks_projection :
        
    MC_truth :
        

    Returns
    -------

    
    """
    loss = 0
    # get the closeast tower fro each track
    np_tracks_proj = np.array([t.cpu().detach().numpy() for t in tracks_projection])
    np_towers_proj = np.array([t.cpu().detach().numpy() for t in towers_projection])
    closest_tower = LinkingFramework.high_dim_proximity(np_tracks_proj, np_towers_proj)
    towers_mask = np.ones(len(towers_projection), dtype=int)
    for closest_n, (track_n, tower_n) in zip(closest_tower, MC_truth.items()):
        track_p = tracks_projection[track_n]
        towers_mask[:] = 1
        if tower_n is not None:
            # penalise for distance to mc truth partner
            tower_p = towers_projection[tower_n]
            loss += torch.sum((track_p - tower_p)**2)
            towers_mask[tower_n] = 0
        # repell all other towers
        loss -= torch.sum(torch.sqrt(torch.sum((track_p - towers_projection[towers_mask])**2, dim=1)))
        #loss += 1./torch.clamp(torch.sum((track_p - close_p)**2),
        #                                  0., 1.) # but we dont care is more than 1 away
    return loss

# this should allow for some of the tracks never making towers?
def old_prox_criterion(towers_projection, tracks_projection, proximities, MC_truth):
    """
    

    Parameters
    ----------
    towers_projection :
        param tracks_projection:
    proximities :
        param MC_truth:
    tracks_projection :
        
    MC_truth :
        

    Returns
    -------

    
    """
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

def a_truth_criterion(towers_projection, tracks_projection, proximities, MC_truth):
    """
    

    Parameters
    ----------
    towers_projection :
        param tracks_projection:
    proximities :
        param MC_truth:
    tracks_projection :
        
    MC_truth :
        

    Returns
    -------

    
    """
    loss = 0
    # get the closeast tower fro each track
    #loss = torch.sum(towers_projection**2) + torch.sum(tracks_projection**2)
    loss = torch.sum(tracks_projection**2)
    return loss

def truth_criterion(towers_projection, tracks_projection, proximities, MC_truth):
    """
    

    Parameters
    ----------
    towers_projection :
        param tracks_projection:
    proximities :
        param MC_truth:
    tracks_projection :
        
    MC_truth :
        

    Returns
    -------

    
    """
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
def a_prox_criterion(towers_projection, tracks_projection, proximities, MC_truth):
    """
    

    Parameters
    ----------
    towers_projection :
        param tracks_projection:
    proximities :
        param MC_truth:
    tracks_projection :
        
    MC_truth :
        

    Returns
    -------

    
    """
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


def begin_training(run, viewer=None):
    """
    

    Parameters
    ----------
    run :
        param viewer: (Default value = None)
    viewer :
        (Default value = None)

    Returns
    -------

    
    """
    torch.set_default_tensor_type('torch.DoubleTensor')
    end_time = run.settings['time'] + time.time()
    # Device configuration
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    assert run.settings['net_type'] == "tracktower_projectors"
    # create the dataset
    #dataset = Datasets.TracksTowersDataset(folder_name=run.settings['data_folder'])
    dataset = run.dataset
    def test_losser(event_data, nets, device):
        """
        

        Parameters
        ----------
        event_data :
            param nets:
        device :
            
        nets :
            

        Returns
        -------

        
        """
        towers_data, tracks_data, proximities, MC_truth = event_data
        tower_net, track_net = nets
        towers_data = towers_data.to(device)
        towers_projection = tower_net(towers_data)
        tracks_data = tracks_data.to(device)
        tracks_projection = track_net(tracks_data)
        loss = truth_criterion(towers_projection, tracks_projection, proximities, MC_truth)
        return loss

    def train_losser(event_data, nets, device):
        """
        

        Parameters
        ----------
        event_data :
            param nets:
        device :
            
        nets :
            

        Returns
        -------

        
        """
        towers_data, tracks_data, proximities, MC_truth = event_data
        tower_net, track_net = nets
        towers_data = towers_data.to(device)
        towers_projection = tower_net(towers_data)
        tracks_data = tracks_data.to(device)
        tracks_projection = track_net(tracks_data)
        loss = soft_truth_criterion(towers_projection, tracks_projection, proximities, MC_truth)
        return loss

    def batch_losser(events_data, nets, device, losser):
        """
        

        Parameters
        ----------
        events_data :
            param nets:
        device :
            param losser:
        nets :
            
        losser :
            

        Returns
        -------

        
        """
        losses = [losser(e_data, nets, device) for e_data in events_data]
        return sum(losses)

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
            """
            

            Parameters
            ----------
            m :
                

            Returns
            -------

            
            """
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform_(m.weight, gain=0.5)
                m.bias.data.fill_(0.01)
        net.apply(init_weights)
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

    nets, run = TrainingTools.train(nets, run, dataloader, dataset, validation_sampler,
                                    device, train_losser, batch_losser, test_losser, optimiser, end_time,
                                    dataset_inv_size, val_schedulers, viewer)
    run.last_nets = [net.state_dict() for net in nets]
    return run
