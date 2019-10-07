from ipdb import set_trace as st
import pickle
import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional
from torch.utils.data import WeightedRandomSampler, BatchSampler, RandomSampler
from tree_tagger import CustomDataloader, CustomScheduler, CustomSampler, TrainingTools
# from torch.nn.utils import clip_grad_norm  this looks useful!

class SimpleLinear(nn.Sequential):
    def __init__(self, device, num_jet_feaures, latent_dimensions=100, num_classes=1, layers=4):
        super().__init__()
        self.device = device
        # maps jet features to latent space
        self.embedding = nn.Linear(num_jet_feaures, latent_dimensions)
        # internal nn layer
        if layers > 2:
            self.internals = [nn.Linear(latent_dimensions, latent_dimensions)
                              for _ in range(layers-2)]
        else:
            self.internals = []
        # this makes the node prediction, in simple case called once at end
        self.projection = nn.Linear(latent_dimensions, num_classes)
        # activation for internal or external nodes
        self.activation = torch.nn.ReLU(True)
        self.all_layers = [self.embedding, *self.internals, self.projection]
        process = self.all_layers[:1]
        for layer in self.all_layers[1:]:
            process += [self.activation, layer]
        super().__init__(*process)
    
    def get_weights(self):
        weights = [layer.weight.data for layer in self.all_layers]
        return weights
    
    def get_bias(self):
        bias = [layer.bias.data for layer in self.all_layers]
        return bias


def begin_training(run, viewer=None):
    torch.set_default_tensor_type('torch.DoubleTensor')
    end_time = run.settings['time'] + time.time()
    # Device configuration
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    assert 'standard' in run.settings['net_type'].lower();
    # create the dataset
    dataset = run.dataset
    dataset.to_torch(device)
    criterion = nn.BCEWithLogitsLoss()
    # create the lossers (which get the loss)
    def train_losser(data, nets, device):
        truth, root_node = data
        truth = torch.DoubleTensor(truth).to(device)
        output = nets[0].forward(root_node)
        loss = criterion(output, truth)
        return loss


    def batch_losser(events_data, nets, device, losser):
        losses = [losser(e_data, nets, device) for e_data in events_data]
        return sum(losses)
    
    test_losser = train_losser
    latent_dimension = run.settings['latent_dimension']
    # if the run is not empty there should be a previous net, load that
    if not run.empty_run:
        nets = run.last_nets
    else:
        nets = [SimpleLinear(device, dataset.num_inputs, latent_dimension, dataset.num_targets)]
    # finish initilising the nets
    for net in nets:
        net = net.to(device)
        # Experimental!
        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform_(m.weight, gain=0.5)
                m.bias.data.fill_(0.01)
        net.apply(init_weights)
    # the nature of the data loader depends if we need to reweight
    if not hasattr(dataset, 'train_weights'):
        dataset.train_weights = np.ones(len(dataset))
    if not hasattr(dataset, 'test_weights'):
        dataset.test_weights = np.ones(len(dataset.test_truth))
    sampler = WeightedRandomSampler(dataset.train_weights, len(dataset))
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

    nets, run = TrainingTools.train(nets, run, dataloader, dataset, validation_sampler, device, train_losser, batch_losser, test_losser, optimiser, end_time, dataset_inv_size, val_schedulers, viewer)
    run.last_nets = [pickle.loads(pickle.dumps(net.state_dict()))
                     for net in nets]
    run.write()
    return run

