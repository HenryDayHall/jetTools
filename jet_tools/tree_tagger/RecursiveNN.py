#from ipdb import set_trace as st
import pickle
import time
import torch
import torch.nn as nn
import torch.nn.functional
from torch.utils.data import WeightedRandomSampler, BatchSampler
from jet_tools.tree_tagger import CustomDataloader, CustomScheduler, CustomSampler, TrainingTools
# from torch.nn.utils import clip_grad_norm  this looks useful!

class SimpleRecursor(nn.Module):
    """ """
    def __init__(self, device, num_jet_feaures, latent_dimensions=100, num_classes=1):
        super().__init__()
        self.device = device
        # maps jet features to latent space
        self.embedding = nn.Linear(num_jet_feaures, latent_dimensions)
        # this is the network that combines a left and a right
        self.combine = nn.Linear(2*latent_dimensions, latent_dimensions)
        # this makes the node prediction, in simple case called once at end
        self.projection = nn.Linear(latent_dimensions, num_classes)
        # activation for internal or external nodes
        self.activation = torch.nn.functional.relu
        #self.node_preds = [] # prediction, don't use in simple case
        #self.node_labels = [] # truth values, don't use in simple case

    def traverse(self, node):
        """
        The recusive call
        will go down the net from the given point and calculate correct memories

        Parameters
        ----------
        node :
            

        Returns
        -------

        
        """
        if node.is_leaf:
            # assume the dataset transforms to a sutable tensor
            # we need to put it on the device
            # this is not done till now so that device memory can be handeled flexably
            leaf = node.leaf_inputs.to(self.device)
            hidden_state = self.activation(self.embedding(leaf))
        # recursive call for tree traversal
        # the left and the right hand sides are combined and the result put through W
        else: 
            hidden_state = self.activation(
                             self.combine(
                             torch.cat(
                                 (self.traverse(node.left),
                                  self.traverse(node.right)),0)))
        # all the recursive calls are made by now
        # in a more complext net we would ad the intermediat loss
        # self.node_preds.append(self.projection(currentNode))
        # self.node_labels.append(torch.LongTensor([node.label()]))
        return hidden_state

    def forward(self, root_node):
        """
        

        Parameters
        ----------
        root_node :
            

        Returns
        -------

        
        """
        # the recursion is inside transverse, soo presumably this gets a root node in
        final_hidden = self.traverse(root_node)
        final_out = self.projection(final_hidden)
        # returns the predictions
        return final_out
    
    def get_weights(self):
        """ """
        weights = [self.embedding.weight.data,
                   self.combine.weight.data,
                   self.projection.weight.data]
        return weights
    
    def get_bias(self):
        """ """
        bias = [self.embedding.bias.data,
                self.combine.bias.data,
                self.projection.bias.data]
        return bias


class PseudoStateRecursor(nn.Module):
    """ """
    def __init__(self, device, num_jet_feaures, latent_dimensions=100, num_classes=1):
        super().__init__()
        self.device = device
        # maps jet features to latent space
        self.embedding = nn.Linear(num_jet_feaures, latent_dimensions)
        # this is the network that combines a left and a right and an input
        self.combine = nn.Linear(3*latent_dimensions, latent_dimensions)
        # this makes the node prediction
        self.projection = nn.Linear(latent_dimensions, num_classes)
        # activation for internal or external nodes
        self.activation = torch.nn.functional.relu
        #self.node_preds = []
        #self.node_labels = []

    def traverse(self, node):
        """
        The recusive call
        will go down the net from the given point and calculate correct memories

        Parameters
        ----------
        node :
            

        Returns
        -------

        
        """
        leaf = node.leaf_inputs.to(self.device)
        embedded = self.activation(self.embedding(leaf))
        if node.is_leaf:
            # assume the dataset transforms to a sutable tensor
            # we need to put it on the device
            # this is not done till now so that device memory can be handeled flexably
            hidden_state = embedded
        # recursive call for tree traversal
        # the left and the right hand sides plus leaf are combined and the result put through W
        else: 
            hidden_state = self.activation(
                             self.combine(
                             torch.cat(
                                 (self.traverse(node.left),
                                  self.traverse(node.right),
                                  embedded),0)))
        # all the recursive calls are made by now
        # in a more complext net we would ad the intermediat loss
        # self.node_preds.append(self.projection(hidden_state))
        # self.node_labels.append(torch.LongTensor([node.label()]))
        return hidden_state

    def forward(self, root_node):
        """
        

        Parameters
        ----------
        root_node :
            

        Returns
        -------

        
        """
        # the recursion is inside transverse, soo presumably this gets a root node in
        final_hidden = self.traverse(root_node)
        final_out = self.projection(final_hidden)
        # returns the predictions
        return final_out
    
    def get_weights(self):
        """ """
        weights = [self.embedding.weight.data,
                   self.combine.weight.data,
                   self.projection.weight.data]
        return weights
    
    def get_bias(self):
        """ """
        bias = [self.embedding.bias.data,
                self.combine.bias.data,
                self.projection.bias.data]
        return bias


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
    assert 'recursive' in run.settings['net_type'].lower();
    # create the dataset
    dataset = run.dataset
    criterion = nn.BCEWithLogitsLoss()
    # create the lossers (which get the loss)
    def train_losser(data, nets, device):
        """
        

        Parameters
        ----------
        data :
            param nets:
        device :
            
        nets :
            

        Returns
        -------

        
        """
        truth, root_node = data
        truth = torch.DoubleTensor(truth).to(device)
        output = nets[0].forward(root_node)
        loss = criterion(output, truth)
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

    test_losser = train_losser
    latent_dimension = run.settings['latent_dimension']
    # select the net by the net_type
    if run.settings['net_type'] == 'simple_recursive':
        nets = [SimpleRecursor(device, dataset.num_dimensions, latent_dimension, dataset.num_targets)]
    elif run.settings['net_type'] == 'pseudostate_recursive':
        nets = [PseudoStateRecursor(device, dataset.num_dimensions, latent_dimension, dataset.num_targets)]
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

