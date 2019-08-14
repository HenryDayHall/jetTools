from ipdb import set_trace as st
import pickle
import time
import torch
import torch.nn as nn
import torch.nn.functional
# from torch.nn.utils import clip_grad_norm  this looks useful!

class SimpleRecursor(nn.Module):
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
        """ The recusive call
        will go down the net from the given point and calculate correct memories """
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
        # the recursion is inside transverse, soo presumably this gets a root node in
        final_hidden = self.traverse(root_node)
        final_out = self.projection(final_hidden)
        # returns the predictions
        return final_out

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
    criterion = nn.BCEWithLogitsLoss()
    # create the lossers (which get the loss)
    def train_losser(data, nets, device):
        truth, root_node = data
        output = nets[0].forward(root_node)
        loss = criterion(output, truth)
        return loss


    def batch_losser(events_data, nets, device, losser):
        losses = [losser(e_data, nets[0], criterion, device) for e_data in events_data]
        return sum(losses)

    test_losser = train_losser
    latent_dimension = run.settings['latent_dimension']
    nets = [SimpleRecursor(dataset.num_dimensions, latent_dimension)]
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

    nets, run = train(nets, run, dataloader, dataset, validation_sampler, device, train_losser, batch_losser, test_losser, optimiser, end_time, dataset_inv_size, val_schedulers, viewer)
    run.last_nets = [pickle.loads(pickle.dumps(net.state_dict()))
                     for net in nets]
    return run

