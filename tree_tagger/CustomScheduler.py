import types
import math
import torch
from torch._six import inf
from bisect import bisect_right
from functools import partial
from torch.utils.data import BatchSampler
from torch.optim.optimizer import Optimizer
from scipy.stats import spearmanr
import numpy as np
#from ipdb import set_trace as st


class ReduceBatchSizeOnPlateau(object):
    """
    Reduce batch size when a metric has stopped improving.
    Experimental variation of ReduceLROnPlateau.

    Parameters
    ----------
    optimizer : Optimizer
        Wrapped optimizer.
    mode : str
        One of `min`, `max`. In `min` mode, bs will
        be reduced when the quantity monitored has stopped
        decreasing; in `max` mode it will be reduced when the
        quantity monitored has stopped increasing. Default: 'min'.
    factor : float
        Factor by which the batch size will be
        reduced. new_bs = int(bs * factor). Default: 0.1.
    patience : int
        Number of epochs with no improvement after
        which batch size will be reduced. For example, if
        `patience = 2`, then we will ignore the first 2 epochs
        with no improvement, and will only decrease the LR after the
        3rd epoch if the loss still hasn't improved then.
        Default: 10.
    verbose : bool
        If ``True``, prints a message to stdout for
        each update. Default: ``False``.
    threshold : float
        Threshold for measuring the new optimum,
        to only focus on significant changes. Default: 1e-4.
    threshold_mode : str
        One of `rel`, `abs`. In `rel` mode,
        dynamic_threshold = best * ( 1 + threshold ) in 'max'
        mode or best * ( 1 - threshold ) in `min` mode.
        In `abs` mode, dynamic_threshold = best + threshold in
        `max` mode or best - threshold in `min` mode. Default: 'rel'.
    cooldown : int
        Number of epochs to wait before resuming
        normal operation after bs has been reduced. Default: 0.
    min_bs : float or list
        A scalar. A
        lower bound on the batch size.
        Default: 1.
    eps : float
        Minimal decay applied to bs. If the difference
        between new and old bs is smaller than eps, the update is
        ignored. Default: 1e-8.
        Example:

    Returns
    -------

    
    >>> batch_sampler = BatchSampler(sampler, batch_size=5, drop_last=False)
        >>> scheduler = ReduceBatchSizeOnPlateau(batch_sampler, 'min')
        >>> for epoch in range(10):
        >>>     train(...)
        >>>     val_loss = validate(...)
        >>>     # Note that step should be called after validate()
        >>>     scheduler.step(val_loss)
    """

    def __init__(self, sampler, mode='min', factor=0.1, patience=10,
                 verbose=False, threshold=1e-4, threshold_mode='rel',
                 cooldown=0, min_bs=1, eps=1e-8):

        if factor >= 1.0:
            raise ValueError('Factor should be < 1.0.')
        self.factor = factor

        if not isinstance(sampler, BatchSampler):
            raise TypeError('{} is not a BatchSampler'.format(
                type(sampler).__name__))
        self.sampler = sampler

        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.min_bs = min(min_bs, 1)
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.is_better = None
        self.eps = eps
        self.last_epoch = -1
        self._init_is_better(mode=mode, threshold=threshold,
                             threshold_mode=threshold_mode)
        self._reset()

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def step(self, metrics, epoch=None):
        """
        

        Parameters
        ----------
        metrics :
            param epoch: (Default value = None)
        epoch :
            (Default value = None)

        Returns
        -------

        
        """
        current = metrics
        if epoch is None:
            epoch = self.last_epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self._reduce_bs(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

    def _reduce_bs(self, epoch):
        """
        

        Parameters
        ----------
        epoch :
            

        Returns
        -------

        
        """
        old_bs = self.sampler.batch_size
        new_bs = int(max(old_bs * self.factor, self.min_bs))
        if old_bs - new_bs > self.eps:
            self.sampler.batch_size = new_bs
            if self.verbose:
                print('Epoch {:5d}: reducing batch size'
                      ' to {:.4e}.'.format(epoch, new_bs))

    @property
    def in_cooldown(self):
        """ """
        return self.cooldown_counter > 0

    def _cmp(self, mode, threshold_mode, threshold, a, best):
        """
        

        Parameters
        ----------
        mode :
            param threshold_mode:
        threshold :
            param a:
        best :
            
        threshold_mode :
            
        a :
            

        Returns
        -------

        
        """
        if mode == 'min' and threshold_mode == 'rel':
            rel_epsilon = 1. - threshold
            return a < best * rel_epsilon

        elif mode == 'min' and threshold_mode == 'abs':
            return a < best - threshold

        elif mode == 'max' and threshold_mode == 'rel':
            rel_epsilon = threshold + 1.
            return a > best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        """
        

        Parameters
        ----------
        mode :
            param threshold:
        threshold_mode :
            
        threshold :
            

        Returns
        -------

        
        """
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')

        if mode == 'min':
            self.mode_worse = inf
        else:  # mode == 'max':
            self.mode_worse = -inf

        self.is_better = partial(self._cmp, mode, threshold_mode, threshold)

    def state_dict(self):
        """ """
        return {key: value for key, value in self.__dict__.items() if key not in {'sampler', 'is_better'}}

    def load_state_dict(self, state_dict):
        """
        

        Parameters
        ----------
        state_dict :
            

        Returns
        -------

        
        """
        self.__dict__.update(state_dict)
        self._init_is_better(mode=self.mode, threshold=self.threshold, threshold_mode=self.threshold_mode)


class CammieWeightDecay(object):
    """
    Complexity Abitrated Moderation Method Improving Extrapolation
    for weight decay.
    
    This method consideres 3 states;
    1. Breaking ground; the validation loss is the lowest its been and falling
    2. Redigging; the validation loss is not the lowest it has been and falling
    3. Bedrock plateau; the validation loss is the lowest it has been and static
    4. False plateau; the validation loss is not the lowest it has been and static
    5. Retreating; the validation loss is rising.
    States 1 and 3 have indervidual moderation reduction percentages,
    so that the net is allowed to become more complex when in these phases.
    The last known state 1 or 3 moderation level is knonw as relaxed-mod
    State 5 has a moderation enhancement percentage,
    to pull it back to a more general model.
    State 2 has steady moderation and relax_mod level.
    The last known state 5 moderation is known as stern_mod.
    State 4 has occilating moderation levels between relax_mod and stern_mod to try and break the false plateau.
    
    The metric is in plateau if either the standard devation of the point is in the tollarance
    if the spearmans rank is under rank_min

    Parameters
    ----------
    optimizer : Optimizer
        Wrapped optimizer.
    mode : str
        One of `min`, `max`. In `min` mode, bs will
        be reduced when the quantity monitored has stopped
        decreasing; in `max` mode it will be reduced when the
        quantity monitored has stopped increasing. Default: 'min'.
    breaking_reduction : float
        Percentage by which the
        weight decay should be reduced when breaking ground.
        Default: 0.05
    bedrock_reduction : float
        Percentage by which the
        weight decay should be reduced when at a bedrock plateau.
        Default: 0.1
    retreating_enchancement : float
        Percentage by which
        the weight_decay should be enhanced when retreating.
        Default: 0.1
    verbose : bool
        If ``True``, prints a message to stdout for
        each update. Default: ``False``.
    threshold : float
        Threshold for measuring the new optimum,
        to only focus on significant changes. Default: 1e-4.
    threshold_mode : str
        One of `rel`, `abs`. In `rel` mode,
        dynamic_threshold = best * ( 1 + threshold ) in 'max'
        mode or best * ( 1 - threshold ) in `min` mode.
        In `abs` mode, dynamic_threshold = best + threshold in
        `max` mode or best - threshold in `min` mode. Default: 'rel'.
    cooldown : int
        Number of epochs to wait before resuming
        normal operation after weight_decay has been reduced.
        Must be at least 2. Default: 3.
    min_wd : float or list
        A scalar. A
        lower bound on the weight decay.
        Default: 0.
    max_wd : float or list
        A scalar. A
        upper bound on the weight decay.
        Default: 1.
    eps : float
        Minimal decay applied to bs. If the difference
        between new and old bs is smaller than eps, the update is
        ignored. Default: 1e-8.
        Example:

    Returns
    -------

    
    >>> batch_sampler = BatchSampler(sampler, batch_size=5, drop_last=False)
        >>> scheduler = ReduceBatchSizeOnPlateau(batch_sampler, 'min')
        >>> for epoch in range(10):
        >>>     train(...)
        >>>     val_loss = validate(...)
        >>>     # Note that step should be called after validate()
        >>>     scheduler.step(val_loss)
    """

    def __init__(self, optimizer, mode='min', breaking_reduction=0.02,
                 bedrock_reduction=0.05, retreating_enchancement=0.05,
                 verbose=False, threshold=1e-2, threshold_mode='rel',
                 cooldown=5, min_wd=0., max_wd=1., rank_min=0.1):

        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not a BatchSampler'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        self.current_wd = float(self.optimizer.param_groups[0]['weight_decay'])
        self.mode = mode
        self.breaking_multiplier = 1 - breaking_reduction
        self.bedrock_multiplier = 1 - bedrock_reduction
        self.retreating_multipler = 1 + retreating_enchancement
        self.verbose = verbose
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown
        self.path = [self.current_wd] * (self.cooldown + 1)
        self.cooldown_counter = 0
        self.min_wd = max(min_wd, 0)
        self.max_wd = max(max_wd, 1)
        self.relax_mod = self.stern_mod = float(optimizer.param_groups[0]['weight_decay'])
        self.at_best = True
        self.mode_worse = None  # the worse value for the chosen mode
        self.best = None
        self.set_best = None
        self.state = 1
        self.rank_min = rank_min
        self.last_epoch = -1
        self._reset()
        self._init_set_best(mode, threshold, threshold_mode)

    def get_state(self):
        """ """
        if self.at_best:
            if self.motion == 1:
                state = 1  # breaking ground
            elif self.motion == 0:
                state = 3  # bedrock plateau
            else:  # self.motion = -1
                print(f"Impossible combination; self.at_best = {self.at_best} and self.motion = {self.motion}"
                      "Will force to state 3")
                state = 3
        else:
            if self.motion == 1:
                state = 2  # redigging
            elif self.motion == 0:
                state = 4  # false plateau
            else:  # self.motion = -1
                state = 5  # retreating
        return state

    @property
    def motion(self):
        """ """
        if len(self.recent) < self.cooldown:
            # don't know yet
            raise RuntimeError("mption requested before cooldown period finished.")
        elif self.threshold_mode == 'rel':
            comp = self.threshold * np.mean(self.recent)
        else:
            comp = self.threshold
        if np.std(self.recent) < comp:
            # if was at best befroe then still at best
            return 0
        # from now on the rank is the only factor
        rank = spearmanr(range(self.cooldown + 1), self.recent).correlation
        if abs(rank) < self.rank_min:
            # if was at best befroe then still at best
            return 0
        elif self.mode == 'max':
            return np.sign(rank)
        else:  # self.mode == 'max':
            return -np.sign(rank)
        

    def _reset(self):
        """Resets recent and cooldown counter."""
        self.recent = []
        self.cooldown_counter = self.cooldown

    def _chose_best(self, mode, threshold_mode, threshold, contender):
        """
        

        Parameters
        ----------
        mode :
            param threshold_mode:
        threshold :
            param contender:
        threshold_mode :
            
        contender :
            

        Returns
        -------

        
        """
        if mode == 'min' and threshold_mode == 'rel':
            rel_epsilon = 1. - threshold
            improved = self.best > contender * rel_epsilon
            still_best = self.best > contender / rel_epsilon

        elif mode == 'min' and threshold_mode == 'abs':
            improved =  self.best > contender - threshold
            still_best = self.best > contender + rel_epsilon

        elif mode == 'max' and threshold_mode == 'rel':
            rel_epsilon = threshold + 1.
            improved =  self.best < contender * rel_epsilon
            still_best = self.best < contender / rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            improved =  self.best < contender + threshold
            still_best = self.best < contender - rel_epsilon
        if improved:
            self.best = contender
        if still_best:
            self.at_best = True
        else:
            self.at_best = False

    def _init_set_best(self, mode, threshold, threshold_mode):
        """
        

        Parameters
        ----------
        mode :
            param threshold:
        threshold_mode :
            
        threshold :
            

        Returns
        -------

        
        """
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')

        if mode == 'min':
            self.mode_worse = inf
        else:  # mode == 'max':
            self.mode_worse = -inf
        self.best = self.mode_worse

        self.set_best = partial(self._chose_best, mode, threshold_mode, threshold)

    def step(self, metric, epoch=None):
        """
        

        Parameters
        ----------
        metric :
            param epoch: (Default value = None)
        epoch :
            (Default value = None)

        Returns
        -------

        
        """
        if epoch is None:
            epoch = self.last_epoch = self.last_epoch + 1
        self.last_epoch = epoch

        self.recent.append(metric)
        if self.in_cooldown:
            self.cooldown_counter -= 1
            self._increment()
        else:
            self.set_best(np.mean(self.recent))
            if self.state in (1, 3):
                self.relax_mod = self.current_wd
            elif self.state == 5:
                self.stern_mod = self.current_wd
            self._set_path()
            self._reset()
            self._increment()

    def _set_path(self):
        """ """
        self.state = self.get_state()
        if self.verbose:
            print(f"Entering state {self.state}")
        if self.state == 1:  # breaking ground
            self.path = [self.current_wd * self.breaking_multiplier**(i)
                         for i in range(0, self.cooldown + 1)]
        elif self.state == 2: # redigging
            self.path = [self.relax_mod] * (self.cooldown + 1)
        elif self.state == 3: # bedrock plateau
            self.path = [self.current_wd * self.bedrock_multiplier**(i)
                         for i in range(0, self.cooldown + 1)]
        elif self.state == 4: # false plateau
            self.path = [0.5*(np.sin(p) + 1)*(self.stern_mod - self.relax_mod) + self.relax_mod
                         for p in np.linspace(0, 2*np.pi, self.cooldown + 1)]
        elif self.state == 5: # retreating
            self.path = [self.current_wd * self.retreating_multipler**(i)
                         for i in range(0, self.cooldown + 1)]
        # now check that the path falls between max and min
        self.path = [min(max(p, self.min_wd), self.max_wd)
                     for p in self.path]

    def _increment(self):
        """ """
        next_mod = self.path.pop(0)
        self.optimizer.param_groups[0]['weight_decay'] = next_mod
        self.current_wd = next_mod

            
    @property
    def in_cooldown(self):
        """ """
        return self.cooldown_counter > 0

    def state_dict(self):
        """ """
        return {key: value for key, value in self.__dict__.items() if key not in {'optimizer', 'is_better'}}

    def load_state_dict(self, state_dict):
        """
        

        Parameters
        ----------
        state_dict :
            

        Returns
        -------

        
        """
        self.__dict__.update(state_dict)
