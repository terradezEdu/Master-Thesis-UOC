# -*- coding: utf-8 -*-
""" DDPG model. 

Implementation of the DDPG algorithm for the master's thesis in Data Science

author: Eduardo Terrádez
year: 2021

"""

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

PATH= '/content/drive/MyDrive/TFM/models/save'

class Actor(nn.Module):
    """
    Maps states to actions
    """
    

    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, name='actor', save_dir=PATH):
        super(Actor, self).__init__()
        self.actor1= nn.Linear(input_size[0], hidden_size1)
        self.actor2= nn.Linear(hidden_size1, hidden_size2)
        self.actor3= nn.Linear(hidden_size2, output_size)
        self.name = name
        self.save_dir= save_dir
        self.save_file= os.path.join(self.save_dir, name+ '_ddpg')
    
    def forward(self, state):
        """
        state is a torch tensor
        """
        x= F.relu(self.actor1(state))
        x= F.relu(self.actor2(x))
        x= T.tanh(self.actor3(x))

        return x

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.save_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.save_file))

class Critic(nn.Module):
    """
    Evaluates the actor performances
    """
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, n_action,
         name='critic', save_dir=PATH):
        super(Critic, self).__init__()
        self.critic1= nn.Linear(input_size[0] + n_action, hidden_size1)
        self.critic2= nn.Linear(hidden_size1, hidden_size2)
        self.critic3= nn.Linear(hidden_size2, output_size)
        self.name = name
        self.save_dir= save_dir
        self.save_file= os.path.join(self.save_dir, name+ '_ddpg')
        
    def forward(self, state, action):
        """
        state and actions are torch tensors
        """
        x= T.cat([state, action], 1)
        x= F.relu(self.critic1(x))
        x= F.relu(self.critic2(x))
        x= self.critic3(x)

        return x

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.save_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.save_file))
        
class Orn_Uhlen:
    """
    Adds noise to the actions taken by the deterministic policy.
    The OU process satisfies the following stochastic differential equation:
    dxt = theta*(mu - xt)*dt + sigma*dWt
    where Wt denotes the Wiener process
    
     Based on: 
     https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
    
    """
    def __init__(self, action_space, mu=0, theta=0.15, sigma=0.2):
        self.mu= mu
        self.sigma= sigma
        self.theta= theta
        self.action_dim= action_space.shape[0]
        self.low= action_space.low
        self.high= action_space.high
        self.reset() 

    def reset(self):
        self.state= np.ones(self.action_dim) * self.mu

    def sample(self):
        x= self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        state= x + dx
        return self.state
    
    def get_action(self, action):
        ou_state = self.sample()
        return np.clip(action + ou_state, self.low, self.high)

        