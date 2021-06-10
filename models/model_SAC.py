import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

import numpy as np

PATH= '/content/drive/MyDrive/TFM/models/save'

class Critic(nn.Module):
    def __init__(self, lr, input_dims, n_actions, fc1_dims=256, fc2_dims=256,
            name='critic', save_dir= PATH):
        super(Critic, self).__init__()
        self.input_dims= input_dims
        self.fc1_dims= fc1_dims
        self.fc2_dims= fc2_dims
        self.n_actions= n_actions
        self.name= name
        self.save_dir= save_dir
        self.save_file= os.path.join(self.save_dir, name+'_sac')

        self.fc1= nn.Linear(self.input_dims[0]+n_actions, self.fc1_dims)
        self.fc2= nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q= nn.Linear(self.fc2_dims, 1)

        self.optimizer= optim.Adam(self.parameters(), lr=lr)

    def forward(self, state, action):
        action_value= self.fc1(T.cat([state, action], dim=1))
        x= F.relu(action_value)
        x= self.fc2(x)
        x= F.relu(x)

        q= self.q(x)

        return q

    def save_checkpoint(self):
        T.save(self.state_dict(), self.save_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.save_file))

class ValueAnn(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims=256, fc2_dims=256,
            name='value', save_dir=PATH):
        super(ValueAnn, self).__init__()
        self.input_dims= input_dims
        self.fc1_dims= fc1_dims
        self.fc2_dims= fc2_dims
        self.name= name
        self.save_dir= save_dir
        self.save_file= os.path.join(self.save_dir, name+'_sac')

        self.fc1= nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2= nn.Linear(self.fc1_dims, fc2_dims)
        self.v= nn.Linear(self.fc2_dims, 1)

        self.optimizer= optim.Adam(self.parameters(), lr=lr)

    def forward(self, state):
        x= self.fc1(state)
        x= F.relu(x)
        x= self.fc2(x)
        x= F.relu(x)

        v= self.v(x)

        return v

    def save_checkpoint(self):
        T.save(self.state_dict(), self.save_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.save_file))

class Actor(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims=256, 
            fc2_dims=256, n_actions=2, name='actor', save_dir=PATH):
        super(Actor, self).__init__()
        self.input_dims= input_dims
        self.fc1_dims= fc1_dims
        self.fc2_dims= fc2_dims
        self.n_actions= n_actions
        self.name= name
        self.save_dir= save_dir
        self.save_file= os.path.join(self.save_dir, name+'_sac')
        self.reparam_noise= 1e-6

        self.fc1= nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2= nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu= nn.Linear(self.fc2_dims, self.n_actions)
        self.sigma= nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, state):
        x= self.fc1(state)
        x= F.relu(x)
        x= self.fc2(x)
        x= F.relu(x)

        mu= self.mu(x)
        sigma= self.sigma(x)

        sigma= T.clamp(sigma, min=self.reparam_noise, max=1)

        return mu, sigma


    def save_checkpoint(self):
        T.save(self.state_dict(), self.save_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.save_file))