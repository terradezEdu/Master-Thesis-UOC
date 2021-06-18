import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

import numpy as np

PATH= '/content/drive/MyDrive/TFM/models/save'

class Critic(nn.Module):
    '''
    Critic model
    '''
    def __init__(self, lr, input_dims, n_actions, hidden_size1=256, hidden_size2=256,
            name='critic', save_dir= PATH):
        """
        Construct critic class with the parameters
        
        input_dims: size of the observation space
        hidden_size1: nº of neurons in the firts layer
        hidden_size2: nº of neurons in the second layer
        name: name to save the network
        save_dir: path to save the weights
        """
        super(Critic, self).__init__()
        self.input_dims= input_dims
        self.hidden_size1= hidden_size1
        self.hidden_size2= hidden_size2
        self.n_actions= n_actions
        self.name= name
        self.save_dir= save_dir
        self.save_file= os.path.join(self.save_dir, name+'_sac')

        self.fc1= nn.Linear(self.input_dims[0]+n_actions, self.hidden_size1)
        self.fc2= nn.Linear(self.hidden_size1, self.hidden_size2)
        self.q= nn.Linear(self.hidden_size2, 1)

        self.optimizer= optim.Adam(self.parameters(), lr=lr)

    def forward(self, state, action):
                """
        Return the q function stimation
        """
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
    '''
    Estimation of the value function
    '''
    def __init__(self, lr, input_dims, hidden_size1=256, hidden_size2=256,
            name='value', save_dir=PATH):
        """
        Construct Value class with the parameters
        
        input_dims: size of the observation space
        hidden_size1: nº of neurons in the firts layer
        hidden_size2: nº of neurons in the second layer
        name: name to save the network
        save_dir: path to save the weights
        """
        super(ValueAnn, self).__init__()
        self.input_dims= input_dims
        self.hidden_size1= hidden_size1
        self.hidden_size2= hidden_size2
        self.name= name
        self.save_dir= save_dir
        self.save_file= os.path.join(self.save_dir, name+'_sac')

        self.fc1= nn.Linear(*self.input_dims, self.hidden_size1)
        self.fc2= nn.Linear(self.hidden_size1, hidden_size2)
        self.v= nn.Linear(self.hidden_size2, 1)

        self.optimizer= optim.Adam(self.parameters(), lr=lr)

    def forward(self, state):
        """
        Return the value function stimation
        """
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
    '''
    Actor model
    '''
    def __init__(self, lr, input_dims, hidden_size1=256, 
            hidden_size2=256, n_actions=2, name='actor', save_dir=PATH):
        """
        Construct Actor class with the parameters:
        input_dims: observation space dimentions
        hidden_size1: nº of neurons in the first layer
        hidden_size2: nº of neurons in the second layer
        save_dir: Path to save the weights
        name: netowork's name
        """
        super(Actor, self).__init__()
        self.input_dims= input_dims
        self.hidden_size1= hidden_size1
        self.hidden_size2= hidden_size2
        self.n_actions= n_actions
        self.name= name
        self.save_dir= save_dir
        self.save_file= os.path.join(self.save_dir, name+'_sac')
        self.reparam_noise= 1e-6

        self.fc1= nn.Linear(*self.input_dims, self.hidden_size1)
        self.fc2= nn.Linear(self.hidden_size1, self.hidden_size2)
        self.mu= nn.Linear(self.hidden_size2, self.n_actions)
        self.sigma= nn.Linear(self.hidden_size2, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, state):
        """
        Return the mu and sigma for the probability distribution
        """
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