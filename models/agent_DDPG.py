""" DDPG Agent. 

Implementation of the DDPG Agent

author: Eduardo Terrádez
year: 2021

"""
import os
import torch as T
import torch.autograd
import torch.optim as optim
import torch.nn as nn
import pandas as pd
from models import model_DDPG
from models.buffer import ReplayBuffer
from copy import deepcopy, copy
import numpy as np
from gym.wrappers import Monitor
import base64


#tensordboard
from torch.utils.tensorboard import SummaryWriter

device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

#Path directories from Google Drive
PATH_TENSORDBOARD= '/content/drive/MyDrive/TFM/TENSORDBOARD/DDPG'
VIDEO_PATH= '/content/drive/MyDrive/TFM/Video/DDPG/episode/'

#create the SummaryWriter for tensorboard
writer = SummaryWriter(log_dir=PATH_TENSORDBOARD)



class DDPGAgent:
    """
    Implementation of the agent DDPG
    """
    def __init__(self, env, noise, hidden_size1=400, hidden_size2=300, 
                 actor_lr=1e-4, critic_lr=1e-3, 
                 gamma=0.99, tau=1e-3, nblock=300, max_step=2000, max_reward=300,
                 max_size= 100000):
        """
        Parameters:
        env: environment to training the agent
        noise: Noise method from models/class Orn_Uhlen
        hidden_size1: nº of neurons in the first layer
        hidden_size2: nº of neurons in the second layer
        actor_lr: actor's networks learning rate
        critic_lr: critic's networks learning rate
        gamma: gamma parameter "target Q-values"
        tau: tau parameter "conservative policy iteration"
        nblock: nº of episodes to calculate the avg. rewards
        max_step: max. to complete the episode
        max_rewards: rewards' threshold
        max_size: max. size buffer experience replay
        """
        self.num_states= env.observation_space.shape
        self.num_actions= env.action_space.shape[0]
        self.gamma= gamma
        self.tau= tau
        self.env= env
        self.nblock= nblock
        self.noise= noise
        self.max_step= max_step
        self.max_reward= max_reward
        self.memory = ReplayBuffer(max_size, self.num_states, self.num_actions)
        self.hidden_size1= hidden_size1
        self.hidden_size2= hidden_size2
        self.actor_lr= actor_lr
        self.critic_lr= critic_lr

        self.initialize()

    def initialize(self):
        """
        Initialize:
        control variables
        environments
        ANNs
        Optimizers
        """
        self.total_reward= 0 #variable to control the cum. sum rewards
        self.step_count= 0 # variable to control the nº steps
        self.state = self.env.reset()
        self.training_rewards= []
        # ANNs
        self.actor= model_DDPG.Actor(self.num_states, self.hidden_size1, self.hidden_size2, 
                           self.num_actions).to(device)
        self.target_actor= deepcopy(self.actor).to(device)
        self.critic= model_DDPG.Critic(self.num_states, self.hidden_size1,
                             self.hidden_size2, output_size=1, n_action= self.num_actions).to(device)
        self.target_critic= deepcopy(self.critic).to(device)
        # OPTIMIZERS
        self.actor_optim= optim.Adam(self.actor.parameters(), 
                                      lr=self.actor_lr)
        self.critic_optim= optim.Adam(self.critic.parameters(), 
                                       lr=self.critic_lr)
        
        self.critic_criterion= nn.MSELoss()

    def get_action(self, state):
        """
        Get actions from the actor
        """
        state= T.from_numpy(state).float().unsqueeze(0).to(device)
        action= self.actor.forward(state)
        action= action.detach().cpu().numpy().reshape(-1)
        return action
        
    def update(self, batch_size):
        # Get samples from buffer replay
        states, actions, rewards, next_states, _ = self.memory.sample_buffer(batch_size)
        states= T.FloatTensor(states).to(device)
        actions= T.FloatTensor(actions).to(device)
        rewards= T.FloatTensor(rewards).to(device)
        next_states= T.FloatTensor(next_states).to(device)
        
        # Critic loss
        Qvals= self.critic.forward(states, actions)
        next_actions= self.target_actor.forward(next_states)
        next_Q= self.target_critic.forward(next_states, next_actions.detach())
        Qprime= rewards + self.gamma * next_Q.view(-1)
        
        critic_loss= self.critic_criterion(Qvals.view(-1), Qprime)

        # Actor loss
        actions_forward= self.actor.forward(states)
        policy_loss= -self.critic.forward(states, actions_forward ).mean()
        
        # update ANNs
        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()

        self.critic_optim.zero_grad()
        critic_loss.backward() 
        self.critic_optim.step()
        return  policy_loss.cpu().data.numpy(), critic_loss.cpu().data.numpy()

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()

    def take_step(self, mode='train'):
        if mode == 'explore':             
            action= self.env.action_space.sample()  # random sample until filling in buffer
        else:           
            action= self.get_action(self.state) # action propose by actor
            action= self.noise.get_action(action)
            self.step_count += 1 

        

        # Observe new state, rewards, done and push to memory
        new_state, reward, done, _ = self.env.step(action)  #
        self.total_reward += reward
        self.memory.store_transition(self.state, action, reward,  new_state, done) # push to buffer
        self.state= new_state.copy()
        
        if self.step_count > self.max_step:
            done= True

        if done:
            self.state= self.env.reset()
        return done

    def soft_update(self):
        """
        Update parameters from target networks and main networks
        """
        for target_param, param in zip(self.target_actor.parameters(), 
                               self.actor.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        for target_param, param in zip(self.target_critic.parameters(), 
                               self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
    
    

    def train(self, max_episodes=5000, 
              batch_size=32, max_buffer=50000):
        """
        Training the agent.
        Parameters:
        max_episode: nº of episodes to train
        batch_size: nº of batch to update the loss
        max_buffer: nº of max samples storage in the buffer
        """
        # initialize arrays to store the loss functions values
        loss_actor_ep= []
        loss_critic_ep= []
        # Filling in the buffer replay with random actions
        print("Filling in replay buffer...")
        while self.memory.mem_cntr <  self.memory.mem_size:
            self.take_step(mode='explore')

           
        episode = 0
        training= True
        print("Training...")
        while training:
            self.state0 = self.env.reset()
            self.total_reward= 0
            gamedone= False
            episode_act_loss= []
            episode_crt_loss= []

            while gamedone == False:
                gamedone = self.take_step(mode='train')
                actor_loss, critic_loss= self.update(batch_size)
                episode_act_loss.append(actor_loss)
                episode_crt_loss.append(critic_loss)
                self.soft_update()
                
                if gamedone:                   
                    episode += 1

                    mean_act_loss= 0
                    mean_act_loss= 0

                    self.training_rewards.append(self.total_reward) # append rewards
                    mean_rewards= np.mean(self.training_rewards[-self.nblock:]) # mean rewards
                    mean_act_loss= np.mean(episode_act_loss) #mean actor loss
                    mean_crt_loss= np.mean(episode_crt_loss) # mean critic loss
                    loss_actor_ep.append(mean_act_loss) # append mean actor loss per episode
                    loss_critic_ep.append(mean_crt_loss) # append mean critic loss per episode

                    print("\rEpisode {:d} Mean Rewards {:.2f} \t\t".format(
                        episode, mean_rewards), end="")
                    
                    #add to tensorboard
                    writer.add_scalar("Mean rewards", mean_rewards, episode)
                    writer.add_scalar("Mean actor loss", mean_act_loss, episode)
                    writer.add_scalar("Mean critic loss", mean_crt_loss, episode)
                    self.step_count= 0

                    if episode % 100 == 0:
                        #Save models and csv with rewards and loss functions
                        self.save_models()
                        df_rewards = pd.DataFrame ()
                        df_loss = pd.DataFrame ()
                        df_rewards['training_rewards']= self.training_rewards
                        df_loss['episode_actor_loss']= loss_actor_ep
                        df_loss['episode_critic_loss']= loss_critic_ep
                        # ----------NOTICE: changes path to save the info-----------
                        df_rewards.to_csv('/content/drive/MyDrive/TFM/models/save/DDPG/df_ddpg_rewards.csv' , index=False)
                        df_loss.to_csv('/content/drive/MyDrive/TFM/models/save/DDPG/df_ddpg_loss.csv' , index=False)
                        path= VIDEO_PATH + str(episode)
                        #print(path)
                        
                        #Record episode in mp4
                        env_video = Monitor(self.env, directory= path, force=True, video_callable=lambda episode: True)
                        obs= env_video.reset()
                        
                        
                        done = False
                        while not done:
                            action= self.get_action(obs)
                            obs, r, done, _ = env_video.step(action)
                            
                        env_video.close()
                
                if episode >= max_episodes:
                        training= False
                        print('\nEpisode limit reached.')
                        writer.flush()
                        writer.close()
                        break