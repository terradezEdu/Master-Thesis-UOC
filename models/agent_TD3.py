import torch as T
import torch.nn.functional as F
from torch.distributions.normal import Normal

import os
import numpy as np
import pandas as pd
from copy import deepcopy, copy

from gym.wrappers import Monitor
from IPython import display as ipythondisplay
from pyvirtualdisplay import Display
from torch.utils.tensorboard import SummaryWriter

from models.buffer import ReplayBuffer
from models.model_TD3 import Actor, Critic


# Global variables
PATH_TENSORDBOARD= '/content/drive/MyDrive/TFM/TENSORDBOARD/TD3'
VIDEO_PATH= '/content/drive/MyDrive/TFM/Video/TD3/'
writer = SummaryWriter(log_dir=PATH_TENSORDBOARD)
device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

class Agent():
    def __init__(self, input_dims, tau, env,
            gamma=0.99, lr_ac=0.001, lr_cr=0.001,
            update_actor_interval=2,
            n_actions=2, max_size=1000000, layer1_size=400,
            layer2_size=300, batch_size=100, noise=0.1):
        self.gamma = gamma
        self.tau = tau
        self.max_action = env.action_space.high
        self.min_action = env.action_space.low
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.learn_step_cntr = 0
        self.time_step = 0 
        self.n_actions = n_actions
        self.update_actor_iter = update_actor_interval
        self.env= env

        self.actor = Actor(lr_ac, input_dims, layer1_size,
                                  layer2_size, n_actions=n_actions,
                                  name='actor').to(device)
        self.critic_1 = Critic(lr_cr, input_dims, layer1_size,
                                      layer2_size, n_actions=n_actions,
                                      name='critic_1').to(device)
        self.critic_2 = Critic(lr_cr, input_dims, layer1_size,
                                      layer2_size, n_actions=n_actions,
                                      name='critic_2').to(device)

        self.target_actor = deepcopy(self.actor).to(device)
        self.target_critic_1 = deepcopy(self.critic_1).to(device)
        self.target_critic_2 = deepcopy(self.critic_2).to(device)

        self.noise = noise

    def choose_action(self, obs):
        state = T.tensor(obs, dtype=T.float).to(device)
        mu = self.actor.forward(state).to(device)
        mu_prime = mu + T.tensor(np.random.normal(scale=self.noise), dtype=T.float).to(device)
        mu_prime = T.clamp(mu_prime, self.min_action[0], self.max_action[0])
        self.time_step += 1
        return mu_prime.cpu().detach().numpy()

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def update(self):
    
        state, action, reward, new_state, done = \
                self.memory.sample_buffer(self.batch_size)

        reward = T.tensor(reward, dtype=T.float).to(device)
        done = T.tensor(done).to(device)
        state_ = T.tensor(new_state, dtype=T.float).to(device)
        state = T.tensor(state, dtype=T.float).to(device)
        action = T.tensor(action, dtype=T.float).to(device)

        target_actions = self.target_actor.forward(state_)
        target_actions = target_actions + \
                T.clamp(T.tensor(np.random.normal(scale=0.2)), -0.5, 0.5)
        target_actions = T.clamp(target_actions, self.min_action[0], self.max_action[0])

        q1_ = self.target_critic_1.forward(state_, target_actions)
        q2_ = self.target_critic_2.forward(state_, target_actions)

        q1 = self.critic_1.forward(state, action)
        q2 = self.critic_2.forward(state, action)

        q1_[done] = 0.0
        q2_[done] = 0.0

        q1_ = q1_.view(-1)
        q2_ = q2_.view(-1)

        critic_value_ = T.min(q1_, q2_)

        target = reward + self.gamma*critic_value_
        target = target.view(self.batch_size, 1)

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()

        q1_loss = F.mse_loss(target, q1)
        q2_loss = F.mse_loss(target, q2)
        critic_loss = q1_loss + q2_loss
        critic_loss_save= critic_loss
        critic_loss.backward()

        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        #self.learn_step_cntr += 1
        
        if self.learn_step_cntr % self.update_actor_iter != 0:
            return 

        self.actor.optimizer.zero_grad()
        actor_q1_loss = self.critic_1.forward(state, self.actor.forward(state))
        actor_loss = -T.mean(actor_q1_loss)
        actor_loss_save= actor_loss
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_parameters()

        return critic_loss.cpu().data.numpy(), actor_loss.cpu().data.numpy()

    def update_parameters(self):

        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            ) 
            
        for target_param, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            ) 
            
        for target_param, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            ) 

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.target_critic_1.save_checkpoint()
        self.target_critic_2.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.target_critic_1.load_checkpoint()
        self.target_critic_2.load_checkpoint()
    
    def train(self, max_episodes=5000, nblock=100, episode_start=0,
              save_episode=100, max_step= 1000):
        
        training_rewards=[]
        actor_loss_list= []
        critic_loss_list= []
        critic_loss= 0
        actor_loss= 0
        total_steps= []

        # Filling in the buffer replay with random actions
        print("Filling in replay buffer...")
        while self.memory.mem_cntr <  self.memory.mem_size:
            observation= self.env.reset()
            score= 0
            done= False
            step= 0
            while done == False:
                action = self.env.action_space.sample()
                observation_, reward, done, info = self.env.step(action)
                score += reward
                self.remember(observation, action, reward, observation_, done)
                observation= observation_
                step += 1
                if step > max_step:
                    done= True



           
        episode = 0
        training= True
        print("Training...")
        while training:
            observation = self.env.reset()
            score= 0
            done= False
            episode_act_loss= []
            episode_crt_loss= []
            step= 0

            while done == False:
                action = self.choose_action(observation)
                observation_, reward, done, info = self.env.step(action)
                score += reward
                self.remember(observation, action, reward, observation_, done)
                observation= observation_
                step += 1
                self.learn_step_cntr += 1

                if self.learn_step_cntr % self.update_actor_iter != 0:
                    self.update()
                else:
                    critic_loss, actor_loss =self.update()
                    episode_act_loss.append(actor_loss)
                    episode_crt_loss.append(critic_loss)
                
                

                if step > max_step:
                    done= True

                if done:                   
                    episode += 1

                    

                    training_rewards.append(score) # append rewards
                    total_steps.append(self.memory.mem_cntr) #append nº buffer transition

                    if self.learn_step_cntr % self.update_actor_iter == 0:
                        
                        mean_act_loss= 0
                        mean_crt_loss= 0

                        
                        mean_act_loss= np.mean(episode_act_loss)
                        mean_crt_loss= np.mean(episode_crt_loss)

                        actor_loss_list.append(mean_act_loss)
                        critic_loss_list.append(mean_crt_loss)

                        writer.add_scalar("Mean actor loss", mean_act_loss, episode)
                        writer.add_scalar("Mean critic loss", mean_crt_loss, episode)
                    
                    mean_rewards= np.mean(training_rewards[-nblock:]) # mean rewards
                    print("\rEpisode {:d} Mean Rewards {:.2f} \t\t".format(
                        episode, mean_rewards), end="")
                    
                    writer.add_scalar("Mean rewards", mean_rewards, episode)
                    writer.add_scalar("rewards", score, episode)
                   
            
                    if episode % save_episode == 0:
                        #save
                        self.save_models()
                        df_rewards = pd.DataFrame ()
                        df_loss = pd.DataFrame ()
                        df_rewards['training_rewards']= training_rewards
                        df_rewards['n_steps']= total_steps
                        df_rewards['episode']= episode
                        df_loss['episode_actor_loss']= actor_loss_list
                        df_loss['episode_critic_loss']= critic_loss_list
                        df_rewards.to_csv('/content/drive/MyDrive/TFM/models/save/TD3/df_rewards_td3' + '_'+ str(episode_start) + '.csv', index=False)
                        df_loss.to_csv('/content/drive/MyDrive/TFM/models/save/TD3/df_loss_td3' + '_'+ str(episode_start) + '.csv', index=False)
                        
                        #record
                        path= VIDEO_PATH + str(episode)
                        print(path)
                        env_video = Monitor(self.env, directory= path, force=True, video_callable=lambda episode: True)
                        obs= env_video.reset()
                        
                        
                        done = False
                        while not done:
                            action= self.choose_action(obs)
                            obs, r, done, _ = env_video.step(action)
                            
                        env_video.close()

                if episode >= max_episodes:
                        training= False
                        print('\nEpisode limit reached.')
                        writer.flush()
                        writer.close()
                        return mean_rewards
        return mean_rewards