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
from models.model_SAC import Actor, Critic, ValueAnn

# Global variables
PATH_TENSORDBOARD= '/content/drive/MyDrive/TFM/TENSORDBOARD/SAC'
VIDEO_PATH= '/content/drive/MyDrive/TFM/Video/SAC/'
writer = SummaryWriter(log_dir=PATH_TENSORDBOARD)
device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

class Agent():
    def __init__(self, lr_ac=0.0003, lr_cr=0.0003, input_dims=[8],
            env=None, gamma=0.99, n_actions=2, max_size=1000000, tau=0.005,
            layer1_size=256, layer2_size=256, batch_size=100, reward_scale=2, alpha=1):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.env= env
        self.n_actions = n_actions
        self.reparam_noise= 1e-6
        self.max_action= self.env.action_space.high
        self.alpha= alpha
        self.actor = Actor(lr_ac, input_dims, n_actions=n_actions,
                    name='actor').to(device)
        self.critic_1 = Critic(lr_cr, input_dims, n_actions=n_actions,
                    name='critic_1').to(device)
        self.critic_2 = Critic(lr_cr, input_dims, n_actions=n_actions,
                    name='critic_2').to(device)
        self.value = ValueAnn(lr_cr, input_dims, name='value').to(device)
        self.target_value = deepcopy(self.value).to(device)

        self.scale = reward_scale

    def choose_action(self, obs):
        state = T.Tensor([obs]).to(device)
        actions, _ = self.sample_normal(state, reparameterize=False)

        return actions.cpu().detach().numpy()[0]

    def sample_normal(self, state, reparameterize=True):
        mu, sigma= self.actor.forward(state)
        prob= Normal(mu, sigma)

        if reparameterize:
            actions= prob.rsample()
        else:
            actions= prob.sample()

        action= T.tanh(actions)*T.tensor(self.max_action).to(device)
        log_probs= prob.log_prob(actions)
        log_probs -= T.log(1-action.pow(2) + self.reparam_noise)
        log_probs= log_probs.sum(1, keepdim=True)

        return action, log_probs

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def save_models(self):
        print('.... saving models ....')
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()

    def load_models(self):
        print('.... loading models ....')
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()

    def update(self):
        state, action, reward, new_state, done = \
                self.memory.sample_buffer(self.batch_size)

        reward = T.tensor(reward, dtype=T.float).to(device)
        done= T.tensor(done).to(device)
        state_ = T.tensor(new_state, dtype=T.float).to(device)
        state= T.tensor(state, dtype=T.float).to(device)
        action= T.tensor(action, dtype=T.float).to(device)

        value= self.value(state).view(-1)
        value_ = self.target_value(state_).view(-1)
        value_[done] = 0.0

        actions, log_probs = self.sample_normal(state, reparameterize=False)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        self.value.optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * F.mse_loss(value, value_target)
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        actions, log_probs = self.sample_normal(state, reparameterize=True)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)
        
        actor_loss = self.alpha*log_probs - critic_value
        actor_loss = T.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q_hat = self.scale*reward + self.gamma*value_

        q1_old_policy = self.critic_1.forward(state, action).view(-1)
        q2_old_policy = self.critic_2.forward(state, action).view(-1)

        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        for target_param, param in zip(self.target_value.parameters(), self.value.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            ) 

        return critic_loss.cpu().data.numpy(), actor_loss.cpu().data.numpy(), value_loss.cpu().data.numpy()
    
    def train(self, max_episodes=5000, nblock=100, episode_start=0,
              save_episode=100, max_step= 1000):
        
        training_rewards=[]
        actor_loss_list= []
        critic_loss_list= []
        value_loss_list= []
        
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
            episode_value_loss= []
            step= 0

            while done == False:
                action = self.choose_action(observation)
                observation_, reward, done, info = self.env.step(action)
                score += reward
                self.remember(observation, action, reward, observation_, done)
                observation= observation_
                step += 1

                critic_loss, actor_loss, value_loss= self.update()
                episode_act_loss.append(actor_loss)
                episode_crt_loss.append(critic_loss)
                episode_value_loss.append(value_loss)
                
                if step > max_step:
                    done= True

                if done:                   
                    episode += 1

                    mean_act_loss= 0
                    mean_crt_loss= 0

                    training_rewards.append(score) # append rewards
                    
                    mean_rewards= np.mean(training_rewards[-nblock:]) # mean rewards
                    mean_act_loss= np.mean(episode_act_loss)
                    mean_crt_loss= np.mean(episode_crt_loss)
                    mean_value_loss= np.mean(episode_value_loss)

                    actor_loss_list.append(mean_act_loss)
                    critic_loss_list.append(mean_crt_loss)
                    value_loss_list.append(mean_value_loss)

                    print("\rEpisode {:d} Mean Rewards {:.2f} \t\t".format(
                        episode, mean_rewards), end="")
                    
                    writer.add_scalar("Mean rewards", mean_rewards, episode)
                    writer.add_scalar("rewards", score, episode)
                    writer.add_scalar("Mean actor loss", mean_act_loss, episode)
                    writer.add_scalar("Mean critic loss", mean_crt_loss, episode)
                    writer.add_scalar("Mean value loss", mean_value_loss, episode)
            
                    if episode % save_episode == 0:
                        #save
                        self.save_models()
                        df = pd.DataFrame ()
                        df['training_rewards']= training_rewards
                        df['episode_actor_loss']= actor_loss_list
                        df['episode_critic_loss']= critic_loss_list
                        df['episode_value_loss']= value_loss_list
                        df.to_csv('/content/drive/MyDrive/TFM/models/save/sac/df_sac' + '_'+ str(episode_start) + '.csv', index=False)
                        
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
    