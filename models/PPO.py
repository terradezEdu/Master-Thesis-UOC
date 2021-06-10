import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.distributions import MultivariateNormal

import numpy as np
from gym.wrappers import Monitor
from IPython import display as ipythondisplay
from pyvirtualdisplay import Display

#tensordboard
from torch.utils.tensorboard import SummaryWriter

PATH_TENSORDBOARD= '/content/drive/MyDrive/TFM/TENSORDBOARD/PPO'
VIDEO_PATH= '/content/drive/MyDrive/TFM/Video/PPO/episode/'
writer = SummaryWriter(log_dir=PATH_TENSORDBOARD)
PATH= '/content/drive/MyDrive/TFM/models/save'

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.probs),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []

class Actor(nn.Module):
    def __init__(self, n_actions, input_dims, lr_ac,
            fc1_dims=256, fc2_dims=256, save_dir= PATH, cov_var=0.5, name='actor'):
        super(Actor, self).__init__()

        self.save_dir= save_dir
        self.save_file= os.path.join(self.save_dir, name+'_PPO')
        self.cov_var = T.full(size=(n_actions,), fill_value=cov_var)
        self.actor = nn.Sequential(
                nn.Linear(*input_dims, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, n_actions),
                nn.Tanh())

        self.optimizer = optim.Adam(self.parameters(), lr=lr_ac)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)


    def forward(self, state):
        mu    = self.actor(state)
        cov_mat = T.diag(self.cov_var).to(self.device)
        dist = MultivariateNormal(mu, cov_mat) 
        return dist

    def save_checkpoint(self):
        T.save(self.state_dict(), self.save_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.save_file))

class Critic(nn.Module):
    def __init__(self, input_dims, lr_critic, fc1_dims=256, fc2_dims=256,
             save_dir= PATH, name='critic'):
        super(Critic, self).__init__()

        self.save_dir= save_dir
        self.save_file= os.path.join(self.save_dir, name+'_PPO')
        
        self.critic = nn.Sequential(
                nn.Linear(*input_dims, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr_critic)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)

        return value

    def save_checkpoint(self):
        T.save(self.state_dict(), self.save_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.save_file))

class Agent:
    def __init__(self, n_actions, input_dims, gamma=0.95, lr_ac=0.001, lr_critic=0.0003 ,gae_lambda=0.95,
            policy_clip=0.2, batch_size=32, n_epochs=10, fc1_dims=256, fc2_dims=256):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda

        self.actor = Actor(n_actions, input_dims, lr_ac=lr_ac)
        self.critic = Critic(input_dims, lr_critic= lr_critic)
        self.memory = PPOMemory(batch_size)
        
    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)

        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        probs = log_prob.detach().cpu().data.numpy()
        actions = action.detach().cpu().data.numpy()
        value = T.squeeze(value).item()

        return actions, probs, value

    def update(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr,\
            reward_arr, dones_arr, batches = self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*\
                            (1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t
            advantage = T.tensor(advantage).to(self.actor.device)

            values = T.tensor(values).to(self.actor.device)
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                old_probs = T.tensor(old_prob_arr[batch]).to(self.actor.device) #here
                actions = T.tensor(action_arr[batch]).to(self.actor.device)

                dist = self.actor(states)
                critic_value = self.critic(states)

                critic_value = T.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                #prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip,
                        1+self.policy_clip)*advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.2*critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()
        return  actor_loss.cpu().data.numpy(), critic_loss.cpu().data.numpy(), total_loss.cpu().data.numpy()

    def train(self, env, max_episode=1000, batch_size=32, N=320, max_steps= 2000,
    nblock=100):
            
            training_rewards= []
            episode = 0
            learn_iters= 0
            total_steps= 0
            training= True
            loss_actor_ep= []
            loss_critic_ep= []
            list_act_loss= []
            list_crt_loss= []
            list_total_loss= []

            print("Training...")
            while training:
                obs = env.reset()
                total_reward= 0
                step_count= 0
                
                done= False
                
                
                while done == False:
                    action, prob, val = self.choose_action(obs)
                    new_obs, reward, done, _ = env.step(action[0])
                    total_steps += 1
                    step_count += 1
                    total_reward += reward
                    self.remember(obs, action, prob, val, reward, done)
                    if total_steps % N == 0:

                        actor_loss, critic_loss, total_loss= self.update()
                        list_act_loss.append(actor_loss)
                        list_crt_loss.append(critic_loss)
                        list_total_loss.append(total_loss)

                        learn_iters += 1
                    obs = new_obs
                    
                    if step_count > max_steps:
                        done= True
                    
                    if done:                   
                        episode += 1

                        training_rewards.append(total_reward) # append rewards
                        mean_rewards= np.mean(training_rewards[-nblock:]) # mean rewards

                        print("\rLearn iterations {:d} Mean Rewards {:.2f} Episodes {:d} \t\t".format(
                            learn_iters, mean_rewards, episode), end="")
                        
                        writer.add_scalar("Mean rewards", mean_rewards, episode)
                        step_count= 0
                
                        if episode % 100 == 0:
                            self.save_models()
                            df_rewards = pd.DataFrame ()
                            df_loss = pd.DataFrame ()
                            df_rewards['training_rewards']= training_rewards
                            df_loss['actor_loss']= list_act_loss
                            df_loss['critic_loss']= list_crt_loss
                            df_loss['total_loss']= list_total_loss

                            df_rewards.to_csv('/content/drive/MyDrive/TFM/models/save/PPO/df_ppo_rewards.csv' , index=False)
                            df_loss.to_csv('/content/drive/MyDrive/TFM/models/save/PPO/df_ppo_loss.csv' , index=False)

                            reward_rec= 0
                            path= VIDEO_PATH + str(episode)
                            env_video = Monitor(env, directory= path, force=True, video_callable=lambda episode: True)
                            obs= env_video.reset()
                            
                            
                            done = False
                            while not done:
                                action, _, _= self.choose_action(obs)
                                obs, r, done, _ = env_video.step(action[0])
                                reward_rec += r
                            env_video.close()
                            print("\Mean Rewards {:.2f} \t\t".format( reward_rec), end="")

                        if episode >= max_episode:
                                training= False
                                print('\nEpisode limit reached.')
                                writer.flush()
                                writer.close()
                                break