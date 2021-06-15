# Exploring the continuous actions spaces through the BipedalWalker-v3 environment

This work is the result of my master thesis in data science  (Universitat Oberta de Catalunya UOC - 2021)  and contains all the code used. In this Github, you can find implementations of [DDPG](https://arxiv.org/pdf/1509.02971.pdf), [PPO](https://arxiv.org/pdf/1707.06347.pdf), [SAC](https://arxiv.org/abs/1801.01290) and [TD3](https://arxiv.org/abs/1802.09477) for the BidepalWalker-V3. 

## Abstract

In this work, we study the BipedalWalker-v3 a particular kind of environment with a continuous
action space, where a robot with four-join should learn to walk fast and not trip himself off. This
environment is related to optimal continuous control, where the action space is not possible to
discretize. Most of the real-world problems in robotics are similar to this kind of environment.
In this study, we will conduct several experiments on policy optimization algorithms (PG) in
terms of time spent and rewards achieved to compare their performance and select which one
of them fits better in the environment. Finally, we provide a working implementation of the
tested methods in Python.



## Prerequisites
- Python (tested for python 3.7.10 in Google Colab)
- box2d-py
- Pytorch torchvision
- Tensorboard
- Gym pyvirtualdisplay (OpenAI)
- Box2d-py
- gym[Box_2D]
- xvfb ffmpeg
- xvfb python-opengl

## Folder structure

  The folder structure is a copy of the structure I used in Google Drive. Please change the path in the code before changing the folder structure.

### TENSORBOARD

 Path for the writesummary:

- TENSORBOARD/DDPG
- TENSORBOARD/PPO
- TENSORBOARD/SAC
- TENSORBOARD/TD3
- TENSORBOARD/TD3_exp (TD3 after improvements)

### Video
episodes recordings.

- Video/DDPG/episodes
- Video/PPO/episodes
- Video/SAC/episodes
- Video/TD3/episodes
- Video/TD3_exp/episodes (TD3 after improvements)

### models

- **PPO.py:** full implementation of the PPO algorithm. Agent + ANN models.
- **agent_DDP.py:** Agent implementation DDPG.
- **agent_SAC.py:** Agent implementation SAC.
- **agent_TD3.py:** Agent implementation TD3 before improvements.
- **agent_TD3_comp.py:** Agent implementation TD3 after improvements.
- **buffer.py:** Buffer replay used in all implementations.
- **model_DDPG.py:** model with ANNS implementations + the OU process to add noise.
- **model_SAC.py:** model with ANNS implementations.
- **model_TD3.py:**  model with ANNS implementations.

#### save

Path for the csv with info about rewards and loss function

- save/DDPG
- save/PPO
- save/sac
- save/PPO

### test_envs

- **env-test01.ipynb:** This notebook contains time execution test in different environments taken random actions. I want to compare the time execution of different algorithms befero select the best algorithm for my master's thesis.

### utilities

- **utilities.py:** alternative version of buffer replay and auxiliar function to show videos.

## The code implementations are inspired by:
- [Deep Reinforcement Learning Hands-On](https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition)
- [Phil Tabor](https://github.com/philtabor)
- [Petros Christodoulou](https://github.com/p-christ)
