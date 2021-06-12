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


## The code implementations are inspired by:
- [Deep Reinforcement Learning Hands-On](https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition)
- [Phil Tabor](https://github.com/philtabor)
- [Petros Christodoulou](https://github.com/p-christ)
