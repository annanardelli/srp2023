#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 13:19:30 2023

@author: lukeshao
"""

# Code for registering the environment and running pygame window of grid_world

import gym
from gym import spaces
import pygame
import numpy as np 
import random

from gym.envs.registration import register

register(
    id='gym_examples/GridWorld-v0',
    entry_point='gym_examples.envs:GridWorldEnv',
    max_episode_steps=300,
)

env = gym.make("gym_examples/GridWorld-v0", render_mode="human")
observation, info = env.reset()

alpha = 0.1
gamma = 0.6
epsilon = 0.2
q = np.zeros([25, 4])

states = {} 
index = 0
for x in range(5): 
    for y in range(5): 
        pair = (x,y)
        states.update({pair : index})
        index = index + 1
        
print(states)

for _ in range(1000):    
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
