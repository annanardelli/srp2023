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

env = gym.make("gym_examples/GridWorld-v0", render_mode="rgb_array")
observation, info = env.reset()
state_size = env.get_state_size()
print(state_size)
action_size = env.action_space.n
print(action_size)
size = env.get_size()
print(size)

states = {}
index = 0
for x in range(size):
    for y in range(size):
        pair = (y, x)
        states.update({pair: index})
        index = index + 1
print(states)

alpha = 0.1  # learning rate
gamma = 0.6  # discount rate
epsilon = 0.5  # probability that our agent will explore
decay_rate = 0.01 # of epsilon

q = np.zeros([state_size, action_size])

# training variables
num_episodes = 1000
max_steps = 100  # per episode

for episode in range(num_episodes):
    # reset the environment
    observation, info = env.reset()
    pairTuple = tuple(observation["agent"])
    state = states[pairTuple]
    terminated = False
    truncated = False
    print(state)

    for s in range(max_steps):
        # exploration-exploitation tradeoff
        if random.uniform(0,1) < epsilon:
            # explore
            action = env.action_space.sample()
        else:
            # exploit
            action = np.argmax(q[state,:])

        # take action and observe reward
        print(action)
        observation, reward, terminated, truncated, info = env.step(action)
        print(observation)
        pairTuple = tuple(observation["agent"])
        print(pairTuple)
        new_state = states[pairTuple]
        print(new_state)
        # Q-learning algorithm
        q[state,action] = q[state,action] + alpha * (reward + gamma * np.max(q[new_state,:])-q[state,action])

        # Update to our new state
        state = new_state

        if terminated or truncated:
            break
print(q)

"""
observation, info = env.reset()
pairTuple = tuple(observation["agent"])
state = states[pairTuple]
rewards = 0

for _ in range(max_steps):

    observation, reward, terminated, truncated, info = env.step(action)
    pairTuple = tuple(observation["agent"])
    new_state = states[pairTuple]

    rewards += reward
    print(f"score: {rewards}")
    state = new_state

    if terminated or truncated:
        print(_)
        break
"""
env.close()
