#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 13:19:30 2023

@author: lukeshao
"""

# Code for registering the environment and running pygame window of grid_world
import gym
import numpy as np
import random

from gym.envs.registration import register

# sets max steps for both register() and training agent
max_steps = 1000

register(
    id='gym_examples/GridWorld-v0',
    entry_point='gym_examples.envs:GridWorldEnv',
    max_episode_steps=max_steps,
)

env = gym.make("gym_examples/GridWorld-v0", render_mode="human", size=10)

'''
x1 = int(input("Set first obstacle location: first integer value in ordered pair: "))
y1 = int(input("Set first obstacle location: second integer value in ordered pair: "))

x2 = int(input("Set second obstacle location: first integer value in ordered pair: "))
y2 = int(input("Set second obstacle location: second integer value in ordered pair: "))
env.set_obstacles(x1, y1, x2, y2)
'''
env.set_obstacles_hospital()
observation, info = env.reset()
state_size = env.get_state_size()

action_size = env.action_space.n

size = env.get_size()


states = env.get_states()

alpha = 0.8  # learning rate
gamma = 0.8  # discount rate
epsilon = 1.0  # probability that our agent will explore
decay_rate = 0.001  # of epsilon

q = np.zeros([state_size, action_size])

# training variables
num_episodes = 1000
max_steps_training = max_steps  # per episode

for episode in range(num_episodes):
    # reset the environment
    observation, info = env.reset()
    pairTuple = (tuple(observation["agent"]), env.get_is_picked_up())
    state = states[pairTuple]
    terminated = False
    truncated = False

    #states
    for s in range(max_steps_training):
        # exploration-exploitation tradeoff
        if random.uniform(0, 1) < epsilon:
            # explore
            action = env.action_space.sample()
        else:
            # exploit
            action = np.argmax(q[state, :])

        # epsilon decreases exponentially --> our agent will explore less and less
        epsilon = np.exp(-decay_rate * episode)

        # take action and observe reward
        observation, reward, terminated, truncated, info = env.step(action)
        pairTuple = (tuple(observation["agent"]), env.get_is_picked_up())
        new_state = states[pairTuple]

        # Q-learning algorithm
        q[state, action] = q[state, action] + alpha * (reward + gamma * np.max(q[new_state, :]) - q[state, action])

        # Update to our new state
        state = new_state

        if terminated or truncated:
            break

env.trained()

observation, info = env.reset()
pairTuple = (tuple(observation["agent"]), env.get_is_picked_up())
state = states[pairTuple]
rewards = 0

for _ in range(max_steps):
    action = np.argmax(q[state, :])
    observation, reward, terminated, truncated, info = env.step(action)
    pairTuple = (tuple(observation["agent"]), env.get_is_picked_up())
    new_state = states[pairTuple]

    print(f"Current State: {state}")
    print(f"Action: {action}")
    print(observation)
    print(pairTuple)

    rewards += reward
    print(f"score: {rewards}")
    state = new_state

    if terminated or truncated:
        print("Steps Taken: " + str(_ + 1))
        break

env.close()
