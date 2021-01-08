#!/usr/bin/env python
# coding: utf-8

# # Navigation
# 
# ---
# 
# This notebook solves the Banana environment from the Unity ML-Agents for the first project of the [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893).
# 
# Author: Piotr Zioło

# ### Load the environment

# In[1]:


from unityagents import UnityEnvironment
import gym
import random
import torch
import numpy as np
from collections import deque
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Fix the dying kernel problem
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# In[2]:


env = UnityEnvironment(file_name="Banana_Windows_x86_64/Banana.exe")


# In[3]:


# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]


# ### The game
# 
# The simulation contains a single agent that navigates a large environment.  At each time step, it has four actions at its disposal:
# - `0` - walk forward 
# - `1` - walk backward
# - `2` - turn left
# - `3` - turn right
# 
# The state space has `37` dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  A reward of `+1` is provided for collecting a yellow banana, and a reward of `-1` is provided for collecting a blue banana. 

# In[4]:


# Reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# Number of agents in the environment
print('Number of agents:', len(env_info.agents))

# Number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# The state space 
state = env_info.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)


# ### Define and train the agent

# In[5]:



from dqn_agent import Agent

agent = Agent(state_size=37, action_size=4, seed=0)


# In[6]:


def dqn(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    best_score = -np.inf
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        for t in range(max_t):
            action = int(agent.act(state, eps))
            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished
            score += reward   
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        mean_score = np.mean(scores_window)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, mean_score), end="")
#         if mean_score > best_score and mean_score > 0:
#             torch.save(agent.qnetwork_local.state_dict(), 'models/checkpoint{}_{}.pth'.format(i_episode, mean_score))
#             best_score = mean_score
        if i_episode % 100 == 0:
            torch.save(agent.qnetwork_local.state_dict(), 'models/checkpoint{}_{}.pth'.format(i_episode, mean_score))
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, mean_score))
        if mean_score >= 30.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, mean_score))
            torch.save(agent.qnetwork_local.state_dict(), 'models/checkpoint.pth')
            break
    return scores

# scores = dqn()
#
# data = pd.DataFrame(scores, columns=['reward'])
# data.loc[:, 'episode'] = range(0, len(scores))
# data.to_csv("models/scores.csv", index=False)


# ### Plot the scores

# In[6]:


# data = pd.read_csv("models/scores.csv")
# data.loc[:, ""]
# sns_plot = sns.lineplot(x='episode', y='reward', data=data)
# sns_plot.figure.savefig("models/scores.png")
# plt.show()
#

# ### Test the agent

# In[6]:


# load the weights from file
agent.qnetwork_local.load_state_dict(torch.load('models/checkpoint.pth'))

env_info = env.reset(train_mode=False)[brain_name] # reset the environment
state = env_info.vector_observations[0]            # get the current state
score = 0                                          # initialize the score
while True:
    action = int(agent.act(state, 0))              # select an action
    env_info = env.step(action)[brain_name]        # send the action to the environment
    next_state = env_info.vector_observations[0]   # get the next state
    reward = env_info.rewards[0]                   # get the reward
    done = env_info.local_done[0]                  # see if episode has finished
    score += reward                                # update the score
    state = next_state                             # roll over the state to next time step
    if done:                                       # exit loop if episode finished
        break
    
print("Score: {}".format(score))


# In[7]:


env.close()


# In[ ]:




