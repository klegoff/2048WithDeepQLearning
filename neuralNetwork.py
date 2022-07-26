# -*- coding: utf-8 -*-
"""
Contains deep q-learning network class
and related functions
"""
import random
import numpy as np
from collections import deque, namedtuple

import torch
import torch.nn as nn

from game import gridIsFinished, reward2

class DQN(nn.Module):
	"""
	neural network class for deep q-learning
	Estimate Q value for the possible actions, from game state (torch.Tensor, of shape (batch_size, feature_size))
	"""
	def __init__(self):
		super().__init__()
		self.dense1 = nn.Linear(16, 512) # 4*4 input array
		self.dense2 = nn.Linear(512, 512)
		self.dense3 = nn.Linear(512, 512)
		self.dense4 = nn.Linear(512, 4) # 4 possible actions

	def forward(self, x):
		x = self.dense1(x)
		x = self.dense2(x)
		x = self.dense3(x)
		x = self.dense4(x)
		return x

def processGrid(grid):
	"""
	format grid, as input of DQN
	"""
	stateArray = np.stack(grid).reshape(1, -1)
	stateTensor =  torch.tensor(stateArray).float()
	return stateTensor

#######################
#
# Replay memory class
# source = pytorch
#
#######################

Transition = namedtuple('Transition',
                        ('state', 'action', 'new_state', 'reward'))


class replayMemory(object):
	"""
	class to store all transition
	"""
	def __init__(self, capacity):
		self.capacity = capacity
		self.memory = deque([],maxlen=capacity)

	def push(self, *args):
		"""Save a transition"""
		self.memory.append(Transition(*args))

	def sample(self, batch_size):
		return random.sample(self.memory, batch_size)

	def __len__(self):
		return len(self.memory)

	def reset(self):
		self.memory = deque([],maxlen=capacity)


	def fill(self, agent, state_action_value_model):
		"""
		agent play some moves, and fill the transition memory up to full capacity
		:inputs:
			agent (type = agentClass)
			memory (type = replayMemory)
			state_action_value_model (class = DQN)
		"""
		while len(self.memory) < self.capacity: 
			# if game is finished, we reset the grid
			if gridIsFinished(agent.env.grid):
				agent.resetGameEnv()

			# choose action (epsilon greedy)
			with torch.no_grad():
				action = agent.choose_action(state_action_value_model)

			# execute action
			old_state = agent.env.grid
			agent.interact(action)
			new_state = agent.env.grid

			# compute reward
			reward = reward2(old_state, new_state)

			# fill memory from agent experience
			self.push(old_state, action, new_state, reward)




