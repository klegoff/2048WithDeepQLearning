# -*- coding: utf-8 -*-
"""
Contains deep q-learning network class
and related functions
"""
import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn

from game import gridIsFinished

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


	def fill(self, agent, state_action_value_model, reward_function):
		"""
		agent play some moves, and fill the transition memory up to full capacity
		:inputs:
			agent (type = agentClass)
			memory (type = replayMemory)
			state_action_value_model (class = DQN)
			reward_function (function of reward)
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
			reward = reward_function(old_state, new_state)

			# fill memory from agent experience
			self.push(old_state, action, new_state, reward)


	def sampleAndFormat(self, sampleSize):
		"""
		sample from replayMemory, and format for NN
		"""
		actions = ["down", "left", "right", "up"]

		# retrive states
		transitions = self.sample(sampleSize)
		batch = Transition(*zip(*transitions))

		# coordinates of experienced (state, action)
		actionList = batch.action
		func = lambda x: actions.index(x)
		actionList = list(map(func, actionList))
		actionCoordinate = tuple(range(sampleSize)), tuple(actionList) # index of the value for Q(s,a)

		# format
		stateArray = np.stack(batch.state).reshape(sampleSize, -1)
		newStateArray = np.stack(batch.new_state).reshape(sampleSize, -1)
		rewardArray = np.stack(batch.reward)

		# cast to torch type
		stateTensor =  torch.tensor(stateArray).float()
		newStateTensor = torch.tensor(newStateArray).float()
		rewardTensor = torch.tensor(rewardArray).float()

		return stateTensor, newStateTensor, rewardTensor, actionCoordinate
