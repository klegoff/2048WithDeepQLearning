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
import torch.nn.functional as F

from game import gridIsFinished

class DNN(nn.Module):
	"""
	implementation of dense NN
	"""
	def __init__(self):
		super().__init__()
		self.dense1 = nn.Linear(16, 512) # 4*4 input array
		self.dense2 = nn.Linear(512, 512)
		self.dense3 = nn.Linear(512, 512)
		self.dense4 = nn.Linear(512, 4) # 4 possible actions

	def forward(self, x):
		# formatting to be treated as 1D input
		x = x.reshape(-1,16)

		# forward
		x = F.relu(self.dense1(x)) 
		x = F.relu(self.dense2(x))
		x = F.relu(self.dense3(x))
		x = F.relu(self.dense4(x))
		return x

class CNN(nn.Module):
	"""
	implementation of convolutionnal NN
	"""
	def __init__(self):
		super().__init__()
		self.l1 = nn.Conv2d(2, 10, (1,2), padding="same")
		self.l2 = nn.Conv2d(10, 20, (1,3), padding="same")
		self.dense = nn.Linear(320,4) # output layer
		self.flat = nn.Flatten()

	def forward(self, x):
		# formatting
		x = x.unsqueeze(1)
		x_transposed = torch.transpose(x, 2, 3) #transpose grid so we can apply same convolution filters
		X = torch.cat((x, x_transposed), dim=1)# concat both as different channels

		# forward through 2D convolutionnal layers
		X = F.relu(self.l1.forward(X))
		X = F.relu(self.l2.forward(X))
		X = self.flat(X)
		return self.dense(X)


def DQN(nn_type="cnn"):
	"""
	return neural network object for deep q-learning
	Estimate Q value for the possible actions, from game state (torch.Tensor, of shape (batch_size, 4, 4))
	"""

	if nn_type=="cnn":
		return CNN()

	elif nn_type=="dnn":
		return DNN()
	
	else :
		print("Unknown model")
		return None

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

	def __len__(self):
		return len(self.memory)

	def reset(self):
		self.memory = deque([],maxlen=self.capacity)


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


	def sample(self, sampleSize):
		"""
		sample from replayMemory, and format for NN
		"""
		actions = ["down", "left", "right", "up"]

		# retrive states
		transitions = random.sample(self.memory, sampleSize)
		batch = Transition(*zip(*transitions))

		# coordinates of experienced (state, action)
		actionList = batch.action
		func = lambda x: actions.index(x)
		actionList = list(map(func, actionList))
		actionCoordinate = tuple(range(sampleSize)), tuple(actionList) # index of the value for Q(s,a)

		# format
		stateArray = np.stack(batch.state)#.reshape(sampleSize, -1)
		newStateArray = np.stack(batch.new_state)#.reshape(sampleSize, -1)
		rewardArray = np.stack(batch.reward)

		# cast to torch type
		stateTensor =  torch.tensor(stateArray).float()
		newStateTensor = torch.tensor(newStateArray).float()
		rewardTensor = torch.tensor(rewardArray).float()

		return stateTensor, newStateTensor, rewardTensor, actionCoordinate
