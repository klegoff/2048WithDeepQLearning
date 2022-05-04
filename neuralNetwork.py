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

class policyNetworkClass(nn.Module):
	"""
	neural network class for deep q-learning
	Estimate Q value, from game state (torch.Tensor, of shape (batch_size, feature_size))
	"""
	def __init__(self):
		super().__init__()
		self.dense1 = nn.Linear(16, 512) # 4*4 input array
		self.dense2 = nn.Linear(512, 512)
		self.dense3 = nn.Linear(512, 4) # 4 possible actions

	def forward(self, x):
		x = self.dense1(x)
		x = self.dense2(x)
		x = self.dense3(x)
		return x

def processGrid(grid):
	"""
	format grid, as input of DQN
	"""
	stateArray = np.stack(grid).reshape(1, -1)
	stateTensor =  torch.tensor(stateArray).float()
	return stateTensor
	#return torch.tensor(grid).float().reshape(1,-1)

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




