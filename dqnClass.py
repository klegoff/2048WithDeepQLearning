# -*- coding: utf-8 -*-
"""
Contains deep q-learning network class
and related functions
"""
import torch
import torch.nn as nn

class DQN(nn.Module):
	"""
	neural network class for deep q-learning
	Estimate Q value, from game state (torch.Tensor, of shape (batch_size, feature_size))
	"""
	def __init__(self):
		super().__init__()
		self.dense1 = nn.Linear(16, 128) # 4*4 input array
		self.dense2 = nn.Linear(128, 128)
		self.dense3 = nn.Linear(128, 4) # 4 possible actions

	def forward(self, x):
		x = self.dense1(x)
		x = self.dense2(x)
		x = self.dense3(x)
		return x

def processGrid(grid):
	"""
	format grid, as input of DQN
	"""
	return torch.tensor(env.grid).float().reshape(1,-1)
