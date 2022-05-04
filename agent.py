# -*- coding: utf-8 -*-
"""
Contains RL agent class and NN
"""
import random
from copy import deepcopy
import numpy as np

from game import gameEnvironmentClass
from neuralNetwork import processGrid

#######################
#
# Functions relating 
# to environment
#
#######################

def computeReward(old_grid, new_grid):
	"""
	Computes the reward from one state to another
	:inputs:
		old_grid, new_grid (type=np.array), state of the game before and after an action
	:output:
		reward (type=int), reward of the state evolution
	"""
	# count number of new empty cells (==0)
	reward0 = len(np.where(new_grid==0)[0]) - len(np.where(old_grid==0)[0])

	# difference of highest value
	reward1 = (new_grid.max() - old_grid.max()) ** 2

	return reward0 + reward1

#######################
#
# Agent class
#
#######################

class agentClass:
	"""
	deep q learning agent
	contains the neural network for reward computation
	"""
	def __init__(self, epsilon, initial_env=None):
		# possible actions
		self.actions = ['left', 'right', 'down', 'up']

		# hyperparameters
		self.epsilon = epsilon

		# generate env if None as input
		if type(initial_env) == type(None):
			self.env = gameEnvironmentClass()
		else:
			self.env = deepcopy(initial_env)

		# store initial state
		self.state = deepcopy(self.env.grid)


	def interact(self, action):
		"""
		execute action in the environment, return the reward
		"""
		old_grid = self.env.grid
		self.env.step(action)
		new_grid = self.env.grid
		reward = computeReward(old_grid, new_grid)

		return reward

	def choose_action(self, policy_model):
		"""
		choose the action
		randomly or the one the maximise q
		"""
		if random.random() < self.epsilon:
			# random action (exploration)
			return random.choice(self.actions)

		else:
			# get action with highest q value
			grid = self.env.grid 
			tensor = processGrid(grid)
			output_tensor = policy_model.forward(tensor)
			return self.actions[np.argmax(output_tensor.detach().numpy())] 

	def resetGameEnv(self):
		self.env = gameEnvironmentClass()



