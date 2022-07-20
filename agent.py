# -*- coding: utf-8 -*-
"""
Contains RL agent class and NN
"""
import random
from copy import deepcopy
import numpy as np

from game import gameEnvironmentClass
from neuralNetwork import processGrid


class agentClass:
	"""
	deep q learning agent
	-stores game environment, including game state
	-store epsilon (ratio exploration / exploitation)
	"""
	def __init__(self, epsilon, initial_env=None):
        
		self.epsilon = epsilon

		# possible actions
		self.actions = ['left', 'right', 'down', 'up']

		# generate new env if None as input
		if type(initial_env) == type(None):
			self.env = gameEnvironmentClass()
		else:
			self.env = deepcopy(initial_env)


	def interact(self, action):
		"""
		execute action in the environment, update the state
		"""
		old_grid = self.env.grid
		self.env.step(action)
		new_grid = self.env.grid

	def choose_action(self, policy_model):
		"""
		choose the action
		randomly or the one that maximises Q value (epsilon greedy)
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



