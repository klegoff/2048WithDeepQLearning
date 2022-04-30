# -*- coding: utf-8 -*-
"""
Contains RL agent class and NN
"""
import random
from copy import deepcopy
from collections import deque, namedtuple
import numpy as np

from gameClass import gameEnvironment
from dqnClass import DQN

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
	reward1 = 5 * (new_grid.max() - old_grid.max())

	return reward0 + reward1

def gameContinues(grid):
	"""
	check if the grid is still playable or not
	:input:
		grid (type=np.array), state of the game
	:output:
		(type = bool), True if grid is still playble, False otherwise
	"""
	zeroCount = len(np.where(env.grid==0)[0])
	if zeroCount == 0:
		return False
	else :
		return True

#######################
#
# Replay memory class
# source = pytorch
#
#######################

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class replayMemory(object):
	"""
	class to store all transition
	"""
	def __init__(self, capacity):
		self.memory = deque([],maxlen=capacity)

	def push(self, *args):
		"""Save a transition"""
		self.memory.append(Transition(*args))

	def sample(self, batch_size):
		return random.sample(self.memory, batch_size)

	def __len__(self):
		return len(self.memory)

class agent:
	"""
	deep q learning agent
	contains the neural network for reward computation
	"""
	def __init__(self, epsilon, gamma, alpha,initial_env=None):
		# possible actions
		self.actions = ['left', 'right', 'down', 'up']

		# hyperparameters
		self.gamma = gamma
		self.alpha = alpha
		self.epsilon = epsilon

		# generate env if None as input
		if type(initial_env) == type(None):
			self.env = gameEnvironment()
		else:
			self.env = deepcopy(initial_env)

		# store initial state
		self.state = deepcopy(self.env.grid)


	def interact(action):
		pass

	#reward = computeReward(old_grid, new_grid)

	def choose_action(self):
		"""
		choose the action
		randomly or the one the maximise q
		"""
		if random.random() < self.epsilon:
			# random action (exploration)
			return random.choice(self.actions)

		else:
			# get action with highest q value
			tensor = self.grid 
			output_tensor = self.policy_dqn.forward(output_tensor)
			return action[np.argmax(output_tensor.detach().numpy())] 
