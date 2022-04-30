# -*- coding: utf-8 -*-
"""
Contains RL agent class and NN
"""

#from keras.models import Sequential
#from keras.layers import Dense, Activation, Flatten
#from keras.optimizer import Adam
import random
from collections import deque, namedtuple
import numpy as np

from gameClass import gameEnvironment

#######################
#
#
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
	def __init__(self, epsilon, env, gamma, alpha,initial_state=None):
		self.actions = ['left', 'right', 'down', 'up']
		self.epsilon = epsilon
		self.state = copy.deepcopy(env.start)
		self.gamma = gamma
		self.alpha = alpha


	def interact(action)
	#reward = computeReward(old_grid, new_grid)

"""
class DQN(nn.Module):

    def __init__(self):
        super(self).__init__()
        


    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
    	pass
"""