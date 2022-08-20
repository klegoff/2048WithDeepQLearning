# -*- coding: utf-8 -*-
"""
Contains reward functions
"""
import numpy as np

#######################
#
# Reward functions
#
#######################

def reward1(old_grid, new_grid):
	"""
	Computes the reward from one state to another
	:inputs:
		old_grid, new_grid (type=np.array), state of the game before and after an action
	:output:
		reward (type=int), reward of the state evolution
	"""
	if np.max(new_grid) == 2048:
		return 10000
	else :
		# -1 to encourage the agent to play efficient moves
		return -1

def reward2(old_grid, new_grid):
	"""
	Computes the reward from one state to another
	This reward gives more indication to the agent

	:inputs:
		old_grid, new_grid (type=np.array), state of the game before and after an action
	:output:
		reward (type=int), reward of the state evolution
	"""

	# if highest tile increased, we return the square value of that new tile 
	if np.max(old_grid) < np.max(new_grid):
		return np.max(new_grid) 

	# -1 to encourage the agent to play efficient moves
	else :
		return -1

def reward3(old_grid, new_grid):
	"""
	Computes the reward from one state to another
	Higher tiles gives much more reward

	:inputs:
		old_grid, new_grid (type=np.array), state of the game before and after an action
	:output:
		reward (type=int), reward of the state evolution
	"""

	# if highest tile increased, we return the square value of that new tile 
	if np.max(old_grid) < np.max(new_grid):
		return np.max(new_grid)**2

	# -1 to encourage the agent to play efficient moves
	else :
		return -1