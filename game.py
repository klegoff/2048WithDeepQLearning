# -*- coding: utf-8 -*-
"""
Contains 2048 implementation as an interactive envrionment
And other functions related to environment (reward, terminal state detection)
"""
from copy import deepcopy
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
		return np.max(new_grid) ** 2 

	# -1 to encourage the agent to play efficient moves
	else :
		return -1

#######################
#
# Environment related
#
#######################

def updateGrid(grid_, action,transposed = False):
	"""
	Makes the move of the grid and generate randomly a new number (2 or 4) in an empty case
	use recursivity to exploit the "right" move
	"""
	try: 
		grid = deepcopy(grid_) # to avoid weird duplicate of arrays

		if action == "right" or transposed:
			# iterate on each row, and compare consecutive elements
			for i in range(4):
				for j in range(3,0,-1):
					#print(grid[i][j],grid[i][j-1])
					# collide (same values in 2 tiles = we add them)
					if grid[i][j] == grid[i][j-1]:
						grid[i][j] += grid[i][j]
						grid[i][j-1] = 0
						grid = deepcopy(grid)

					if j>=2 and grid[i][j] == grid[i][j-2] and grid[i][j-1] ==0: # if two equal tiles are separated with a zero
						grid[i][j] += grid[i][j]
						grid[i][j-2] = 0
						grid = deepcopy(grid)

					if j ==3 and grid[i][j] == grid[i][j-3] and grid[i][j-1] ==0 and grid[i][j-2] == 0: # if two equal tiles are separated with 2 zero
						grid[i][j] += grid[i][j]
						grid[i][j-3] = 0
						grid = deepcopy(grid)
				
				# if the row has a zero in it, we slide its elements to the right
				non_zero_idx = np.where(grid[i]!=0)[0]
				
				if len(non_zero_idx) != 4:
					non_zero = grid[i][non_zero_idx]
					new_row = np.zeros(4,int)
					new_row[4-len(non_zero):] = non_zero
					grid[i] = new_row

			#return (after reversed transposing operation)
			if action == "down":
				grid = grid.transpose()
			if action == "up":
				grid = np.flip(grid, 1).transpose()
			if action =="left":
				grid = np.flip(grid,1)

			# add random tile
			zero_idx = np.where(grid==0)
			tile_idx = np.random.randint(len(zero_idx[0]))
			idx = tuple([zero_idx[0][tile_idx], zero_idx[1][tile_idx]])
			grid[idx] = np.random.choice([2,4])

			return grid

		else :
			transposed = True
			if action == "down" :
				grid = grid.transpose()
			if action == "up":
				grid = np.flip(grid.transpose(), 1)
			if action == "left":
				grid = np.flip(grid,1)
			return updateGrid(grid, action, transposed)

	except:
		# if chosen action doesnt change the grid
		return grid_

def gridIsFinished(grid):
	"""
	return True if grid is finished, False otherwise
	"""
	len_ = grid.shape[0]
	# if one cell is empty, return False
	if len(np.where(grid==0)[0]) > 0:
		return False

	# if two cells aligned share the same value, return False
	else:
		for i in range(len_):
			for row in [grid[i], grid[:,i]]:
				# remove zeros, since they reprensent empty cells
				row = row[np.where(row != 0)]
				if (row[1:] == row[:-1]).sum() > 0:
					return False
		return True

class gameEnvironmentClass:
	"""
	Class for the 2048 game
	Includes random initialisation of grid, and step function to execute action in the environment
	"""
	def __init__(self):
		# empty grid
		self.grid = np.zeros((4,4), int)
		self.finished = False

		# add a first number (2 or 4) in a random place
		new_val = np.random.choice([2,4])
		new_idx = tuple(np.random.choice([0,1,2,3],2))
		self.grid[new_idx] = new_val

	def step(self, action):
		"""
		makes the step according to the chosen action
		:input:
			action (type=str), from list of actions ["down", "left", "right", "up"]
		"""
		# compute new grid, from the previous grid and the action choosen
		# if an action has no impact, we return same grid with one additionnal tile
		new_grid = updateGrid(self.grid, action)
		self.grid = new_grid

		# check if grid is finished
		self.finished = gridIsFinished(grid)