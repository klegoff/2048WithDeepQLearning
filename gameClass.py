# -*- coding: utf-8 -*-
"""
Contains 2048 implementation as an interactive envrionment
"""
from copy import deepcopy
import numpy as np

def updateGrid(grid_, action,transposed = False):
	"""
	Makes the move of the grid and generate randomly a new number (2 or 4) in an empty case
	use recursivity to exploit the "right" move
	"""
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


class gameEnvironment:
	"""
	Class for the 2048 game
	Includes random initialisation of grid, and step function to execute action in the environment
	"""
	def __init__(self):
		# empty grid
		self.grid = np.zeros((4,4), int)

		# add a first number (2 or 4) in a random place
		new_val = np.random.choice([2,4])
		new_idx = tuple(np.random.choice([0,1,2,3],2))
		self.grid[new_idx] = new_val

	def step(self, action):
		"""
		makes the step according to the chosen action
		:input:
			action (type=str), from list of actions ["down", "left", "right", "up"]
		:outputs:
			reward (type = int), reward of the action
			new_grid (type = np.array), new grid of the game state
		"""
		old_grid = deepcopy(self.grid)
		# compute new grid, from the previous grid and the action choosen
		new_grid = updateGrid(old_grid, action)
		self.grid = new_grid
		reward = computeReward(old_grid, new_grid)
		return reward, new_grid

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