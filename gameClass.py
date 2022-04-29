# -*- coding: utf-8 -*-
"""
Main script of the project
Contains environment & agent classes
"""
import numpy as np
from copy import deepcopy


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
	Class for the 2048 grid
	contains all we need to interact with it
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
		"""
		self.grid = updateGrid(self.grid, action)
		reward = -1
		state = self.grid
		return reward, state

if __name__ == '__main__':

	grid = np.random.randint(2,size=(4,4))
	print(grid)
	grid = updateGrid(grid, "up")
	print(grid)