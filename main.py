# -*- coding: utf-8 -*-
"""
main script :
learn optimal policy model
"""
import torch.optim as optim

from game import *
from neuralNetwork import *
from agent import *

def fillExperience(agent, memory, policy_model):
	"""
	agent play some moves, and fill the transition memory up to full capacity
	"""
	# fill memory from agent experiences
	while len(memory) < memory.capacity: 
		# if game is finished, we reset the grid
		if not gameContinues(agent.env.grid):
			agent.resetGameEnv() 
		state = agent.env.grid
		action = agent.choose_action(policy_model)
		reward = agent.interact(action)
		new_state = agent.env.grid
		memory.push(state, action, new_state, reward)

if __name__=="__main__":
	# hyperparameters
	epsilon, gamma, alpha = 1, 1, 1

	# instantiate policy model
	policy_model = policyNetworkClass()
	optimizer = optim.RMSprop(policy_model.parameters())

	# instantiate memory replay object
	memorySize = 1000 
	memory = replayMemory(memorySize)

	# instantiate agent
	agent = agentClass(epsilon, gamma, alpha)

	# fill memory with agent experiences
	fillExperience(agent, memory, policy_model)


