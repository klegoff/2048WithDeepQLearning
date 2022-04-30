# -*- coding: utf-8 -*-
"""
main script :
learn optimal policy model
"""
import torch.optim as optim

from game import *
from neuralNetwork import *
from agent import *

if __name__=="__main__":
	# hyperparameters
	epsilon, gamma, alpha = 1, 1, 1

	policy_model = policyNetworkClass()
	optimizer = optim.RMSprop(policy_model.parameters())

	# instantiate memory replay object
	memorySize = 1000 
	memory = replayMemory(memorySize)

	# instantiate agent
	agent = agentClass(epsilon, gamma, alpha)

	#gameContinues(agent.env.grid)


