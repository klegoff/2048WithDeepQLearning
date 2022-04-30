# -*- coding: utf-8 -*-
"""
main script :
learn optimal policy model
"""
import torch.optim as optim

from gameClass import *
from dqnClass import *
from agentClass import *

if __name__=="__main__":
	# hyperparameters
	epsilon, gamma, alpha = 1, 1, 1

	policy_model = DQN()
	optimizer = optim.RMSprop(policy_model.parameters())

	# instantiate agent
	agent_ = agent(epsilon, gamma, alpha)



