# -*- coding: utf-8 -*-
"""
main script :
learn optimal policy model
"""
import torch.optim as optim

from game import *
from neuralNetwork import *
from agent import *
from main import * 

def fillExperienceMemory(agent, memory, policy_model):
	"""
	agent play some moves, and fill the transition memory up to full capacity
	:inputs:
		agent (type = agentClass)
		memory (type = replayMemory)
		policy_model (policyNetworkClass)
	"""
	while len(memory) < memory.capacity: 
		# if game is finished, we reset the grid
		if gridIsFinished(agent.env.grid):
			agent.resetGameEnv() 

		# agent choose action, and interct with environment
		state = agent.env.grid
		action = agent.choose_action(policy_model)
		reward = agent.interact(action)
		new_state = agent.env.grid

		# fill memory from agent experience
		memory.push(state, action, new_state, reward)

def getError(memory, policy_model, gamma):
	"""
	memory -> error
	& format to pass as model input
	"""
	actions = ["down", "left", "right", "up"]
	memorySize = memory.capacity

	# retrive states
	transitions = memory.sample(memorySize)
	batch = Transition(*zip(*transitions))

	# coordinates of experienced (state, action)
	actionList = batch.action
	func = lambda x: actions.index(x)
	actionList = list(map(func, actionList))
	actionCoordinate = tuple(range(memorySize)), tuple(actionList) # index of the value for Q(s,a)

	# format
	stateArray = np.stack(batch.state).reshape(memorySize, -1)
	newStateArray = np.stack(batch.new_state).reshape(memorySize, -1)
	rewardArray = np.stack(batch.reward)

	# cast to torch type
	stateTensor =  torch.tensor(stateArray).float()
	newStateTensor = torch.tensor(newStateArray).float()
	rewardTensor = torch.tensor(rewardArray).float()

	# Bell formula for error, using neural network approximation of Q function
	# error for actions that are not used in the experience, is set to 0
	# error = Q(s, a) - ( R(s,a) + gamma * max(Q(s',a)) )
	error = torch.zeros((memorySize, len(actions)))
	error[actionCoordinate] = policy_model.forward(stateTensor)[actionCoordinate] - rewardTensor + gamma * policy_model.forward(newStateTensor).max(dim=1).values
	return error

if __name__=="__main__":
	# hyperparameters
	epsilon, gamma, alpha = 1, 1, 1
	batch_size = 500
	epoch = 10

	# instantiate policy model
	policy_model = policyNetworkClass()
	optimizer = optim.RMSprop(policy_model.parameters())
	criterion = nn.SmoothL1Loss()

	for e in range(epoch):
		print("epoch",e)
		# instantiate memory replay object
		memorySize = batch_size
		memory = replayMemory(memorySize)

		# instantiate agent
		agent = agentClass(epsilon, gamma, alpha)
		
		# fill memory with agent experiences
		fillExperienceMemory(agent, memory, policy_model)

		# experiences to loss values
		error = getError(memory, policy_model, gamma)
		loss = error.sum() # sum of errors, on batch

		# propagate error & update weights
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		print("loss",loss)
