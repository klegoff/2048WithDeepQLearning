# -*- coding: utf-8 -*-
"""
main script :
learn optimal policy model
"""
import os
import uuid
import json
import pickle
import torch.optim as optim

from game import *
from neuralNetwork import *
from agent import *

def fillExperienceMemory(agent, memory, state_action_value_model):
	"""
	agent play some moves, and fill the transition memory up to full capacity
	:inputs:
		agent (type = agentClass)
		memory (type = replayMemory)
		state_action_value_model (class = DQN)
	"""
	while len(memory) < memory.capacity: 
		# if game is finished, we reset the grid
		if gridIsFinished(agent.env.grid):
			agent.resetGameEnv()

		# agent choose action, and interct with environment
		state = agent.env.grid
		with torch.no_grad():
			action = agent.choose_action(state_action_value_model)
		new_state = agent.env.grid

		# compute reward
		reward = reward2(state, new_state)

		# fill memory from agent experience
		memory.push(state, action, new_state, reward)

def computeLoss(memory, sampleSize, state_action_value_model, criterion, gamma):
	"""
	memory sample -> data format -> loss values
	"""
	actions = ["down", "left", "right", "up"]
	memorySize = memory.capacity

	# retrive states
	transitions = memory.sample(sampleSize)
	batch = Transition(*zip(*transitions))

	# coordinates of experienced (state, action)
	actionList = batch.action
	func = lambda x: actions.index(x)
	actionList = list(map(func, actionList))
	actionCoordinate = tuple(range(sampleSize)), tuple(actionList) # index of the value for Q(s,a)

	# format
	stateArray = np.stack(batch.state).reshape(sampleSize, -1)
	newStateArray = np.stack(batch.new_state).reshape(sampleSize, -1)
	rewardArray = np.stack(batch.reward)

	# cast to torch type
	stateTensor =  torch.tensor(stateArray).float()
	newStateTensor = torch.tensor(newStateArray).float()
	rewardTensor = torch.tensor(rewardArray).float()

	# Bell formula for error, using neural network approximation of Q function
	# error for actions that are not used in the experience, is set to 0
	# error = Q(s, a) - ( R(s,a) + gamma * max(Q(s',a)) )
	error = torch.zeros((sampleSize, len(actions)))
	error[actionCoordinate] = state_action_value_model.forward(stateTensor)[actionCoordinate] - rewardTensor + gamma * state_action_value_model.forward(newStateTensor).max(dim=1).values
	
	# loss :
	target = torch.zeros((sampleSize, len(actions)))
	#print(error.shape, target.shape)
	loss = criterion(error, target)

	return loss


if __name__=="__main__":
    
    # id for the current run, unique
	run_id = str(uuid.uuid1())
    
	# hyperparameters
	hyparameters = {"epsilon" : 0.1, #ratio exploration / exploitation
                    "gamma": 1, # relative importance of future reward
                    "memorySize" : 10000, # size of replay memory
                    "sampleSize" : 500, # number of experience we learn on, randomly sampled on replay memory
                    "epoch" : 50}
    
    
	### instantiate objects
	state_action_value_model = DQN()
	optimizer = optim.RMSprop(state_action_value_model.parameters())
	agent = agentClass(hyparameters["epsilon"])
	criterion = nn.MSELoss()

	# some object for post-training analysis
	lossDict = {}
	modelWeightsDict = {}

	for e in range(hyparameters["epoch"]):

		# instantiate memory replay object
		memory = replayMemory(hyparameters["memorySize"])
		
		# fill memory with agent experiences
		fillExperienceMemory(agent, memory, state_action_value_model)

		# from a sample of experiences, we compute the error
		loss = computeLoss(memory, hyparameters["sampleSize"], state_action_value_model,criterion, hyparameters["gamma"])
		lossDict[e] = loss.detach().numpy()[()]
		print("epoch",e,"Loss=",lossDict[e])

		# propagate error & update weights
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# add new model_state to dict
		modelWeightsDict[e] = state_action_value_model.state_dict()

	# save model state, training loss & hyperparameters
	modelPath = "model/" + run_id + "/"
	try :
		os.mkdir(modelPath) # create directory if needed
	except:
		pass
    
	with open(modelPath + "modelWeightsDict.pickle", "wb") as f:
		pickle.dump(modelWeightsDict, f)
        
	with open(modelPath + "/lossDict.pickle", "wb") as f:
		pickle.dump(lossDict, f)
        
	with open(modelPath + "/hyperparameters.json", "w") as f:
		json.dump(hyparameters, f)
    
        
        

    
