# -*- coding: utf-8 -*-
"""
main script :
learn optimal policy model
"""
import os
import uuid
import json
import pickle
import torch
import torch.optim as optim

from game import *
from neuralNetwork import *
from agent import *

# check if cuda device is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

actions = ["down", "left", "right", "up"]


def runExperiment(hyparameters):
	"""
	run training of model, according to the input hyperparameters
	save results in model/run_id/
	"""
	# id for the current run, unique
	run_id = str(uuid.uuid1())

	### instantiate objects
	state_action_value_model = DQN().to(device)
	optimizer = optim.RMSprop(state_action_value_model.parameters())
	agent = agentClass(hyparameters["epsilon"])
	criterion = nn.MSELoss()

	# some object for post-training analysis
	lossDict = {}
	modelWeightsDict = {}

	for e in range(hyparameters["epoch"]):

		### Agent play some games to fill replay memory, and we retrieve a sample of these experiments

		# instantiate memory replay object
		memory = replayMemory(hyparameters["memorySize"])
		
		# fill memory with agent experiences
		memory.fill(agent, state_action_value_model)
		
		# retrive a sample of replayMemory
		stateTensor, newStateTensor, rewardTensor, actionCoordinate = memory.sampleAndFormat(hyparameters["sampleSize"])

		### from a sample of experiences, we compute the error, and backpropagate

		# Bell formula for error, using neural network approximation of Q function
		# error for actions that are not used in the experience, is set to 0
		# error = Q(s, a) - ( R(s,a) + gamma * max(Q(s',a)) )
		error = torch.zeros((hyparameters["sampleSize"], len(actions)))
		error[actionCoordinate] = state_action_value_model.forward(stateTensor)[actionCoordinate] - rewardTensor + hyparameters["gamma"] * state_action_value_model.forward(newStateTensor).max(dim=1).values
		target = torch.zeros((hyparameters["sampleSize"], len(actions))) # we want the error to converge to zero
		loss = criterion(error, target)
		lossDict[e] = loss.detach().numpy()[()] # store loss
		loss = loss.to(device)
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

if __name__=="__main__":
    
	# hyperparameters
	hyparameters = {"epsilon" : 0.1, #ratio exploration / exploitation
					"gamma": 1, # relative importance of future reward
					"memorySize" : 15000, # size of replay memory
					"sampleSize" : 1000, # number of experience we learn on, randomly sampled on replay memory
					"epoch" : 100}

	runExperiment(hyparameters)


