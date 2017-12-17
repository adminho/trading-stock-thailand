import os.path
import random
import numpy as np
import pandas as pd
from collections import deque

from keras import optimizers
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import BatchNormalization
import utilmodel as utm

# directory for saving model files
MODEL_PATH = 'model'
if os.path.exists(MODEL_PATH) == False:
	os.makedirs(MODEL_PATH)

class DeepQAgent:
	# config parameters here
	GAMMA  = 0.95  			# parameter for calculate Q-function
	epsilon  = 1.0  			# exploration rate
	MAX_REPLAY_MEMORY = 200 	# maximum of previous transitions (previous states) to remember		
	OBSERVATION = 100.   		# observe before training
	BATCH_SIZE = 100			# batch size for training my neural network model
		
	# store the previous transitions (previous states) in replay memory (use deque)
	D = deque()		

	def __init__(self,env):
		self.h5_file = os.path.join(MODEL_PATH,'%s_weights.h5' % (env.symbol))
		self.model = self._buildModel(env.num_features, env.num_action)
		self.json_file = os.path.join(MODEL_PATH,'%s_structure.json' % (env.symbol))
		self.num_action = env.num_action 	# 2 action: BUY or SELLs
		
	# This is neural network model for the the Q-function		
	def _buildModel(self, num_features, num_output):				
		model = Sequential()				
		model.add(Dense(input_shape=(num_features, ), units=8))
		model.add(BatchNormalization(trainable = True))	 		
		model.add(Dense(units=8, activation='relu'))
		model.add(BatchNormalization(trainable = True))
		model.add(Dense(num_output, activation='linear'))
		
		print("\nBuild model...")
		print(model.summary())		
		try:
			if os.path.exists(self.h5_file):
				print("\nLoaded model(weights) from file: %s" % (self.h5_file))
				model.load_weights(self.h5_file)	
		except Exception as inst:
			print(inst)
	
		opt = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) 
		model.compile(optimizer=opt, loss='mse', metrics=['accuracy'])
		return model
		
	def clearMemory(self):
		self.D.clear()
								
	def getAction(self, state):		
		if random.random() <= self.epsilon:
			# With probability ϵ (epsilon) select random action
			# Exploration method (greedy)
			return random.randrange(self.num_action)		
			
		# Otherwise select the best action from Q(now_state, all_action)
		# Exploit method found actions that there is maximum of Q score
		best_index, _ = self.getBestAction(state)
		return best_index
			
	def getBestAction(self, state):
		Q_scores = self.model.predict(state)
		# np.argmax() will return the indices of the maximum values 
		best_index = np.argmax(Q_scores)
		return	best_index, Q_scores
		
	def saveExperience(self, state, action_index, reward, nextstate, terminate):
		# store the transition (states) in D (replay memory)
		self.D.append((state, action_index, reward, nextstate, terminate))
		if len(self.D) > self.MAX_REPLAY_MEMORY:
			self.D.popleft() # remove the oldest states
			
	def _updateQScore(self, minibatch):	
		X_train, y_train = [], []
		for memory in minibatch:
			state, action_index, reward, nextstate, terminate = memory														
			Q_scores = self.model.predict(state)[0] 		# output shape is [1, num_action]
			
			if terminate:
				Q_scores[action_index] = reward
			else:
				# formula for Q-function
				# Q(state, action) = reward + gamma* max(Q(state_next, all_action))				
				allNextQ = self.model.predict(nextstate)    	# output shpe is [1, num_action]				
				Q_scores[action_index] = reward + self.GAMMA * np.max(allNextQ)				
							
			# the first value on top while the lastest value at bottom			
			X_train.append(state)
			y_train.append(Q_scores)	
			
		# Train my neural network to remember (learning) new Q Scores
		X_train = np.squeeze(np.array(X_train), axis=1)
		y_train = np.array(y_train)		
		# Single gradient update over one batch of samples.
		# equals this command ==> self.model.fit(X_train, y_train, batch_size=self.BATCH_SIZE, epochs=1, verbose=0)
		self.model.train_on_batch(X_train, y_train)

	def replayExperienceWhen(self, step_observe):
		#Now we do the experience replay
		if step_observe  > self.OBSERVATION:		
			#Train my neural network model when timestep more than observing					
			#sample a minibatch to train 
			minibatch = random.sample(self.D, self.BATCH_SIZE)
			# Update all new Q scores (based on old experiences that stored)
			self._updateQScore(minibatch)					   
        
	def reduceExplore(self, constant):
		if self.epsilon > 0.1: #decrement epsilon (exploration rate) over time
			self.epsilon -= (1.0/constant)
		
	def saveModel(self):
		# save the structure of model and weights into json file and h5 file
		utm.saveTrainedModel(self.model, self.json_file, self.h5_file)				
