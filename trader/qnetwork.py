import numpy as np
import os.path
import random

from keras import optimizers
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers import LSTM, Conv1D, Flatten, Input, Activation, Dense,BatchNormalization

# directory for saving model files
MODEL_PATH = 'model'
if os.path.exists(MODEL_PATH) == False:
	os.makedirs(MODEL_PATH)
		
class QNetwork:
	def __init__(self, env, indicators=6, days=60, action_size=3, seed=1412):	
		'''
		np.random.seed(seed)		
		model = Sequential()				
		model.add(Dense(input_dim=indicators, units=128))
		model.add(BatchNormalization(trainable = True))	 		
		model.add(Dense(units=128, activation='relu'))
		model.add(BatchNormalization(trainable = True))
		model.add(Dense(units=action_size, activation='linear'))
		'''
		self.symbol = env.symbol
		
		random.seed(seed)	
		model = Sequential()
		model.add(Conv1D(filters=30, kernel_size=3, strides=1, padding='same', input_shape=(indicators, days)))	
		# (batch, new_steps, filters)
		model.add(Activation('relu'))	
		model.add(LSTM( units=30, return_sequences=True) ) # (batch_size, timesteps, units)
		model.add(Conv1D(filters=15, kernel_size=3, strides=2, padding='same'))
		model.add(Activation('relu'))	
		model.add(LSTM( units=15, return_sequences=True) )		
		model.add(Flatten())		
		model.add(Dense(units=128, activation='relu'))
		model.add(Dense(units=action_size, activation='linear'))
		
		print("\nBuild model...")
		print(model.summary())
		try:
			model_h5 = os.path.join( MODEL_PATH, '%s_weights.h5' % (self.symbol) )			
			if os.path.exists(model_h5):
				print("\nLoaded feature (weights) from file: %s" % (model_h5))
				model.load_weights(model_h5)
				
		except Exception as inst:			
			print(inst)
	
		opt = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) 
		model.compile(optimizer=opt, loss='mse', metrics=['accuracy'])
		self.model = model
			
	def output(self, state):		
		Q_scores = self.model.predict(state)[0] # output shape is [1, num_action] 
		return Q_scores
		
	def learn(self, transition_list, GAMMA = 0.99):		
		X_train, y_train = [], []
		for transition in transition_list:
			state, action, reward, next_state, terminate = transition
			Q_scores = self.model.predict(state)[0] # output shape is [1, num_action]
			action_index = action
			if terminate:# last state
				Q_scores[action_index] = reward
			else:
				# formula for Q-function
				# Q(state, action) = reward + gamma* max(Q(state_next, all_action))				
				allNextQ = self.model.predict(next_state)[0]    	# output shpe is [1, num_action]				
				Q_scores[action_index] = reward + GAMMA * np.max(allNextQ)				
							
			# the first value on top while the lastest value at bottom			
			X_train.append(state)
			y_train.append(Q_scores)	
			
		# Train my neural network to remember (learning) new Q Scores
		X_train = np.squeeze(np.array(X_train), axis=1)
		y_train = np.array(y_train)		
		# Single gradient update over one batch of samples.
		# equals this command ==> self.model.fit(X_train, y_train, batch_size=self.BATCH_SIZE, epochs=1, verbose=0)
		#self.model.train_on_batch(X_train, y_train)
		self.model.fit(X_train, y_train, epochs=30, verbose=0)
		return y_train

	def save(self):
		h5_file = os.path.join(MODEL_PATH,'%s_weights.h5' % (self.symbol))		
		json_file = os.path.join(MODEL_PATH,'%s_structure.json' % (self.symbol))		
		# serialize model to JSON
		model_json = self.model.to_json()
		with open(json_file, "w") as file:
			file.write(model_json)		
			# serialize weights to HDF5
			self.model.save_weights(h5_file, overwrite=True)
		print("Save model to your disk: %s and %s" % (json_file, h5_file))		
		