#reference: https://github.com/Vict0rSch/deep_learning/tree/master/keras/recurrent
import dataset as data
import utilmodel as utm
import utilgraph as utg
import indicator as ind

import history as his
import numpy as np
import time
import os.path
import datetime
import sys

from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop
from sklearn.model_selection import train_test_split
from keras.regularizers import l2, activity_l2

# fix random seed for reproducibility
#np.random.seed(7)

MODEL_PATH = 'model'
if os.path.exists(MODEL_PATH) == False:
	os.makedirs(MODEL_PATH)

def buildModel(X_shape, modelName):	
	# batch size, sequence num, features
	_, sequence, features = X_shape	
	model = Sequential()
	model.add(LSTM(input_dim=features, 
				input_length=sequence,
				output_dim=50,
				dropout_W=0.1, dropout_U=0.1,
				return_sequences=True))
	# now model.output_shape == (None, 60)
	# note: `None` is the batch dimension.
	model.add(Activation("relu"))
	
	model.add(LSTM(50, dropout_W=0.1, dropout_U=0.1, return_sequences=True))
	model.add(Activation("relu"))
	
	model.add(LSTM(50, dropout_W=0.1, dropout_U=0.1, return_sequences=False))
	# now model.output_shape == (None, 60)	
	model.add(Activation("relu"))
	
	model.add(Dense(output_dim=100))
	model.add(Activation("relu"))
	model.add(Dropout(0.5))	# reduce overfitting
	
	model.add(Dense(output_dim=100))
	model.add(Activation("relu"))
	model.add(Dropout(0.5))	# reduce overfitting
	
	model.add(Dense(output_dim=1))
	model.add(Activation("sigmoid"))
	
	try:
		if os.path.exists(modelName):
			print("Loaded model from disk: %s " % (modelName))
			model.load_weights(modelName)	
	except Exception as inst:
		print(inst)
	
	#adam=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)	
	#keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
	optm=RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0001)	
	start = time.time()
	# algorithim to train models use
	# compute loss with function: binary_crossentropy or loss crossentropy (Also known as logloss)
	model.compile(optimizer=optm,
			  loss='binary_crossentropy',
			  metrics=['accuracy'])

	sec = datetime.timedelta(seconds=int(time.time() - start))			  
	print ("Compilation Time : ", str(sec))	
	return model

def trainModel(symbolName, Xtrain, Ytrain, Xtest, Ytest, epochs):
	history = his.TrainingHistory()
	h5_file = os.path.join(MODEL_PATH,'%s_weights.h5' % (symbolName))
	json_file = os.path.join(MODEL_PATH,'%s_structure.json' % (symbolName))
	
	global_start_time = time.time()	
	# Shape: example number, sequence lenght, feature number
	model = buildModel(Xtrain.shape, h5_file)
		
	# Learning	
	#model.fit(Xtrain, Ytrain, batch_size=500, nb_epoch=epochs, verbose=2, validation_split=0.05 )
	print('Wait training ... ')
	model.fit(Xtrain, Ytrain, batch_size=500, 
				nb_epoch=epochs, 
				validation_data=(Xtest, Ytest),
				verbose=2,
				callbacks=[history])
				
	sec = datetime.timedelta(seconds=int(time.time() - global_start_time))
	print ('Training duration : ', str(sec))	
		
	# After train, the model is saved to a file
	utm.saveTrainedModel(model, json_file, h5_file)	
	return model, history

def filterSignal(Ydigits):
	signal = np.copy(Ydigits)	
	for index in range(1, len(Ydigits)-1):
		previous = Ydigits[index-1]
		current = Ydigits[index]
		future = Ydigits[index+1]
		signal[index+1] = utm.getSignal(previous, current, future) # future		
	return np.reshape(signal,(-1,1))

import pickle
symbolList = pickle.load( open( "symbol_list.p", "rb" ) )
symbolList = symbolList[0:20]
symbol = np.random.choice(symbolList) # select symbols randomly
symbol = 'SET'

X, Y, close = data.getTrainData_1(symbol, '2014-01-01', '2017-03-10');	
numDays, numIndicator = X.shape
assert numDays == Y.shape[0]
assert numDays == close.shape[0]

count_upTrend = np.sum(Y)
if count_upTrend < 0.4 * count_upTrend:
	print('Information not balance between 1 and 0 in Y datasets')
	print('Number uptrend: %s , not: %s' % (count_upTrend , len(Y)-count_upTrend))
	sys.exit(1) 

signal = filterSignal(Y)
print("Select securities symbol:", symbol)
print("Total days: ", numDays)
utg.plot1ColLine(symbol, close, signal,'Signal: ' + symbol)

sequence_length = 30 # lenght of sequence input
Xtrain, Xtest, Ytrain, Ytest = data.packSeqData(X, Y, sequence_length)	
nsample, seq, nfeature = Xtrain.shape 
assert seq == sequence_length
assert Ytrain.shape[0] == nsample
assert numIndicator == nfeature

model, history = trainModel(symbol, Xtrain, Ytrain, Xtest, Ytest, epochs=1)
print("Input size (model) : ", Xtrain.shape)
utg.plotLine(history.losses) # plot error graph

# evaluate the model after trained
scores = model.evaluate(Xtrain, Ytrain, verbose=0)
print("Evaluate model with %s: %.2f%%" % (model.metrics_names[0], scores[0]))
print("Evaluate model with %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# for test
scores = model.evaluate(Xtest, Ytest, verbose=0)
print('After learning: ',symbol)
print('Testing less: %.2f%%"' % (scores[0]))
print('Testing accuracy: %.2f%%"' % (scores[1]*100))

predEncoded = utm.encoded(model.predict(Xtrain))
predTestEncoded = utm.encoded(model.predict(Xtest))
print("Targt:\n", Ytest.reshape(-1))
print("But predict (test):\n", predTestEncoded.reshape(-1))

trainPrice = close.ix[0:predEncoded.size]
testPrice = close.ix[-predTestEncoded.size:]
signal = filterSignal(predEncoded)

utg.plot1ColLine(symbol, trainPrice, signal, 'Predict signal: ' + symbol)
print(ind.compute_gain(trainPrice, filterSignal(Ytrain)))

        



