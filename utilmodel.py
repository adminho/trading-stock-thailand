from __future__ import print_function
import os
import sys
import tarfile
from six.moves.urllib.request import urlretrieve

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from keras.models import model_from_json

import pandas as pd
import os.path
import numpy as np

def getPCAvalues(X, n_components):
	estimator = PCA(n_components=n_components)
	return estimator.fit_transform(X)	

def loadTrainedModel(jsonFile, h5File):
	if os.path.exists(jsonFile)== False or os.path.exists(h5File) == False:
		return None
	
	# use backup model
	# First, load a model structure from json file
	json_file = open(jsonFile, 'r')
	loaded_model_json = json_file.read()
	json_file.close()
		
	# Then, load weights into the model
	model = model_from_json(loaded_model_json)		
	model.load_weights(h5File)
	print("Loaded model from disk: %s and %s" % (jsonFile, h5File))
	return model

def saveTrainedModel(model, jsonFile, h5File):
	# serialize model to JSON
	model_json = model.to_json()
	with open(jsonFile, "w") as json_file:
		json_file.write(model_json)		
	# serialize weights to HDF5
	model.save_weights(h5File, overwrite=True)
	print("Saved model to disk: %s and %s" % (jsonFile, h5File))

def encoded(predicted):
	return (predicted > 0.5) *1
	
def get_buy_sell(pevious, current, future):	
	if future == 1: # has uptrend in the 14 days ahead
		if pevious == 1 and current == 1:
			return 1 # confrim singal as buy
		else:
			return current # not confirm
			
	if future == 0: # has not up trend
		if pevious == 0 and current == 0:
			return 0 # confrim singal as sell
		else:
			return current # not conrim
			
def getSignal(Ydigits):
	signal = np.copy(Ydigits)	
	for index in range(1, len(Ydigits)-1):
		previous = Ydigits[index-1]
		current = Ydigits[index]
		future = Ydigits[index+1]
		signal[index+1] = get_buy_sell(previous, current, future) # future		
	return np.reshape(signal,(-1,1))

# Borrowed code: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/udacity/1_notmnist.ipynb
url = 'http://siamchart.com/stock/download.php'
last_percent_reported = None
data_root = '.' # Change me to store data elsewhere

def download_progress_hook(count, blockSize, totalSize):
	"""A hook to report the progress of a download. This is mostly intended for users with
	slow internet connections. Reports every 5% change in download progress.
	"""
	global last_percent_reported
	percent = int(count * blockSize * 100 / totalSize)

	if last_percent_reported != percent:
		if percent % 5 == 0:
			sys.stdout.write("%s%%" % percent)
			sys.stdout.flush()
		else:
			sys.stdout.write(".")
			sys.stdout.flush()
      
		last_percent_reported = percent
        
def maybe_download(filename, force=False):
	"""Download a file if not present, and make sure it's the right size."""
	dest_filename = os.path.join(data_root, filename)
	if force or not os.path.exists(dest_filename):
		print('Attempting to download:', filename) 
		filename, _ = urlretrieve(url, dest_filename, reporthook=download_progress_hook)
		print('\nDownload Complete!')
	statinfo = os.stat(dest_filename)
	print('Size :' , statinfo.st_size)
	return dest_filename

#csv_filename = maybe_download('set-archive_EOD.zip')

