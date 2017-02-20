from os import listdir, makedirs, remove
from os.path import isfile, join, exists
import numpy as np
import pandas as pd
import shutil	
import utility as util
import indicator as ind

def getFileNameInDir(path):
	onlyFiles = [ join(path,f) for f in listdir(path) if isfile(join(path, f)) ]
	return onlyFiles
	
def getHeaderFile(eodFiles):
	for f in eodFiles:
		df = pd.read_csv(f)
		return df.columns.values			# read a header file from first file
		
def getStockData(eodFiles, selectedSmbol = []):			
	dict = {}	# empty dictionary
	
	for i, f in enumerate(eodFiles):# read all file		
		df = pd.read_csv(f)
		total_row = len(df.index)	
		all_data = df.values;
				
		#range(start, stop, step)
		for row in range(total_row-1, -1, -1):	# reveserse form range(0, total_row)
			symbol = all_data[row][0]			# name of stock in frist column			
			if len(selectedSmbol)!=0 :
				if not symbol in selectedSmbol: continue			
			
			if symbol == "COM7": symbol = "COM7_" # fix bug for this symbol only
			
			current_row = all_data[row]				
			if symbol in dict: 	# There are symbol data in dictionary
				dict[symbol].append(current_row)	# append new data to old data
			else: # no symbol data in dictionary
				dict[symbol] = [current_row]				
		
		if(i%500 == 0):	# for debug			
			print("Reading total files : {} ....".format(i))						
		
	return dict

def clearDir(dirPath):
	if exists(dirPath):			
		shutil.rmtree(dirPath)
	
	makedirs(dirPath)	

DIR_SEC_CSV = "sec_csv"
# download: http://siamchart.com/stock/			
EOD_file = "set-archive_EOD_UPDATE"
def createSymbolCSV(start_idex, outputPath=DIR_SEC_CSV):
	eodFiles = data.getFileNameInDir(EOD_file)	
	eodFiles = eodFiles[-1 * start_idex:]		# select files latest N days
	
	clearDir(outputPath) # delete old files
	
	dataStock  = data.getStockData(eodFiles)	
	headers = getHeaderFile(eodFiles)	# Read header of CSV files
	columnNames = { index:value  for index, value in enumerate(headers)}
	
	# write data to csv files seperate file name follow symbol names of security	
	count = 0
	for key, allRow in dataStock.items():		
		df = pd.DataFrame(allRow)
		df.rename(columns=columnNames, inplace=True) # change column names in data frame: convert from number to symbol names				
		
		fileName = "{}.csv".format(join(outputPath, key))		
		df.to_csv(fileName, index = False) # write data into CSV file (without index)
				
		if(count%3000 == 0):	# for debug			
			print("Writing total files : {} ....".format(count))						
		count+=1;

output_dataset="D:\python\output_dataset"
def writeDataSetY(df, symbols, predict=0, window=20, output_dir=output_dataset):
	symbols = df.columns.values	
	df_sr = ind.create_dataframe_SR(df, symbols)		
	df_result = ind.isStrongSR(df_sr)
	
	# By defult, Use data of 20 days (backward) to predict 15 days (foward)
	# Because I would like to know sharpe ratio that are more than 0 continue 10 days or not
	shift = window + predict -1
	df_result = df_result.shift(-1*shift)
	df_result=df_result.dropna(0)
	df_result.to_csv(join(output_dir, "@Dataset_Y.csv"), index_label='<DTYYYYMMDD>') 		
		
def writeDataSetX(df, output_file, predict=0, window=20):	
	#By defult, Use data of 20 days to compute
	total_row = df.shape[0]
	length = total_row-window + 1 - predict # ลบกับ [predict] N days
	total_data = df.values			
	temp_all_data=[]
	
	for row in range(0, length):		
		temp = total_data[row: row+window] # slice data
		temp = np.reshape(temp,-1)
		
		if row == 0:
			temp_all_data = temp	
		else:
			temp_all_data = np.vstack([temp_all_data,temp])
	
	column_names = []	
	for i in range(1,  window + 1):
		for col in df.columns:		
			column_names = np.append(column_names, '{}{}'.format(col,i))
	
	list_date=df.index[0:length]	
	df_result = pd.DataFrame(temp_all_data, columns=column_names, index=list_date)	
	df_result.to_csv(output_file, index_label='<DTYYYYMMDD>') 
	return df_result

def createDataFrame(symbol, dates, column_name, csv_dir):
	df_main = pd.DataFrame(index=dates)	# empty data frame that has indexs as dates
	csv_file = join(csv_dir, "{}.csv".format(symbol)) 	
	
	if not isfile(csv_file):
		print("Can't found: ",csv_file)
		return df_main # return empty data frame
			
	column_date_name = '<DTYYYYMMDD>'
	if not column_date_name in column_names:		
		column_names = np.append([column_date_name], column_names)
		
	df_csv = pd.read_csv(csv_file, index_col=column_date_name,
			parse_dates=True, usecols=column_names, na_values=['nan'])
		
	df_main = df_main.join(df_csv)
	return df_main.dropna(0)

def prepareDataSet(symbols, dates, csv_dir=DIR_SEC_CSV,output_dir=output_dataset):		
	clearDir(output_dir)
	column_names = ['<CLOSE>', '<VOL>']
	column_date = '<DTYYYYMMDD>'
	
	count = 0
	for sym in symbols:		
		df_X = createDataFrame(sym, dates, column_names, csvdir)							
		#df_X.to_csv(join(output_dir, "{}_test_check.csv".format(sym))) 	
		df_norm = util.normalize_data(df_X)		
		#df_norm.to_csv(join(output_dir, "{}_test_check_norm.csv".format(sym))) #ไม่มี index ในไฟล์
		writeDataSetX(df_norm, join(output_dir, "{}_X.csv".format(sym)))

		if(count%20 == 0):
			print("Writing total files : {} ....".format(count))			
		count+=1;		
		
	df_Y = util.loadPriceData(symbols, dates)			
	writeDataSetY(df_Y, symbols)
	
def getTrainData(symbol, dates, periods=14):
	# day periods that predict a price is new high or not
	price = util.loadPriceData([symbol], dates)	
	# skip periods day latest
	price_sliced = price[0: len(price) - periods] 
		
	roc = ind.roc(price_sliced)
	rsi = ind.rsi(price_sliced)/10 			# normalize
	sr = ind.create_dataframe_SR(price_sliced, [symbol])
	myRatio = ind.get_myRatio(price_sliced)
	slope = ind.fitLine(price_sliced)
	
	ema26, ema12, MACD = ind.average_convergence(price_sliced)
	MACD  = MACD # normalize
	# bbands = ind.get_BBANDS(close, symbol)
		
	# rename column
	#roc.rename(columns={symbol:'ROC'},inplace=True)
	rsi.rename(columns={symbol:'RSI'},inplace=True)
	sr.rename(columns={symbol:'SR'},inplace=True)
	#close.rename(columns={symbol:'CLOSE'},inplace=True)
	#ema26.rename(columns={symbol:'EMA26'},inplace=True)
	#ema12.rename(columns={symbol:'EMA12'},inplace=True)
	MACD.rename(columns={symbol:'MACD'},inplace=True)
	myRatio.rename(columns={symbol:'MY'},inplace=True)
	slope.rename(columns={symbol:'SLOPE'},inplace=True)

	volume = util.loadVolumeData(symbol, dates)
	# skip periods day latest
	volume_sliced = volume.ix[0: len(volume) - periods] 
	assert len(volume_sliced) == len(price_sliced)
	obv = ind.OBV(volume_sliced, price_sliced)

	#obv = util.normalize_data(obv)
	obv_rsi = ind.rsi(obv)/10 	# calcuate momentum and normalize
	obv_rsi.rename(columns={symbol:'OBV_RSI'},inplace=True)

	Xtrain = pd.DataFrame(index=price_sliced.index)
	#Xtrain = Xtrain.join(roc['ROC']) # I think it not work
	Xtrain = Xtrain.join(rsi['RSI'])
	Xtrain = Xtrain.join(sr['SR'])
	#Xtrain = Xtrain.join(close['CLOSE'])
	#Xtrain = Xtrain.join(ema26['EMA26'])
	#Xtrain = Xtrain.join(ema12['EMA12'])
	Xtrain = Xtrain.join(MACD['MACD'])
	#Xtrain = Xtrain.join(bbands['UPPER'])
	#Xtrain = Xtrain.join(bbands['LOWER'])
	Xtrain = Xtrain.join(myRatio['MY'])
	Xtrain = Xtrain.join(slope['SLOPE'])
	Xtrain = Xtrain.join(obv_rsi['OBV_RSI'])

	newHight = ind.isNewHigh(price, periods=periods)
	Ylogit = 1*newHight[symbol]  # 1 is True (new hight) , 0 is False

	#close = util.normalize_data(price_sliced)	# normalize
	
	# skip at head row, avoid NaN
	Xtrain = Xtrain[30:]
	Ylogit = Ylogit[30:]
	close = price_sliced.ix[30:][symbol]
	closeDf = pd.DataFrame(close,columns=[symbol])	
	return Xtrain, Ylogit, closeDf
		
def _getStockSymbol(csvFilename):	
	df = pd.read_csv(csvFilename, encoding = "TIS-620")
	symbols = np.append(['SET'], df.values.T[0])
	
	symbols = [ sym.replace('COM7','COM7_') for sym in symbols ] # fix bug only
	return symbols

def getSETHDSymbol():	
	return _getStockSymbol("list_SETHD.csv")	
	
def getSET100Symbol():	
	return _getStockSymbol("list_SET_100.csv")	

def getSET50Symbol():		
	return _getStockSymbol("list_SET_50.csv")

def getSubIndustrySymbol():		
	return _getStockSymbol("list_Index_Sub_Instrudry.csv")
	
def getAllSymbol(path=DIR_SEC_CSV):
	symbols = [ f.replace(".csv","") for f in listdir(path) if isfile(join(path, f)) ]	
	#symbols2 = ['SET']
	#symbols2 = np.append(symbols2, getSET100Symbol())
	#symbols2 = np.append(symbols2, getSETHDSymbol())
	#symbols2 = np.append(symbols2, getSubIndustrySymbol())
	
	#result = list(set(symbols1) & set(symbols2))	
	#result = np.unique(result)
	return symbols




