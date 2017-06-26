from os import listdir, makedirs, remove
from os.path import isfile, join, exists
import numpy as np
import pandas as pd
import shutil	
import utility as util
import indicator as ind
from sklearn import preprocessing

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
	eodFiles = getFileNameInDir(EOD_file)	
	eodFiles = eodFiles[-1 * start_idex:]		# select files latest N days
	
	clearDir(outputPath) # delete old files
	
	dataStock  = getStockData(eodFiles)	
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

def getTrainData_1(symbol, startDate, endDate, periods=14, remove_head=19):
	dates = pd.date_range(startDate, endDate)	
	df = util.loadPriceData([symbol], dates)		
	util.fill_missing_values(df)
	
	# skip periods day latest (14 days) that predict a price is new high or not
	df_sliced = df.ix[0: len(df) - periods] 
	price_close = pd.DataFrame(df_sliced[symbol])
	set = pd.DataFrame(df_sliced['SET'])
	
	rsi = ind.rsi(price_close)/100 # normalize
	sr = ind.create_dataframe_SR(df_sliced, [symbol])
	myRatio = ind.get_myRatio(price_close)
	daily = ind.daily_returns(price_close)*100 # normalize
	_, _, macd = ind.average_convergence(price_close)
		
	ohlcv = util.load_OHLCV(symbol, dates)
	percent_KD = ind.percent_KD(ohlcv)/100  # normalize 		
	c2o =	ind.daily_returns_2(ohlcv)*100 # normalize
	
	volume = util.loadVolumeData([symbol], dates)	
	#skip periods day latest
	volume_sliced = volume.ix[0: len(volume) - periods]
	assert len(volume_sliced) == len(df_sliced)
	obv = ind.OBV(volume_sliced, df_sliced)
	obv_rsi = ind.rsi(obv)/100 	# calcuate momentum with rsi and normalize
	set_rsi = ind.rsi(set)/100 	# calcuate momentum with rsi and normalize
	
	# Join data frame
	# rename column	
	rsi.rename(columns={symbol:'RSI'},inplace=True)
	#sr.rename(columns={symbol:'SR'},inplace=True)		
	myRatio.rename(columns={symbol:'MY'},inplace=True)	
	daily.rename(columns={symbol:'DAILY'},inplace=True)	
	macd.rename(columns={symbol:'MACD'},inplace=True)
	obv_rsi.rename(columns={symbol:'OBV_RSI'},inplace=True)
	set_rsi.rename(columns={'SET':'SET_RSI'},inplace=True)
	
	Xtrain = pd.DataFrame(index=df_sliced.index)	
	Xtrain = Xtrain.join(rsi['RSI'])
	Xtrain = Xtrain.join(percent_KD['%K'])
	#Xtrain = Xtrain.join(sr['SR'])		
	Xtrain = Xtrain.join(myRatio['MY'])
	Xtrain = Xtrain.join(daily['DAILY'])
	Xtrain = Xtrain.join(macd['MACD'])	
	Xtrain = Xtrain.join(c2o)	
	Xtrain = Xtrain.join(obv_rsi['OBV_RSI'])
	Xtrain = Xtrain.join(set_rsi)
	
	upTrend = ind.isUpTrend(df, symbol, periods=periods)
	Ydigit = 1 * upTrend[symbol]  # multiply 1 : True is converted to 1 (up trend) , False becomes 0
	
	# skip at head row, avoid NaN values
	Xtrain = Xtrain.ix[remove_head:]
	Ydigit = Ydigit.ix[remove_head:]
	price_close = price_close.ix[remove_head:]
	return Xtrain, Ydigit, price_close

def packSeqData(X, Y, sequence_length, size_test=5):
	Xpacked = []
	height, width = X.shape
	
	for index in range(0, height - sequence_length + 1):
		Xsliced = X[index: index + sequence_length]
		Xpacked.append(Xsliced.values)

	#examples, time series or length of input (n days), dim. of each value or number of features (each technical indicator)	
	Xpacked = np.array(Xpacked)
	assert np.shape(Xpacked) == Xpacked.shape
		
	Xtrain, Xtest = Xpacked[:-size_test], Xpacked[-size_test:]
	assert Xtest.shape[0] == size_test
	assert Xtrain.shape[0] + size_test == Xpacked.shape[0]
	
	Ypacked = Y[sequence_length-1:]
	Ytrain, Ytest = Ypacked[:-size_test], Ypacked[-size_test:]
	assert Ytest.shape[0] == size_test
	assert Ytrain.shape[0] + size_test == Ypacked.shape[0]
		
	return Xtrain, Xtest, Ytrain, Ytest

def getTrainData_2(symbol, startDate, endDate, remove_head=15):
	dates = pd.date_range(startDate, endDate)	
	df = util.loadPriceData([symbol], dates)				#1	
	close = df.ix[0:len(df)-1] 	# skip tail
	assert len(df) == len(close) + 1
	
	ema5 = ind.ema(close, periods=5)						#2
	sma5 = ind.ema(close, periods=5)						#3	
	ema15 = ind.ema(close, periods=5)						#4
	sma15 = ind.ema(close, periods=5)						#5
		
	bb1p = ind.get_BBANDS(close, symbol, periods=14, mul=1) #6, #7
	bb2p = ind.get_BBANDS(close, symbol, periods=14, mul=2)	#8, #9
	_, _, macd = ind.average_convergence(close)		#10
	signal_macd = ind.signal_MACD(macd)						#11
	
	rsi = ind.rsi(close)									#12
	
	ohlcv = util.load_OHLCV(symbol, dates)
	percent_KD = ind.percent_KD(ohlcv) 						#14, 15
	
	# rename columns
	close.rename(columns={symbol:'CLOSE'},inplace=True)	
	ema5.rename(columns={symbol:'EMA5'},inplace=True)
	sma5.rename(columns={symbol:'SMA5'},inplace=True)
	ema15.rename(columns={symbol:'EMA15'},inplace=True)
	sma15.rename(columns={symbol:'SMA15'},inplace=True)	
	bb1p.rename(columns={'LOWER':'LOWER_BB1P'},inplace=True)
	bb1p.rename(columns={'UPPER':'UPPER_BB1P'},inplace=True)
	bb2p.rename(columns={'LOWER':'LOWER_BB2P'},inplace=True)
	bb2p.rename(columns={'UPPER':'UPPER_BB2P'},inplace=True)		
	macd.rename(columns={symbol:'MACD'},inplace=True)
	signal_macd.rename(columns={symbol:'SG_MACD'},inplace=True)
	rsi.rename(columns={symbol:'RSI'},inplace=True)
	
	Xtrain = pd.DataFrame(index=close.index)	
	Xtrain = Xtrain.join(close['CLOSE'])		# 1
	Xtrain = Xtrain.join(ema5['EMA5'])			# 2
	Xtrain = Xtrain.join(sma5['SMA5'])			# 3
	Xtrain = Xtrain.join(ema15['EMA15'])		# 4
	Xtrain = Xtrain.join(sma15['SMA15'])		# 5
	Xtrain = Xtrain.join(bb1p['LOWER_BB1P'])	# 6
	Xtrain = Xtrain.join(bb1p['UPPER_BB1P'])	# 7
	Xtrain = Xtrain.join(bb2p['LOWER_BB2P'])	# 8
	Xtrain = Xtrain.join(bb2p['UPPER_BB2P'])	# 9
	Xtrain = Xtrain.join(macd['MACD'])			# 10
	Xtrain = Xtrain.join(signal_macd['SG_MACD']) # 11
	Xtrain = Xtrain.join(rsi['RSI'])			# 12
	Xtrain = Xtrain.join(percent_KD)			# 13
	
	Ytrain = df[symbol].shift(-1) # skip SET and shift 
	Ytrain.dropna(0, inplace=True) 
	assert len(Ytrain) == len(Xtrain)
	
	# skip at head row, avoid NaN values
	Xtrain = Xtrain[remove_head:]
	Ytrain = Ytrain[remove_head:]		
	
	return Xtrain,  Ytrain
	
def getTrainData_3(symbol, startDate, endDate, remove_head=15):
	dates = pd.date_range(startDate, endDate)	
	
	df = util.loadPriceData([symbol], dates)				#1	
	close = df.ix[0:len(df)-1] 	# skip tail
	assert len(df) == len(close) + 1
	
	ema5 = ind.ema(close, periods=5)						#2
	sma5 = ind.ema(close, periods=5)						#3	
	ema15 = ind.ema(close, periods=5)						#4
	sma15 = ind.ema(close, periods=5)						#5
		
	bb1p = ind.get_BBANDS(close, symbol, periods=14, mul=1) #6, #7
	bb2p = ind.get_BBANDS(close, symbol, periods=14, mul=2)	#8, #9
	_, _, macd = ind.average_convergence(close)		#10
	signal_macd = ind.signal_MACD(macd)						#11
	
	rsi = ind.rsi(close)									#12
	
	ohlcv = util.load_OHLCV(symbol, dates)
	percent_KD = ind.percent_KD(ohlcv) 						#14, 15
	
	# rename columns
	close.rename(columns={symbol:'CLOSE'},inplace=True)	
	ema5.rename(columns={symbol:'EMA5'},inplace=True)
	sma5.rename(columns={symbol:'SMA5'},inplace=True)
	#ema15.rename(columns={symbol:'EMA15'},inplace=True)
	#sma15.rename(columns={symbol:'SMA15'},inplace=True)	
	bb1p.rename(columns={'LOWER':'LOWER_BB1P'},inplace=True)
	bb1p.rename(columns={'UPPER':'UPPER_BB1P'},inplace=True)
	#bb2p.rename(columns={'LOWER':'LOWER_BB2P'},inplace=True)
	#bb2p.rename(columns={'UPPER':'UPPER_BB2P'},inplace=True)		
	macd.rename(columns={symbol:'MACD'},inplace=True)
	#signal_macd.rename(columns={symbol:'SG_MACD'},inplace=True)
	rsi.rename(columns={symbol:'RSI'},inplace=True)
	
	Xtrain = pd.DataFrame(index=close.index)	
	Xtrain = Xtrain.join(close['CLOSE'])		# 1
	Xtrain = Xtrain.join(ema5['EMA5'])			# 2
	Xtrain = Xtrain.join(sma5['SMA5'])			# 3
	#Xtrain = Xtrain.join(ema15['EMA15'])		# 4
	#Xtrain = Xtrain.join(sma15['SMA15'])		# 5
	Xtrain = Xtrain.join(bb1p['LOWER_BB1P'])	# 6
	Xtrain = Xtrain.join(bb1p['UPPER_BB1P'])	# 7
	#Xtrain = Xtrain.join(bb2p['LOWER_BB2P'])	# 8
	#Xtrain = Xtrain.join(bb2p['UPPER_BB2P'])	# 9
	Xtrain = Xtrain.join(macd['MACD'])			# 10
	#Xtrain = Xtrain.join(signal_macd['SG_MACD']) # 11
	Xtrain = Xtrain.join(rsi['RSI'])			# 12
	Xtrain = Xtrain.join(percent_KD)			# 13
	
	df = df[symbol] # skip SET
	daily = ind.daily_returns(df)	
	daily = daily.shift(-1) 		# predict tommorow 
	daily.dropna(0, inplace=True) 	# drop NaN in last row
	Ytrain = 1 * (daily > 0.0) 		# if positive it converted 1, or 0 in negative
	assert len(Ytrain) == len(Xtrain)
	
	# skip at head row, avoid NaN values
	Xtrain = Xtrain[remove_head:]
	Ytrain = Ytrain[remove_head:]		
	
	# normalize	
	normalizer = preprocessing.Normalizer().fit(Xtrain)
	Xnorm = normalizer.transform(Xtrain) 
	Xnorm = pd.DataFrame(Xnorm, columns=Xtrain.columns)
		
	return Xnorm,  Ytrain

def getTrainData_4(symbol, startDate, endDate, periods=14, remove_head=19):
	dates = pd.date_range(startDate, endDate)	
	
	# day periods that predict a price is new high or not
	df = util.loadPriceData([symbol], dates)		
	
	# skip periods day latest
	df_sliced = df.ix[0: len(df) - periods]
		
	price_close = pd.DataFrame(df_sliced[symbol])
	set = pd.DataFrame(df_sliced['SET'])
	
	bbands = ind.get_BBANDS(price_close, symbol) 
	ema26, ema12, macd = ind.average_convergence(price_close)
	rsi = ind.rsi(price_close)
	daily = ind.daily_returns(price_close)
		
	ohlcv = util.load_OHLCV(symbol, dates)
	percent_KD = ind.percent_KD(ohlcv)
		
	volume = util.loadVolumeData(symbol, dates)
	#skip periods day latest
	volume_sliced = volume.ix[0: len(volume) - periods] 
	assert len(volume_sliced) == len(price_close)
	
	volume_sliced = pd.DataFrame(volume_sliced[symbol])
	obv = ind.OBV(volume_sliced, price_close)
		
	# Join data frame
	# rename column	
	price_close.rename(columns={symbol:'CLOSE'},inplace=True)
	ema26.rename(columns={symbol:'EMA26'},inplace=True)
	ema12.rename(columns={symbol:'EMA12'},inplace=True)
	daily.rename(columns={symbol:'DAILY'},inplace=True)
	rsi.rename(columns={symbol:'RSI'},inplace=True)	
	obv.rename(columns={symbol:'OBV'},inplace=True)
		
	Xtrain = price_close			
	Xtrain = Xtrain.join(bbands['UPPER']) 
	Xtrain = Xtrain.join(bbands['LOWER']) 
	Xtrain = Xtrain.join(ema26) 
	Xtrain = Xtrain.join(ema12)
	Xtrain = Xtrain.join(rsi)
	Xtrain = Xtrain.join(percent_KD['%K'])
	Xtrain = Xtrain.join(obv)
	Xtrain = Xtrain.join(set)
			
	upTrend = ind.isUpTrend(df, symbol, periods=periods)
	Ydigit = 1 * upTrend[symbol]  # multiply 1 : True is converted to 1 (up trend) , False becomes 0
	assert len(Xtrain) == len(Ydigit)
	
	# skip at head row, avoid NaN values
	Xtrain = Xtrain.ix[remove_head:]
	Ydigit = Ydigit.ix[remove_head:]	
	Xtrain.fillna(0, inplace=True)	 # protected NaN value	
	return Xtrain, Ydigit
	
DIR_LIST_CSV = "list_securities"
def _getStockSymbol(csvFilename):	    
	df = pd.read_csv(join(DIR_LIST_CSV, csvFilename), encoding = "TIS-620")
	symbols = np.append(['SET'], df.values.T[0])
	
	symbols = [ sym.replace('COM7','COM7_') for sym in symbols ] # fix bug only
	return symbols

def getSETHDSymbol():	
	return _getStockSymbol("list_SETHD.csv")	
	
def getSET100Symbol():	
	return _getStockSymbol("list_SET100.csv")

def getSET50Symbol():		
	return _getStockSymbol("list_SET50.csv")

def getSubIndustrySymbol():		
	return _getStockSymbol("list_Index_Sub_Instrudry.csv")

def getAllSymbol(path=DIR_SEC_CSV):
	symbols1 = [ f.replace(".csv","") for f in listdir(path) if isfile(join(path, f)) ]	
	symbols2 = ['SET']
	symbols2 = np.append(symbols2, _getStockSymbol('listedCompanies_th_TH.csv'))
	#symbols2 = np.append(symbols2, getSET100Symbol())
	#symbols2 = np.append(symbols2, getSETHDSymbol())
	

	#symbols2 = np.append(symbols2, getSubIndustrySymbol())
	
	result = list(set(symbols1) & set(symbols2))	
	result = np.unique(result)
	return result




