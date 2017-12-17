import os
from os import listdir, makedirs
from os.path import isfile, join, exists
import numpy as np
import pandas as pd
import shutil	

import matplotlib.pyplot as plt
from matplotlib.finance import date2num
from matplotlib.dates import DateFormatter, WeekdayLocator,DayLocator, MONDAY
from matplotlib.finance import candlestick_ohlc

from tqdm import tqdm

# Requirement
# 1) install Anaconda: https://www.continuum.io/downloads
# 2) pip install tqdm
# 3) download EOD file: http://siamchart.com/stock/	 (Must register to login)

def getFileNameInDir(path):
	onlyFiles = [ join(path,f) for f in listdir(path) if isfile(join(path, f)) ]
	return onlyFiles
	
def getHeaderFile(eodFiles):
	for f in eodFiles:
		df = pd.read_csv(f)
		return df.columns.values			# read a header file from first file
		
def getStockData(eodFiles, selectedSmbol = []):			
	dict = {}	# empty dictionary
		
	totalFiles = len(eodFiles)
	for i in tqdm(range(totalFiles), ascii=True, desc='Reading EOD files'):	
		f = eodFiles[i]
		df = pd.read_csv(f)
		total_row = len(df.index)	
		all_data = df.values;
				
		#range(start, stop, step)
		for row in range(total_row-1, -1, -1):		# reverse form range(0, total_row)
			symbol = all_data[row][0]				# name of stock in the first column
			if len(selectedSmbol)!=0 :
				if not symbol in selectedSmbol: continue			
			
			if symbol == "COM7": symbol = "COM7_" 	# fix bug for this symbol only
			
			current_row = all_data[row]				
			if symbol in dict: 						# There are many symbol data in dictionary
				dict[symbol].append(current_row)	# append new data to old data
			else: # no symbol data in dictionary
				dict[symbol] = [current_row]				
		
	return dict

def clearDir(dirPath):
	if exists(dirPath):			
		shutil.rmtree(dirPath)	
	makedirs(dirPath)	

def changeName(name):
	"""
	I change this column name ["<OPEN>", "<HIGH>", "<LOW>", "<CLOSE>", "<VOL>"] 	
	"""
	if name in ["<OPEN>", "<HIGH>", "<LOW>", "<CLOSE>"]:
		# Frist charector is upper case
		name = name.replace('<', '').replace('>', '')
		#name = name[0] + name[1:].lower()		
	elif name in ["<VOL>"]:
		#name = name.replace("<VOL>", "Volume")
		name = name.replace("<VOL>", "VOLUME")
	elif name in ["<DTYYYYMMDD>"]:
		#name = name.replace("<DTYYYYMMDD>", "Date")
		name = name.replace("<DTYYYYMMDD>", "DATE")
	return name

DIR_CURRENT = os.path.dirname(__file__)
DIR_SEC_CSV = "sec_csv"

# download: http://siamchart.com/stock/	 (Must register to login)
EOD_file = "D:/MyProject/Big-datasets/data_stock/set-archive_EOD_UPDATE"
def createSymbolCSV(start_idex, outputPath=DIR_SEC_CSV):
	eodFiles = getFileNameInDir(EOD_file)	
	eodFiles = eodFiles[-1 * start_idex:]		# select files at latest N days
	
	outputPath = join(DIR_CURRENT,outputPath)
	clearDir(outputPath) # delete old files
	
	eodFiles = [ join(DIR_CURRENT,file)  for file in eodFiles]
	dataStock  = getStockData(eodFiles)	
	headers = getHeaderFile(eodFiles)			# Read a header in CSV files		
	columnNames = { index:changeName(value)  for index, value in enumerate(headers)}
	
	# write all data to csv files and separate file name be followed by symbol names of securities		
	itemList = list(dataStock.items())
	allItem = len(itemList)
	assert allItem == len(dataStock.items())	
	for i in tqdm(range(allItem), ascii=True, desc='Writing CSV files'):
		key, allRow = itemList[i]
		df = pd.DataFrame(allRow)
		df.rename(columns=columnNames, inplace=True) # change column names in data frame: convert from number to symbol names				
		df.drop('<TICKER>', axis=1, inplace=True) # remove column
		fileName = "{}.csv".format(join(outputPath, key))		
		df.to_csv(fileName, index = False) 		# write data into CSV file (without index)				
		
def load_OHLCV(symbol, dates=None, 
					#column_names=['Open', 'High', 'Low', 'Close', 'Volume'],  
					column_names=['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME'],  
					base_dir=DIR_SEC_CSV):
	
	#if 'Date' not in column_names:  
	#	column_names = np.append(['Date'],column_names)
	if 'DATE' not in column_names:  
		column_names = np.append(['DATE'],column_names)
		
	base_dir = join(DIR_CURRENT,base_dir)
	csv_file = os.path.join(base_dir, "{}.csv".format(symbol)) 
	# df_csv = pd.read_csv(csv_file, index_col='Date',
	df_csv = pd.read_csv(csv_file, index_col='DATE',
		parse_dates=True, usecols=column_names, na_values=['nan'])
	
	"""Read securities data for given symbols from CSV files."""
	if dates is None:
		dates = df_csv.index
	df_main = pd.DataFrame(index=dates)	# empty data frame that has indexs as dates
	
	df_main = df_main.join(df_csv)
	df_main = df_main.dropna(0)	
	return df_main

def loadStockQuotes(symbol, dates):
	#col_names=['Open', 'Close', 'High', 'Low']	
	col_names = ['OPEN', 'HIGH', 'LOW', 'CLOSE']
	df = load_OHLCV(symbol, dates, column_names=col_names, base_dir=DIR_SEC_CSV)
	
	#quotes = [ (date, open, close, high, low), .....]
	quotes = df.to_records(convert_datetime64=True).tolist() 
	quotes = [ (date2num(d), o, c, h, l) for d,o,c,h,l in quotes  ]	
	return quotes

def loadManySymbols(symbols, dates, column_name, base_dir):
	"""Read securities data for given symbols from CSV files."""
	df = pd.DataFrame(index=dates)	# empty data frame that has indexs as dates
	if 'SET' not in symbols:  # add SET for reference, if absent
		symbols = np.append(['SET'],symbols)
        
	base_dir = join(DIR_CURRENT,base_dir)
	for symbol in symbols:
		# read CSV file path given symbol.
		csv_file = os.path.join(base_dir, symbol + '.csv'); 
		#df_temp = pd.read_csv(csv_file, index_col='Date',		
			#parse_dates=True, usecols=['Date', column_name], na_values=['nan'])
		df_temp = pd.read_csv(csv_file, index_col='DATE',
			parse_dates=True, usecols=['DATE', column_name], na_values=['nan'])
		
		df_temp = df_temp.rename(columns={column_name: symbol})
		df = df.join(df_temp) # left join by default
		
		if symbol == 'SET':  # drop dates SET did not trade (nan values)
			df = df.dropna(subset=["SET"])
	 
	return df

def loadPriceData(symbol_list, dates,  base_dir=DIR_SEC_CSV):
	#return loadManySymbols(symbol_list, dates, 'Close', base_dir)
	return loadManySymbols(symbol_list, dates, 'CLOSE', base_dir)
	
# Borrowed code from : http://matplotlib.org/examples/pylab_examples/finance_demo.
def plotCandlestick(symbol, dates, title="Selected data"):	
	quotes = loadStockQuotes(symbol, dates)		
	
	mondays = WeekdayLocator(MONDAY)        # major ticks on the mondays
	alldays = DayLocator()              	# minor ticks on the days
	weekFormatter = DateFormatter('%b %d')  # e.g., Jan 12
	dayFormatter = DateFormatter('%d')      # e.g., 12

	fig, ax = plt.subplots()
	fig.subplots_adjust(bottom=0.2)
	ax.xaxis.set_major_locator(mondays)
	ax.xaxis.set_minor_locator(alldays)
	ax.xaxis.set_major_formatter(weekFormatter)
	#ax.xaxis.set_minor_formatter(dayFormatter)
 
	#plot_day_summary(ax, quotes, ticksize=3)
	candlestick_ohlc(ax, quotes, width=0.6)

	ax.xaxis_date()
	ax.autoscale_view()
	ax.set_title(title)
	plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')

	plt.show()

from time import gmtime, strftime
if __name__ == "__main__" :	
	#Create CSV files for securities
	print()
	print("----------------------------------------------")
	print("-------------Creating csv files---------------")
	print("----------------------------------------------")
	createSymbolCSV(2000)
    
	#currentDateStr = strftime("%Y-%m-%d %H:%M:%S", gmtime())    	
	startDate = '2017-03-01'
	endDate = strftime("%Y-%m-%d", gmtime())    
	dates = pd.date_range(startDate, endDate)
   
   #load data such as Open, High, Low, Close and Volume 
	print("Load data: PTT") 
	df = load_OHLCV("PTT", dates)	
	print(df.tail())
   
   # plot graph of candle stick 
	print("\nPlot graph: PTT") 
	plotCandlestick("PTT", dates, title ="PTT symbol")
		
   # load close prices of many stock 
	symbols = ["PTT", "AOT", "SCC", "CPALL"]
	print("\nLoad close prices of:", symbols) 
	df = loadPriceData(symbols, dates)
	print(df.tail())
	
   # plot graph all close prices	
	df = df/df.iloc[0,:] # normalized 
	df.plot()
	plt.show()