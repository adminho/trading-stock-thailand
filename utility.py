import os
import numpy as np
import pandas as pd
from matplotlib.finance import date2num

def changeColumnName(df, postfix):
	df.columns = [ '{}_{}'.format(sym, postfix) for sym in df.columns ]
	return df
	
def wrapDf(values):	
	return pd.DataFrame(values)
	
def unWrapDf(df):
	return df.values.T[0]

def normalize_data(df):
	return df/df.ix[0,:]

def fill_missing_values(df_data):
    """Fill missing values in data frame, in place."""    
    df_data.fillna(method="ffill", inplace="True")
    df_data.fillna(method='bfill', inplace="True")
	
def loadStockData(symbols, dates, column_name, base_dir):
	"""Read stock data (adjusted close) for given symbols from CSV files."""
	df = pd.DataFrame(index=dates)	# empty data frame that has indexs as dates
	if 'SET' not in symbols:  # add SET for reference, if absent
		symbols = np.append(['SET'],symbols)

	for symbol in symbols:
		# read CSV file path given ticker symbol.
		csv_file = os.path.join(base_dir, symbol + '.csv'); 
		df_temp = pd.read_csv(csv_file, index_col='<DTYYYYMMDD>',
			parse_dates=True, usecols=['<DTYYYYMMDD>', column_name], na_values=['nan'])
		
		df_temp = df_temp.rename(columns={column_name: symbol})
		df = df.join(df_temp) # left join by default
		
		if symbol == 'SET':  # drop dates SET did not trade (nan values)
			df = df.dropna(subset=["SET"])
	 
	return df

default_path="sec_csv"

def loadPriceData(symbols, dates,  base_dir=default_path):
	return loadStockData(symbols, dates, '<CLOSE>', base_dir)

def loadVolumeData(symbols, dates,  base_dir=default_path):
	return loadStockData(symbols, dates, '<VOL>', base_dir)

def load_OHLCV(symbol, dates, 
					column_names=['<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<VOL>'],  
					base_dir=default_path):
	"""Read stock data (adjusted close) for given symbols from CSV files."""
	df_main = pd.DataFrame(index=dates)	# empty data frame that has indexs as dates
	
	if '<DTYYYYMMDD>' not in column_names:  
		column_names = np.append(['<DTYYYYMMDD>'],column_names)
		
	csv_file = os.path.join(base_dir, "{}.csv".format(symbol)) 
	df_csv = pd.read_csv(csv_file, index_col='<DTYYYYMMDD>',
		parse_dates=True, usecols=column_names, na_values=['nan'])
	
	df_main = df_main.join(df_csv)
	df_main = df_main.dropna(0)	
	
	def convertString(str):
		return str.replace('<','').replace('>','')
		
	columnNames = { value: convertString(value)  for value in df_main.columns }
	df_main.rename(columns=columnNames, inplace=True) # Edit all column names in data frame			
	
	return df_main

def loadStockQuotes(symbol, dates):
	col_names=['<OPEN>', '<CLOSE>', '<HIGH>', '<LOW>']
	df = load_OHLCV(symbol, dates, column_names=col_names,  base_dir=default_path)
	
	#quotes = [ (date, open, close, high, low), .....]
	quotes = df.to_records(convert_datetime64=True).tolist() 
	quotes = [ (date2num(d), o, c, h, l) for d,o,c,h,l in quotes  ]	
	return quotes

from time import gmtime, strftime
def getCurrentDateStr():
    #currentDateStr = strftime("%Y-%m-%d %H:%M:%S", gmtime())    
    return strftime("%Y-%m-%d", gmtime())    
