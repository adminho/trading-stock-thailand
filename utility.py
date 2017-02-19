# Borrowed some code from : 
# https://www.udacity.com/course/machine-learning-for-trading--ud501
# http://matplotlib.org/examples/pylab_examples/finance_demo.
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.dates import DateFormatter, WeekdayLocator,DayLocator, MONDAY
from matplotlib.finance import date2num, candlestick_ohlc

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
	
	#date_list = [ d.strftime("%Y%m%d") for d in dates]	
	for symbol in symbols:
		# read CSV file path given ticker symbol.
		csv_file = os.path.join(base_dir, "{}.csv".format(symbol)) 
		df_temp = pd.read_csv(csv_file, index_col='<DTYYYYMMDD>',
			parse_dates=True, usecols=['<DTYYYYMMDD>', column_name], na_values=['nan'])
		
		df_temp = df_temp.rename(columns={column_name: symbol})
		df = df.join(df_temp) # left join by default
		
		if symbol == 'SET':  # drop dates SET did not trade (nan values)
			df = df.dropna(subset=["SET"])
	 
	return df

default_path="output_csv"

def loadPriceData(symbols, dates,  base_dir=default_path):
	return loadStockData(symbols, dates, '<CLOSE>', base_dir)

def loadVolumeData(symbols, dates,  base_dir=default_path):
	return loadStockData(symbols, dates, '<VOL>', base_dir)

def load_OHLCV(symbol, dates, 
					column_names=['<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<VOL>'],  
					base_dir=default_path):
	"""Read stock data (adjusted close) for given symbols from CSV files."""
	df_main = pd.DataFrame(index=dates)	# empty data frame that has indexs as dates
	
	if '<DTYYYYMMDD>' not in column_names:  # add SET for reference, if absent
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

def loadStockQuotes(symbol, dates, base_dir=default_path):
	col_names=['<OPEN>', '<CLOSE>', '<HIGH>', '<LOW>']
	df = loadOneStockData(symbol, dates, column_names=col_names)
	
	#quotes = [ (date, open, close, high, low), .....]
	quotes = df.to_records(convert_datetime64=True).tolist() 
	quotes = [ (date2num(d), o, c, h, l) for d,o,c,h,l in quotes  ]	
	return quotes
	
def plotStock(df, columns, start_index, end_index, title="Selected data"):
	"""Plot the desired columns over index values in the given range."""	
	df_plot = df.ix[start_index:end_index, columns]
	
	"""Plot stock prices with a custom title and meaningful axis labels."""
	ax = df_plot.plot(title=title, fontsize=12)
	ax.set_xlabel("Date")
	ax.set_ylabel("Price")
	plt.show()

def plotCandlestick(symbol, start_index, end_index, title="Selected data"):
	dates = pd.date_range(start_index, end_index)	
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
	plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')

	plt.show()