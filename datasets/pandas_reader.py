import os
from os import makedirs
from os.path import join, exists
import shutil	

import numpy as np
import pandas as pd
from pandas_datareader import data as pdr
# Requirement 
# pip install fix_yahoo_finance
import fix_yahoo_finance as yf # <== that's all it takes :-)

import matplotlib.pyplot as plt
from matplotlib.finance import date2num
from matplotlib.dates import DateFormatter, WeekdayLocator,DayLocator, MONDAY
from matplotlib.finance import candlestick_ohlc

def load_OHLCV(symbol, startdate, enddate):
	df = pdr.get_data_yahoo(symbol+".BK", start=startdate, end=enddate)	
	columnNames = { value:value.upper()  for value in df.columns}
	df.rename(columns=columnNames, inplace=True)
	return df

# I think doesn't work
def loadStockQuotes(symbol, startdate, enddate):
    #[ (Date, Open, High, Low, Close, Adj Close, Volume), .....]
	df = load_OHLCV(symbol, startdate, enddate)
	
	#quotes = [ (date, open, close, high, low), .....]
	quotes = df.to_records(convert_datetime64=True).tolist()
	quotes = [ (date2num(d), o, h, l, c) for d, o, h, l, c, adj, v in quotes  ]	
	return quotes

def loadBigData(symbol_list, startdate, enddate):	
	if not "SET" in symbol_list:
   		#symbol_list = np.append("SET", symbol_list)
   		pass # found bug on API yahoo finance          
        
	datas = map (lambda symbol: load_OHLCV(symbol, startdate, enddate), symbol_list)	
	return(pd.concat(datas, keys=symbol_list, names=['Symbol', 'Date']))

def selectColumn(all_data, column_name):    
    all_symbol = set(all_data.index.get_level_values(0))    
    df = None
    for symbol in all_symbol:        
        df_col = all_data[[column_name]]        
        df_temp = df_col.iloc[df_col.index.get_level_values('Symbol') == symbol]
        df_temp.index = df_temp.index.droplevel('Symbol')        
        df_temp = df_temp.rename(columns={column_name: symbol}) # prevent column_name overlaping        
        if df is None:
            df = df_temp
        else:
            df = df.join(df_temp, how="outer") # outer join by default
    return df
    
def loadPriceData(symbol_list, startdate, enddate):
    all_data = loadBigData(symbol_list, startdate, enddate)     
    return selectColumn(all_data, 'ADJ CLOSE')

# Borrowed code from : http://matplotlib.org/examples/pylab_examples/finance_demo.
def plotCandlestick(symbol, startdate, enddate, title="Selected data"):	
	quotes = loadStockQuotes(symbol, startdate, enddate)		
    print(quotes)
	mondays = WeekdayLocator(MONDAY)        # major ticks on the mondays
	alldays = DayLocator()              	# minor ticks on the days
	weekFormatter = DateFormatter('%b %d')  # e.g., Jan 12
#	dayFormatter = DateFormatter('%d')      # e.g., 12

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
    
def clearDir(dirPath):
	if exists(dirPath):			
		shutil.rmtree(dirPath)
	
	makedirs(dirPath)	

DIR_CURRENT = os.path.dirname(__file__)
DIR_SEC_CSV_YAHOO = "sec_csv_yahoo"
def createSymbolCSV(df, symbol, outputPath=DIR_SEC_CSV_YAHOO):
    outputPath = join(DIR_CURRENT, outputPath)
    clearDir(outputPath)
    ptt_csv = join(outputPath, symbol + ".csv")
    df.to_csv(ptt_csv)

from time import gmtime, strftime
if __name__ == "__main__":    
    #startdate = datetime.datetime(2010, 10, 1)       
    startDate = '2017-03-01'
    endDate = strftime("%Y-%m-%d", gmtime())    
        
    #load data such as Open, High, Low, Close, Adj Close and Volume
    print("Load data: PTT") 	
    df = load_OHLCV("PTT", startDate, endDate)    
    print(df.tail())
    
    #save as csv file
    print("\nSave as a csv file")
    createSymbolCSV(df, "PTT")
    
    symbols = ["PTT", "AOT", "SCC", "CPALL"]
    print("\nExample of all data of:", symbols)
    all_data = loadBigData(symbols, startDate, endDate)    
    print(all_data)
    
    adj_close = all_data[['ADJ CLOSE']]
    print(adj_close)
    
    # plot graph of candle stick 
    print("\nPlot graph: PTT") 
    plotCandlestick("PTT", startDate, endDate, title ="PTT symbol")
        
    # load Adj Close of many stock 	
    print("\nLoad adjusted closing price of:", symbols) 
    df = loadPriceData(symbols, startDate, endDate)
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    print(df.tail())    
    
    # plot graph all close prices	
    df = df/df.iloc[0,:] # normalized 
    df.plot()
    plt.show()
    