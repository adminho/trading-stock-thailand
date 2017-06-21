from time import gmtime, strftime
import matplotlib.pyplot as plt
import pandas as pd
import dataset_siamchart as ds

# not sure	
def ema(df, periods=12):
	# compute exponential moving average
	#return pd.ewma(df, span = periods)
      return df.ewm(span=periods, adjust=True, min_periods=0, ignore_na=False).mean()

def average_convergence(df, period_low=26, period_fast=12):
    """ compute the MACD (Moving Average Convergence/Divergence) 
	using a fast and slow exponential moving average'    
    """
    emaslow = ema(df, period_low)
    emafast = ema(df, period_fast)
    return (emaslow, emafast, emafast - emaslow)	

def signal_MACD(df_MACD, periods=9):
	return ema(df_MACD, periods)
	
def plot_graph(values):
	style = ['b-', 'r-', 'g-', 'k-', 'y-']
	column_list = values[0].columns
	num_stock = len(column_list)
	date = values[0].index
    	
	if num_stock > 6:
		num_stock = 6  # Fix size
		
	for pos in range(0, num_stock):
		column = column_list[pos] 
		plt.subplot(2, 3, pos+1)	
		
		for i  in range(0, len(values)):
			plt.plot(date, values[i][column], style[i])
			plt.title(column)		
			plt.xticks([])  

	plt.show()

	
if __name__ == "__main__" :	
	startDate = '2017-03-01'
	endDate = strftime("%Y-%m-%d", gmtime())    
	dates = pd.date_range(startDate, endDate)
	symbols = ["PTT", "AOT", "SCC", "CPALL"]
	df = ds.loadPriceData(symbols, dates)
	
	print("+++++EMA 15+++++")
	ema15 = ema(df, 15)	
	print(ema15.tail())
	ema45 = ema(df, 45)
	ema100 = ema(df, 100)
	
	print("+++++MACD+++++")
	_, _, macd = average_convergence(df)	
	print(macd.tail())
	
	print("+++++Signal MACD+++++")
	signal = signal_MACD(macd)
	print(signal.tail())
    
	values = [df, ema15, ema45, ema100]
	plot_graph(values)
	