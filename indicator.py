import pandas as pd
import numpy as np

def roc(df_close, periods=12):	
	# Close - Close n periods ago, change_n_period_ago = df_close - df_close.shift(periods)
	change_n_period_ago = df_close.diff(periods) 
	# Close n periods ago
	close_n_period_ago = df_close.shift(periods)
	
	#ROC = [(Close - Close n periods ago) / (Close n periods ago)] * 100
	ROC = change_n_period_ago / close_n_period_ago * 100
	return ROC
		
def is_strong(df_close, window=12):
	""" In 12 days periods, How many are ROC'security more than ROC' SET (percent %) """
	# df_roc has calculated by daily_returns	
	df_roc = roc(df_close)
		
	# Empty Data Frame
	df_main = pd.DataFrame(index=df_close.index, columns=df_close.columns) 
	
	for symbol in df_close.columns:
		if symbol == 'SET':
			continue
		
		df_compare 	= df_roc['SET'] < df_roc[symbol]
		# In python True is 1, False is 0
		# 1: meaning 12 days periods, ROC'security > ROC'SET always
		# 2: meaning 12 days periods, ROC'security < ROC'SET always
		df_main[symbol] = pd.rolling_mean(df_compare[window:], window)
			
	df_main['SET']=0
	print(df_main)	

def get_rolling_mean(values, window):
    """Return rolling mean of given values, using specified window size."""
    return pd.rolling_mean(values, window=window)

def get_rolling_std(values, window):
    """Return rolling standard deviation of given values, using specified window size."""
    # TODO: Compute and return rolling standard deviation
    return pd.rolling_std(values, window=window)

def get_bollinger_bands(rm, rstd):
    """Return upper and lower Bollinger Bands."""
    # TODO: Compute upper_band and lower_band
    upper_band = rm + rstd * 2
    lower_band = rm - rstd * 2
    return upper_band, lower_band	

def daily_returns(df):
    """Compute and return the daily return values."""
	# (current_price / previous_price) -1
	# daily_returns = (df / df.shift(1)) - 1
    daily_returns = df.pct_change()
    daily_returns.ix[0, :] = 0
	    
    return daily_returns
	
def	BBANDS(df_price, periods=20):	
	# Middle Band = 20-day simple moving average (SMA)
	df_middle_band = pd.rolling_mean(df_price, window=periods)
	
	# 20-day standard deviation of price
	""" Pandas uses the unbiased estimator (N-1 in the denominator), 
	whereas Numpy by default does not.
	To make them behave the same, pass ddof=1 to numpy.std()."""	
	df_std = pd.rolling_std(df_price, window=periods)
	
	# Upper Band = 20-day SMA + (20-day standard deviation of price x 2)
	df_upper_band = df_middle_band + (df_std * 2)
	
	# Lower Band = 20-day SMA - (20-day standard deviation of price x 2)
	df_lower_band = df_middle_band - (df_std * 2)	
	
	return (df_upper_band, df_middle_band, df_lower_band)

def	get_BBANDS(df, symbol, periods=20):	
	(upper, middle, lower) = BBANDS(df[symbol], periods)		
	df_BBANDS = pd.concat([upper, middle, lower], axis=1, join='inner')
	df_BBANDS.columns = ['UPPER', 'MIDDLE', 'LOWER']
	return df_BBANDS

def diff_BBANDS(df_price, periods=20):		
	(upper, middle, lower) = BBANDS(df_price, periods)	
	return (df_price - upper
			, df_price - middle
			, df_price - lower)
			
def sma(df, periods=12):
	# compute simple moving average
	return pd.rolling_mean(df, window=periods)	

# not sure	
def ema(df, periods=12):
	# compute exponential moving average
	return pd.ewma(df, span = periods)

def average_convergence(df, period_low=26, period_fast=12):
    """
    compute the MACD (Moving Average Convergence/Divergence) 
	using a fast and slow exponential moving average'    
    """
    emaslow = ema(df, period_low)
    emafast = ema(df, period_fast)
    return (emaslow, emafast, emafast - emaslow)	
	
def rsi(df):				
	periods=14
	# Price change,	df_change = df - df.shift(1)
	df_change = df.diff(1)			
		
	df_gain = df_change.where(df_change > 0) 		# Gain
	df_loss = -1 * df_change.where(df_change < 0) 	# loss, multiple -1 to positive values
	df_gain.fillna(0, inplace=True) 				# fill NaN to 0
	df_loss.fillna(0, inplace=True)					# fill NaN to 0
	
	# create DataFrame for saving Average Gain and Average Loss
	df_avg_gain = pd.DataFrame(columns = df.columns, index = df.index)
	df_avg_loss = df_avg_gain.copy()
		
	df_avg_gain.ix[periods] = df_gain[1:periods+1].mean()	# First Average Gain = Sum of Gains over the past 14 periods / 14.
	df_avg_loss.ix[periods] = df_loss[1:periods+1].mean()	# First Average Loss = Sum of Losses over the past 14 periods / 14
	
	for index in range(periods+1, len(df)):
		#Average Gain = [(previous Average Gain) x 13 + current Gain] / 14.	
		df_avg_gain.ix[index] = (df_avg_gain.ix[index-1] * 13 + df_gain.ix[index])/periods
		
		#Average Loss = [(previous Average Loss) x 13 + current Loss] / 14.
		df_avg_loss.ix[index] =	(df_avg_loss.ix[index-1] * 13 + df_loss.ix[index])/periods

	# RS = Average Gain / Average Loss		
	RS = df_avg_gain/df_avg_loss	
	
	#              100
    # RSI = 100 - --------
    #             1 + RS
	RSI = 100 - 100/(RS+1)
	return RSI
	
def sharpe_ratio(rp, rf=float('nan')):	
	if np.isnan(rf):	
		rf = rp.copy()
		rf.ix[0:] = 0
		
	# rp = Expected porfolio return
	# rf = Risk free rate
	ret = rp - rf
	# Sharpe ratio = mean(Expected porfolio return - Risk free rate)/Portfolio standard deviation
	return ret.mean()/ret.std()
	
def rolling_sharpe_ratio(rp, rf, window):
	# Example
	# rp = df_daily_return[symbol]
	# rf = df_daily_return['SET']	
	result = rp - rf	
	mean = get_rolling_mean(result, window)	
	std = get_rolling_std(result, window )
	# Sharpe ratio = mean(Expected porfolio return - Risk free rate)/Portfolio standard deviation
	return mean/std

def create_dataframe_SR(df, symbols, window=5):
	df_sr = pd.DataFrame(columns = symbols, index= df.index)
	df_daily_return = daily_returns(df)
	
	# Use SET for return reference, not use risk-free rate realy as return reference
	rf = df_daily_return['SET']		
	for sym in symbols :				
		df_sr[sym] = rolling_sharpe_ratio(df_daily_return[sym], rf, window)
		
	return df_sr

def isStrongSR(df_sr, window=10):
	df_compare = df_sr > 0					    # Sharpe ratio is more than 0, dialy return is pluss 
	df_sum = pd.rolling_sum(df_compare, window) # Sharpe ratio is more than 0 contiue [window] days
	return 1* (df_sum==window) 					# True When Sharpe ratio > 10 continually (by default), multiply 1 to convert integer number

def true_range(df):
	high = df['<HIGH>']
	low = df['<LOW>']
	previous_close = df['<CLOSE>'].shift(1)
	
	# True Range (TR)
	# Method 1: Current High less the current Low
	# Method 2: Current High less the previous Close (absolute value)
	# Method 3: Current Low less the previous Close (absolute value)
	method1 = high-low	
	method2 = (high - previous_close).abs()
	method3 = (low - previous_close).abs()
	
	df_TR = pd.concat([method1, method2, method3], axis=1, join='inner')
	#df_TR = pd.DataFrame(df_TR.max(axis=1), columns=['<TR>'])	
	return  df_TR.max(axis='columns')

def ATR(df):
	periods = 14
	df_TR = true_range(df)	# True range
	
	df_ATR = pd.DataFrame(columns = ['<ATR>'], index = df.index)
	df_ATR.ix[periods-1] = df_TR[0:periods].mean()	# First ATR = Sum of TR over the past 14 periods / 14
		 
	for index in range(periods, len(df)):
		#Current ATR = [(Prior ATR x 13) + Current TR] / 14
		# - Multiply the previous 14-day ATR by 13.
		# - Add the most recent day's TR value.
		# - Divide the total by 14  		
		df_ATR.ix[index] = (df_ATR.ix[index-1] * 13 + df_TR.ix[index])/periods
			
	return df_ATR

def getBeta(df, stock_name, benchmark_name):
	# Compute returns of stock	
	rs = roc(df[stock_name], periods=1)/100
	rb = roc(df[benchmark_name], periods=1)/100
	
	# Beta = Covariance(rs, rb)/Variance(rb)
	# where rs is the return on the stock and rb is the return on a benchmark index.
	return rs.cov(rb)/rb.var()	
	
def OBV(df_volume, df_close):			
	# create empty Data Frame
	df_OBV = pd.DataFrame(index = df_volume.index, columns = df_volume.columns)
	# first OBV
	df_OBV.ix[0] = df_volume.ix[0] 
		
	# Price change,	df_price_change = df_close - df_close.shift(1)
	df_price_change = df_close.diff(1)	
	
	for symbol in df_volume.columns:				
		for index in range(1, len(df_volume)):
			#If the closing price is above the prior close price then: 
			#Current OBV = Previous OBV + Current Volume
	
			#If the closing price is below the prior close price then: 
			#Current OBV = Previous OBV  -  Current Volume
		
			#If the closing prices equals the prior close price then:
			#Current OBV = Previous OBV (no change)		
		
			change = df_price_change.ix[index][symbol]
			current_volume =  df_volume.ix[index][symbol]
			
			if change > 0:			
				current_volume =  current_volume
			
			elif change < 0:
				current_volume =  -1 * current_volume
				
			else:	
				current_volume = 0
			
			df_OBV.ix[index][symbol] = df_OBV.ix[index -1][symbol]  + current_volume
	
	return df_OBV		

def change_volume(df_volume):
	return df_volume.diff(1)
	
def get_change_price(df_open, df_close):		
	current_open = df_open
	current_close = df_close	
	previous_open = df_open.shift(1)
	previous_close = df_close.shift(1)	
	
	method1 = current_close - current_open	
	method2 = current_open - previous_open 
	method3 = current_open - previous_close
	method4 = current_close - previous_open 
	method5 = current_close - previous_close
		
	df_result = pd.concat([method1, method2, method3, method4, method5]
					, axis=1
					, join='inner')
					
	df_result.columns = ['<CLOSE_OPEN>'
						, '<OPEN_POPEN>'
						, '<OPEN_PCLOSE>'
						, '<CLOSE_POPEN>'
						, '<CLOSE_PCLOSE>']				
	return  df_result
	
def gain(df):
	gain = df.diff(axis='columns')
	return gain.sum() / df['<BUY>'].sum() * 100

import matplotlib.pyplot as plt	
def compare_stock(df, x_stock, y_stock):	
	df_dialy_returns = daily_returns(df)	
	df_dialy_returns.plot(kind='scatter', x = x_stock, y = y_stock)
	
	X = df_dialy_returns[x_stock] 
	Y = df_dialy_returns[y_stock]
	
	beta, alpha = np.polyfit(X, Y, 1)
	print('Beta is {}, Alpha is {}'.format(beta, alpha))
	print('fx = {}x + {}'.format(beta, alpha))
	
	fx = beta*X + alpha	
	plt.plot(X, fx, 'r-')
	plt.show()
	
def isNewHigh(df ,periods =14):
	finish = len(df) - periods + 1
	resultDf = pd.DataFrame(columns=df.columns) # empty
	
	for index in range(0, finish):
		windowDf = df.ix[index: index+periods] # slice n periods
		current = df.ix[index]
		compare = windowDf > current
		
		# new high from current price is True
		# but side way or new low is False
		# count boolean
		result = pd.rolling_sum(compare, window=periods)	
		result = result.shift(-(periods-1))
		result.dropna(0, inplace=True)
		result = result > periods-5 # number new high compare with current price
		resultDf = resultDf.append(result)
			
	return resultDf
			
	

