# All formula : reference at http://stockcharts.com/school/doku.php?id=chart_school/
import pandas as pd
import numpy as np

def roc(df_close, periods=12):	
	# Close - Close n periods ago
	# change_n_period_ago = df_close - df_close.shift(periods)
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
    #return pd.rolling_mean(values, window=window)
    return values.rolling(center=False, window=window).mean()

def get_rolling_std(values, window):
    """Return rolling standard deviation of given values, using specified window size."""
    # Compute and return rolling standard deviation
    #return pd.rolling_std(values, window=window)
    return values.rolling(center=False,window=window).std()
	
def get_bollinger_bands(rm, rstd):
    """Return upper and lower Bollinger Bands."""
    # Compute upper_band and lower_band
    upper_band = rm + rstd * 2
    lower_band = rm - rstd * 2
    return upper_band, lower_band	

def daily_returns(df):
	"""Compute and return the daily return values."""
	# (current_price / previous_price) -1
	# daily_returns = (df / df.shift(1)) - 1
	daily_returns = df.pct_change(); 
	daily_returns.iloc[0] = 0
		    
	return daily_returns

def close_2_open(df):
	"""Compute and return the daily return values."""
	# from pervious close to current open close
	# (current_open_price / previous_close_price) -1	
	current_open = df['OPEN']
	previos_close = df['CLOSE'].shift(1)
	daily_returns_2 = (current_open/ previos_close) - 1
	#daily_returns = df.pct_change(); 
	#daily_returns_2.iloc[0] = 0		    		
	return pd.DataFrame(daily_returns_2, columns=['C2O'])
	
def	BBANDS(df_price, periods=20, mul=2):	
	# Middle Band = 20-day simple moving average (SMA)
	df_middle_band = pd.rolling_mean(df_price, window=periods)
	#df_middle_band = pd.rolling(window=periods,center=False).mean()
	
	# 20-day standard deviation of price
	""" Pandas uses the unbiased estimator (N-1 in the denominator), 
	whereas Numpy by default does not.
	To make them behave the same, pass ddof=1 to numpy.std()."""	
	df_std = pd.rolling_std(df_price, window=periods)
	#df_std = pd.rolling(window=periods,center=False).std()
	
	# Upper Band = 20-day SMA + (20-day standard deviation of price x 2)
	df_upper_band = df_middle_band + (df_std * mul)
	
	# Lower Band = 20-day SMA - (20-day standard deviation of price x 2)
	df_lower_band = df_middle_band - (df_std * mul)	
	
	return (df_upper_band, df_middle_band, df_lower_band)

def get_BBANDS(df, periods=20, mul=2):	
	(upper, middle, lower) = BBANDS(df, periods, mul)		
	df_BBANDS = pd.concat([upper, middle, lower], axis=1, join='inner')
	df_BBANDS.columns = ['UPPER', 'MIDDLE', 'LOWER']
	return df_BBANDS

def get_myRatio(df_price, periods=20):
	# Middle Band = 20-day simple moving average (SMA)
	#df_middle_band = pd.rolling_mean(df_price, window=periods)
	df_middle_band = df_price.rolling(center=False, window=periods).mean()	
 
	# 20-day standard deviation of price
	""" Pandas uses the unbiased estimator (N-1 in the denominator), 
	whereas Numpy by default does not.
	To make them behave the same, pass ddof=1 to numpy.std()."""	
	#df_std = pd.rolling_std(df_price, window=periods)
	df_std = df_price.rolling(center=False, window=periods).std()
	
	return (df_price - df_middle_band)/(df_std * 2)

def sma(df, periods=12):
	# compute simple moving average
	#return pd.rolling_mean(df, window=periods)	
	return df.rolling(center=False, window=periods).mean()

# not sure	
def ema(df, periods=12):
	# compute exponential moving average
	#return pd.ewma(df, span = periods)
	return df.ewm(span=periods, adjust=True, min_periods=0, ignore_na=False).mean()

def average_convergence(df, period_low=26, period_fast=12):
    """
    compute the MACD (Moving Average Convergence/Divergence) 
	using a fast and slow exponential moving average'    
    """
    emaslow = ema(df, period_low)
    emafast = ema(df, period_fast)
    return (emaslow, emafast, emafast - emaslow)	

def signal_MACD(df_MACD, periods=9):
	return ema(df_MACD, periods)

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
		
	df_avg_gain.iloc[periods] = df_gain[1:periods+1].mean()	# First Average Gain = Sum of Gains over the past 14 periods / 14.
	df_avg_loss.iloc[periods] = df_loss[1:periods+1].mean()	# First Average Loss = Sum of Losses over the past 14 periods / 14
	
	for index in range(periods+1, len(df)):
		#Average Gain = [(previous Average Gain) x 13 + current Gain] / 14.	
		df_avg_gain.iloc[index] = (df_avg_gain.iloc[index-1] * 13 + df_gain.iloc[index])/periods
		
		#Average Loss = [(previous Average Loss) x 13 + current Loss] / 14.
		df_avg_loss.iloc[index] =	(df_avg_loss.iloc[index-1] * 13 + df_loss.iloc[index])/periods

	# RS = Average Gain / Average Loss	
	# if coding as bellow, it has a bug when df_avg_loss is zero (can't divid with zero)
	# RS = df_avg_gain/df_avg_loss
	# But I change coding with for loop instead
	RS = pd.DataFrame(columns = df.columns, index = df.index)
	for index in RS.index:		
		for sym in df.columns:			
			lossValue = df_avg_loss.loc[index][sym]
			if lossValue == 0:
				RS.loc[index][sym] = 100
			else:	
				RS.loc[index][sym] = df_avg_gain.loc[index][sym]/lossValue			
		
	#              100
    # RSI = 100 - --------
    #             1 + RS
	RSI = 100 - 100/(1 + RS)
	return RSI
	
def sharpe_ratio(rp, rf=None):	
	if rf is None:	
		rf = rp.copy()
		rf.iloc[0:] = 0
		
	# rp = Expected porfolio return
	# rf = Risk free rate
	ret = rp - rf
	# Sharpe ratio = mean(Expected porfolio return - Risk free rate)/Portfolio standard deviation
	return ret.mean()/ret.std()
	
def rolling_sharpe_ratio(rp, rf, window):
	# Example
	# rp = df_daily_return[symbol]
	# rf = df_daily_return['SET']	
	ret = rp - rf	
	mean = get_rolling_mean(ret, window)	
	std = get_rolling_std(ret, window )
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

def true_range(df):
	high = df['HIGH']
	low = df['LOW']
	previous_close = df['CLOSE'].shift(1)
	
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
	
	df_ATR = pd.DataFrame(columns = ['ATR'], index = df.index)
	df_ATR.iloc[periods-1] = df_TR[0:periods].mean()	# First ATR = Sum of TR over the past 14 periods / 14
		 
	for index in range(periods, len(df)):
		#Current ATR = [(Prior ATR x 13) + Current TR] / 14
		# - Multiply the previous 14-day ATR by 13.
		# - Add the most recent day's TR value.
		# - Divide the total by 14  		
		df_ATR.iloc[index] = (df_ATR.iloc[index-1] * 13 + df_TR.iloc[index])/periods
			
	return df_ATR

def getBeta(df, stock_name, benchmark_name):
	# Compute returns of stock	
	rs = roc(df[stock_name], periods=1)/100
	rb = roc(df[benchmark_name], periods=1)/100
	
	# Beta = Covariance(rs, rb)/Variance(rb)
	# where rs is the return on the stock and rb is the return on a benchmark index.
	return rs.cov(rb)/rb.var()	

def percent_KD(df, periods=14):
	"""
	%K = (Current Close - Lowest Low)/(Highest High - Lowest Low) * 100
	%D = 3-day SMA of %K

	Lowest Low = lowest low for the look-back period
	Highest High = highest high for the look-back period
	%K is multiplied by 100 to move the decimal point two places
	"""
	current = None
	if "ADJ CLOSE" in df.columns:
		current = df['ADJ CLOSE']
	else:
		current = df['CLOSE'] # Current Close
		
	low = df['LOW']	# Lowest Low
	high = df['HIGH']	# Highest High
	finish = len(df)-periods + 1
	
	K_ = pd.DataFrame(index=df.index, columns=['%K'])
	K_[0:] = np.float('nan')
	
	for index in range(0, finish):
		Highest_High = np.max(high[index: index+periods])		
		Lowest_Low = np.min(low[index: index+periods])
		position = index + periods - 1
		current_close = current.iloc[position]
		K_.iloc[position] = (current_close - Lowest_Low) /(Highest_High - Lowest_Low) *100
		
	D_ = sma(K_, periods=3)
	D_.rename(columns={'%K':'%D'},inplace=True)
	resultDf = K_.join(D_)
	return resultDf

def OBV(df_volume, df_close):			
	# create empty Data Frame
	df_OBV = pd.DataFrame(index = df_volume.index, columns = df_volume.columns)
	# first OBV
	df_OBV.iloc[0] = df_volume.iloc[0] 
		
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

			change = df_price_change.iloc[index][symbol]
			current_volume =  df_volume.iloc[index][symbol]
			
			if change > 0:			
				current_volume =  current_volume
			
			elif change < 0:
				current_volume =  -1 * current_volume
				
			else:	
				current_volume = 0
			
			df_OBV.iloc[index][symbol] = df_OBV.iloc[index -1][symbol]  + current_volume
	
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
	df_daily_returns = daily_returns(df)	
	df_daily_returns.plot(kind='scatter', x = x_stock, y = y_stock)
	
	X = df_daily_returns[x_stock] 
	Y = df_daily_returns[y_stock]
	
	beta, alpha = np.polyfit(X, Y, 1)
	print('Beta is {}, Alpha is {}'.format(beta, alpha))
	print('fx = {}x + {}'.format(beta, alpha))	
	#fx = beta*X + alpha	
	#plt.plot(X, fx, 'r-')
	#plt.show()
 
def isUpTrend(df_price , symbol, periods =14):	
	finish = len(df_price) - periods +1
	resultDf = pd.DataFrame(columns=df_price.columns) # empty
	
	for index in range(1, finish):
		sliced = df_price.iloc[index-1: index+periods] # slice n periods			
		# compute RSI
		result = rsi(pd.DataFrame(sliced, columns=df_price.columns))
		result = result.shift(-periods)
		result.dropna(0, inplace=True)
		compare = result >50 	# RSI > 50			
		resultDf = resultDf.append(compare)		
	
	return resultDf
			
def listLowVolatility(df):	
	allDaily = daily_returns(df)

	mean = allDaily.mean()
	std = allDaily.std()
	compare = np.abs(allDaily - mean ) > std
	assert compare.shape == allDaily.shape
	
	countVolatility = compare.sum()	
	indexList= np.argsort(countVolatility) # index of min values is at first of the queue 
	columName = allDaily.columns.values
	symbol = [columName[index] for index in indexList] 
	# order symbol names from low variant to high variant
	return symbol
 
def compute_gain(df, signal):    
    sum_buy = 0
    sum_sell = 0

    temp_pred = signal[0]
    if temp_pred == 1:    
        sum_buy += df.values[0][0]

    df_len = len(df)
    for index in range(1, df_len):
        pred = signal[index]
        close = df.values[index][0]
    
        if temp_pred  == pred:
            if index == df_len - 1 and pred == 1:
                sum_sell += close
            continue
           
        temp_pred = pred            
        if pred == 0:        
            sum_sell += close        
            continue     
        # if pred == 1
        sum_buy += close

    # Gain(%) = 100 x Sum(Sell(i) - Buy(i))/Sum(Buy(i))
    gain =  100 * (sum_sell - sum_buy)/sum_buy
    return gain