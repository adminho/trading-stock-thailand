import os.path
import numpy as np
import pandas as pd

import matplotlib.pyplot  as plt
import matplotlib

# my modules
import indicator as ind
import utilgraph as utg
import prepare_data as data

#matplotlib.use('Agg')
# directory for log file
LOG_PATH = 'log'
if os.path.exists(LOG_PATH) == False:
	os.makedirs(LOG_PATH)
PIC_PATH = 'temp_pic'
if os.path.exists(PIC_PATH) == False:
	os.makedirs(PIC_PATH)

# Declare buy and sell state (It is indexs of Q score list)
BUY = 1
SELL = 0
	
class Portfolio:	
	def __init__(self, name):
		self.buy_atprice = -1 # don't hold a stock
		self.sum_return = 0
		# for loging only
		self.all_date = []		# save all date
		self.all_price = []   	# save all close price
		self.all_signal = [] 	# save all signal BUY(1), SELL(0)
		self.all_gain = [] 		# save all gain 	
		self.all_pic = [] 		# for html log only
		self.name = name		
		
	def _buy(self, price):	
		# ...SELL SELL SELL SELL => BUY
		# buy after that calculate cumulative return return (don't save)
		if self.buy_atprice < 0:
			self.buy_atprice = price				
				
		# ...BUY BUY BUY BUY => BUY
		# buy and hold (don't re-buy)
		# and calculate cumulative return (don't save)
		_return = price/self.buy_atprice - 1		
		return self.sum_return + _return
	
	def _sell(self, price):
		# ...SELL SELL SELL SELL => SELL
		# don't take action and get old cumulative return
		if self.buy_atprice < 0:
			return self.sum_return
		
		# ...BUY BUY BUY BUY => SELL: 
		# buy and calculate cumulative return (save it)
		_return = price/self.buy_atprice - 1
		self.sum_return += _return
		self.buy_atprice = -1 # don't hold stock
		return self.sum_return

	def isHold(self):
		# -1 is not hold stock
		return self.buy_atprice > 0
		
	def update(self, action_index, date, price):		
		sum_return = 0
		if action_index == BUY:			
			sum_return = self._buy(price)
		elif action_index == SELL:
			sum_return = self._sell(price)
			
		self.all_date.append(date)			 # save all dates
		self.all_price.append(price)		 # save all prices
		self.all_signal.append(action_index) # save all signals BUY(1), SELL(0)
		self.all_gain.append(sum_return)	 # save all gain
		return sum_return
	
	def gain(self):
		return ind.compute_gain(pd.DataFrame(self.all_price), self.all_signal)
		
	def log(self, episode):		
		df = pd.DataFrame(index=range(0, len(self.all_date)))
		df['date'] = pd.DataFrame(self.all_date)
		df['price'] = pd.DataFrame(self.all_price)
		df['signal'] = pd.DataFrame(self.all_signal)
		df['cumulative return'] = pd.DataFrame(self.all_gain)
		
		file_name = "portfolio_%s_%s.csv" % (self.name, episode)
		file_name = os.path.join(LOG_PATH, file_name)
		df.to_csv(file_name ,index=False)
		return file_name
	
	def saveGraph(self, espisode):		
		df_price = pd.DataFrame(self.all_price, index=self.all_date)		
		df_signal = pd.DataFrame(self.all_signal, index=self.all_date)									
		df_buy = df_price.where(df_signal==BUY)	
		df_sell = df_price.where(df_signal==SELL)
		
		plt.figure(figsize=(10,8))	
		plt.title(self.name)
		plt.plot(self.all_date, self.all_price, 'c--', linewidth=0.5 ) 
		plt.plot(self.all_date, df_buy.values, '^r',  self.all_date, df_sell.values,'og')	
		
		file_name = self.name + "_" + str(espisode) + '.png'		
		file_name = os.path.join(PIC_PATH, file_name)
		#plt.tight_layout()
		plt.savefig(file_name)	
		plt.close()
		return file_name		
		
class Environment:
	def __init__(self, symbol, startDate, endDate):
		# Get stock data into dataframe (pandas)		
		self.all_data = data.getDataInd(symbol, startDate, endDate)
		print("Load total stock data(days): ", len(self.all_data))		
		self.all_data = pd.DataFrame(self.all_data[15:])	#skip blank on top's dataframe
		print("But use: %s\n" % len(self.all_data))
		
		self.index = 0  		# index of dataframe
		self.terminate = False 	# True is terminate trading
		self.port = Portfolio(symbol) # For calculate your portfolio				
				
		# this data to use for building the neural network model
		self.symbol = symbol
		self.num_action = 2 	# buy or sell					
		first_data = self.all_data.iloc[0]
		self.first_state  = self._getState(first_data)
		self.num_features = self.first_state.shape[1]
								
	def reset(self):
		self.index = 0
		self.terminate = False
		self.port = Portfolio(self.symbol)
	"""
	# next plan develop this method but now don't use
	def getReward(self, action_index, current_price, previous_price):
		# get total return of your portfolio
		total_return = self.port.update(action_index, current_price)		
		
		# hold the stock
		if self.port.isHold():
			# ... BUY BUY BUY BUY => BUY
			if action_index == BUY:	  
				# buy and hold (don't re-buy)
				# if portfolio is profit >> it's very good
				# if portfolio is lost 	 >> it's very bad
				return 2 * total_return
			# ... BUY BUY BUY BUY => SELL
			elif action_index == SELL:
				return total_return
		
		# don't tje hold stock
		if self.port.isHold() == False:
			# ... SEL SELL SELL SELL => BUY
			if action_index == BUY:	  
				return total_return
			# ... SEL SELL SELL SELL => SELL
			elif action_index == SELL:  
				# don't hold stock any more
				# if stock price is decrease, don't buy >> it's very good
				# if stock price is increse, don't buy  >> it's very bad
				return 	total_return + (previous_price/current_price - 1)
	"""			
	def _getState(self, next_data):
		# declare inner function here
		# encoding indicator to 0 or 1
		# positive, 			:encode to 0
		# 0 value or negative, 	:encode to 1
		def encode(df):
			# trick in Python 
			# 1 * True 	= 1
			# 1 * False = 0
			return 1* (df > 0)
		
		close = next_data['CLOSE'] # get close price (not use adjusted close price)
		# encoding these indicators
		c2o = encode(next_data['C2O'])						# 1
		daily = encode(next_data['DAILY'])					# 2
		roc = encode(next_data['ROC'])						# 3
		cross_ema = encode( close - next_data['EMA15']) 	# 4
		macd = encode(next_data['MACD'])					# 5
		up_bb1 = encode(close - next_data['UPPER_BB1P']) 	# 6
		low_bb1 = encode(next_data['LOWER_BB1P'] - close) 	# 7
		obv = encode(next_data['OBV'])						# 8
		change_obv = encode(next_data['CH_VOL'])			# 9
		# atr = encode(next_data['ATR'])				
		
		_rsi = next_data['RSI']			# get RSI indicator
		# rsi > 50, encode to 1
		# rsi < 50, encode to 0
		rsi_value 	= 1* (_rsi > 50)							# 10		
		# rsi > 80, over bought
		# rsi < 20, over sell
		rsi_over = 1* (_rsi > 80 or _rsi < 20)					# 11
						
		# posible states are 2^num_features = xxxxx
		# collect encoding data
		concat = [c2o, daily, roc, cross_ema, macd, 
						up_bb1, low_bb1, obv, change_obv, rsi_value, rsi_over]		
		return np.reshape(concat, (1,-1)) 					# shape is [1, num_features]
			
	def step(self, action_index):		
		# calculate reward from action, current price
		current_price = self.all_data.iloc[self.index]['CLOSE']					
		date = self.all_data.index[self.index]
		reward = self.port.update(action_index, date, current_price)
		if reward > 0 : 
			reward = 10
		elif reward<0 :
			reward = -10
			
		# terminate in trading a stock or not
		terminate = False
		next_state = None
		if 	len(self.all_data) <= self.index + 1: # final index
			terminate = True
		else:			
			# increase index of dataframe (increase days)
			self.index +=1 				
			# get all data for next state
			next_data = self.all_data.iloc[self.index] 
			next_state = self._getState(next_data)			
			
		return reward, next_state, terminate
	
	def getPortfolio(self):
		return self.port
