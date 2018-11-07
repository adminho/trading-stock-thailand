import numpy as np
import pandas as pd
import os.path
from time import gmtime, strftime

ACTION_SPACE = [-1,0,1] # sell (-1), nothing (0) and buy (+1). 

STATE_PATH = 'state'
if os.path.exists(STATE_PATH) == False:
	os.makedirs(STATE_PATH)	
	
for root, dirs, files in os.walk(STATE_PATH):
	for file in files:
		os.remove(os.path.join(root, file))
		
class SingleStockMarket:
	def __init__(self, symbol, stock, days=60, commission = 1e-3):
		#constants
		self.start_idx = 0 + days
		self.seq_len = days		
		self.n_actions = len(ACTION_SPACE)
		self.df = stock
		self.symbol = symbol		
		self.df = self.__prepare_data(stock) # prepare_data
		
		self.commission = commission
		#reset semi-constants
		self.reset()
	
	def __prepare_data(self, stock):		
		df = stock.copy()
		df['position'] = 0	#postion; -1, 0, 1 = short, hold, long -> action 0, 1, 2
		df['return'] = df['CLOSE'].shift(-1) / df['CLOSE'] - 1		# return
		df['commission'] = 0 										#commission	
		df.iloc[:,:5] = df.iloc[:,:5] / df.iloc[:,:5].shift(1) - 1 	#state
	
		#filter NaN at first row and last row
		df = df.iloc[1:-1,:]	
		assert (len(stock)-2 == len(df)) ,"length of dataframe don't match"
		return df
	
	def save_state(self, i):
		#save
		#torch.save(self.qnetwork_local.state_dict(), model_name)		
		#self.env.df.to_csv(df_name,index=False)		
		file_csv = os.path.join( STATE_PATH, f'state_{self.symbol}_{i}.csv' )
		self.df.to_csv(file_csv, index=True)
		
	def reset(self):
		self.idx =  self.start_idx
		self.open_idx = self.start_idx - self.seq_len
		#set all positions to zero except for first seq_len to be all hold
		self.df['position'] = 0
		self.df.iloc[:self.idx,self.df.columns.get_loc('position')] = 1
		self.current_state = self.get_state()
		return self.current_state
		
	def step(self, action):
		action = int(action)
		state = self.get_state()
		old_sharpe = self.get_sharpe()
		#position
		self.df.iloc[self.idx,self.df.columns.get_loc('position')] = ACTION_SPACE[action]
		#if new direction; record commission for closing and opening a position
		if (ACTION_SPACE[action] != self.df.iloc[self.idx-1, self.df.columns.get_loc('position')]):
			mult = (self.df.iloc[self.open_idx:self.idx , self.df.columns.get_loc('return')]+1).prod()
			#for closing a position
			self.df.iloc[self.idx,self.df.columns.get_loc('commission')] = self.commission * mult 
			#for opening a new one; 0 for hold
			self.df.iloc[self.idx,self.df.columns.get_loc('commission')] += self.commission if ACTION_SPACE[action]!=0 else 0
			self.open_idx = self.idx
		
		#go to next timestamp
		self.idx+=1
		
		next_state = self.get_state()
		#differential sharpe as reward
		#absolute
#		 reward = self.get_sharpe() - old_sharpe
		#percentage
		#reward = self.get_sharpe() / old_sharpe - 1 
		reward = self.get_sharpe() / (old_sharpe + 1e-20) - 1  #protect divide by zero
		reward = np.nan_to_num(reward)
		reward = np.clip(reward,-1,1)
		done = True if self.idx==self.df.shape[0] else False
		info = f'Currently at index {self.idx}'
		
		return (reward,next_state,done,info)
	
	def get_state(self):
		#current state at timestamp is everything BEFORE current prices (up to sequence length)
		from_idx = self.idx - self.seq_len
		#see as much as the timestamp before
		to_idx = self.idx
		#get open,high,low,close,vwap
		state = np.array(self.df.iloc[from_idx:to_idx, 0:5])
		#standardize to be N(0,1)
		state_norm = (state - state.mean(axis=0)) / (state.std(axis=0))
		positions = self.df.iloc[from_idx:to_idx,self.df.columns.get_loc('position')][:,None]
		state = np.concatenate([state_norm,positions],axis=1).transpose()[None,:,:]
		return state
	
	def get_sharpe(self, rfr = 0.02 / 525600):
		returns = self.get_returns(rfr)
		#s = np.nanmean(returns) / (np.nanstd(returns) * np.sqrt(self.idx))
		s = np.nanmean(returns) /  ( np.nanstd(returns) * np.sqrt(self.idx) )   #protect divide by zero
		s = np.nan_to_num(s)
		s = np.clip(s,-1,1)
		return s
	
	def get_returns(self,rfr = 0.02 / 525600):
		benchmark_returns = self.df['return'][self.start_idx:self.idx] - self.df['commission'][self.start_idx:self.idx] - rfr
		current_position = self.df['position'][self.start_idx:self.idx]
		returns = current_position * benchmark_returns
		return(returns)	
