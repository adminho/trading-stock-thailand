import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt	
import itertools
from datasets import siamchart_csv as csv
from time import gmtime, strftime
from tqdm import tqdm
from trader.environment import SingleStockMarket
from trader.qnetwork import QNetwork
from trader.agent import StockAgent 
from tqdm import tqdm

def train_agent():
	# Initiate environment
	symbol = 'AOT'
	startDate = '2017-03-01'
	endDate = strftime("%Y-%m-%d", gmtime())
	dates = pd.date_range(startDate, endDate)				
	stock = csv.load_OHLCV(symbol, dates)	
	env = SingleStockMarket(symbol, stock)

	# Create an agent.
	qnet = QNetwork(env)
	agent = StockAgent(qnet)

	#Train the agent.
	state = env.reset()

	for i in tqdm(itertools.count(start=0, step=1)):
		#select action
		action = agent.select_action(state,i)  			
		#step
		reward,next_state,terminate,info = env.step(action)			
		agent.save_experience(state, action, reward, next_state, terminate, i)
		state = next_state
		if terminate == True:
			env.save_state(i)
			break # training is finished

if __name__ == "__main__" :	
	train_agent()

