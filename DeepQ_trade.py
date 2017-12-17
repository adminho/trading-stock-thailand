# requiremnet
# pip install tensorflow
# pip install keras
# pip install pandas_datareader
# pip install fix_yahoo_finance

import datetime
import time
from time import gmtime, strftime
import numpy as np
import webbrowser
import os.path

# Requirement
# pip install tqdm
from tqdm import tqdm

# my package
from deep_q.env_trade import Environment, LOG_PATH
from deep_q.agent import DeepQAgent
from deep_q.animation import Visualization

import logging
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=logging.DEBUG)

np.random.seed(555)  # for reproducibility
"""
++++++++++++++For the Deep Q Learning algorithm+++++++++++
Mnih Volodymyr, Koray Kavukcuoglu, David Silver, Andrei A. Rusu, 
Joel Veness, Marc G. Bellemare, Alex Graves, Martin Riedmiller, 
Andreas K. Fidjeland, Georg Ostrovski, Stig Petersen, Charles Beattie, 
Amir Sadik, Ioannis Antonoglou, Helen King, Dharshan Kumaran, Daan Wierstra,
 Shane Legg, and Demis Hassabis. 
Human-level Control through Deep Reinforcement Learning. Nature, 529-33, 2015.

---------The pseudo-code here---------

Initialize replay memory D to size N
Initialize action-value function Q with random weights
for episode = 1, M do
    Initialize state s_1
    for t = 1, T do
        With probability ϵ select random action a_t
        otherwise select a_t=max_a  Q(s_t,a; θ_i)
        Execute action a_t in emulator and observe r_t and s_(t+1)
        Store transition (s_t,a_t,r_t,s_(t+1)) in D
        Sample a minibatch of transitions (s_j,a_j,r_j,s_(j+1)) from D
        Set y_j:=
            r_j for terminal s_(j+1)
            r_j+γ*max_(a^' )  Q(s_(j+1),a'; θ_i) for non-terminal s_(j+1)
        Perform a gradient step on (y_j-Q(s_j,a_j; θ_i))^2 with respect to θ
    end for
end for
"""



"""	
Next plan I try to use LSTM network
class ModelLSTM:
	def __init__(self, num_features, num_output):
		windows = 5					# windows slide
		self.num_sequence = windows
		self.model = self.buil_model(self.num_seq, num_features, num_output)		
	
	# This is neural network model for the the Q-function		
	def builModel(self, num_seq, num_features, num_output):				
		model = Sequential()		
		# I used GRU network		
		model.add(GRU(input_shape=(num_seq, num_features), units=20, return_sequences=True))
		model.add(BatchNormalization(trainable = True))	 
		model.add(GRU(units=20, return_sequences=False))
		model.add(BatchNormalization(trainable = True))			
		model.add(Dense(units=20, activation='relu'))
		model.add(BatchNormalization(trainable = True))
		model.add(Dense(num_output, activation='linear'))
		
		print("\nBuild model...")
		print(model.summary())
		
		try:
			if os.path.exists(self.h5_file):
				print("Loaded model(weights) from file: %s " % (self.h5_file))
				model.load_weights(self.h5_file)	
		except Exception as inst:
			print(inst)
	
		opt = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) 
		model.compile(optimizer=opt, loss='mse', metrics=['accuracy'])
		return model
	
	def prepareInput(self, all_data, index):
		# get sequence of data
		if 	len(all_data) <  index + self.num_sequence: # index out of bound
			return None
		else:
			return all_data[index: index+self.num_sequence] 
"""
# for debug only
def createLogHTML(prefix_name, log_path, all_pic):		
	def _convertListToCode(list):
		# convert list to code in string
		return "['" + "','".join(map(str, list )) + "']"	
	
	list_pic  = _convertListToCode(all_pic)
	list_pic = list_pic.replace("\\","/")	
	#output example 
	# list_pic =   ['temp_pic/BBL_0.png', 'temp_pic/BBL_1.png', 'temp_pic/BBL_2.png']	
	#list_gain = _convertListToCode(all_gain)
	
	f = open("log_template.html","r")
	html_template = f.read()		
	f.close()
	
	# inject lis_code into html template
	html = html_template % (log_path, list_pic)
	# and save html file	
	file_name = prefix_name + "_" + "log.html"
	f = open(file_name, "w")
	f.write(html)
	f.close()
	return file_name
	
def training_agent(symbol, startDate, endDate, total_episode = 100):	
	start_time = time.time()	
	# set Environment for trading	
	env = Environment(symbol, startDate, endDate)		
	agent = DeepQAgent(env)		
	all_gain, all_pic = [], []
	for episode in range(0, total_episode):			
		print("\n==================== Episode: %s ====================" % (episode+1))
		terminate =False			
		env.reset() 
		s_t = env.first_state	# get first state from environment
		port = env.getPortfolio()	# get portfolio
		step_observe = 0 			# count steps to observe before training my neural network model in Agent
		while( terminate != True ):
			# agent select (a_t): random action (Exploration) or the best action (depend the epsilon value)	
			a_t = agent.getAction(s_t)
			
			# Execute action a_t in Environment 
			# and observe reward(r_t), next state(s_t1), terminate trading or not
			r_t, s_t1, terminate = env.step(a_t)
			
			# save experiences in memory
			agent.saveExperience(s_t, a_t, r_t, s_t1, terminate)			
			# agent go to the next state			
			s_t = s_t1
			
			# replay experiences in memory (Update new Q scores into my neural network when step_observe meet condition)
			agent.replayExperienceWhen(step_observe)
			
			# for debug only
			step_observe += 1
			if step_observe%100 == 0: 
				print("Step trade at: %s | Cumulative return: %s " % (step_observe, port.sum_return))        
						
		# when finish each episode
		agent.saveModel()	# save the neural network model to files
		agent.clearMemory()	# clear all experiences in deque memory of the agent
		# reduce random action and increase select the best action (decrement the epsilon value)
		agent.reduceExplore(total_episode)		
		print("epsilon: %.2f" % agent.epsilon)
		
		#+++++++++ for debug only in each episode +++++++++++++++++++
		file_pic, file_csv, gain = port.report(episode+1)			
		print("Compute gain: ", gain)								
		print("Generate a graph figure to a file:", file_pic) 
		print("Log portfolio to a csv file: ", file_csv)	
		all_gain.append(gain)
		all_pic.append(file_pic)
		
	sec = datetime.timedelta(seconds=int(time.time() - start_time))	
	print ("\nCompilation Time : ", str(sec))			
	print("Training all episode finished!")
	print("************************")
	
	
	print("\n*********Runing on the best action**********")
	print("Run agent")
	env.reset()  # reuse same environment with reset to start trading
	terminate = False
	s_t = env.first_state
	log_agent = []
	while(terminate != True):
		a_t, qscore = agent.getBestAction(s_t)		
		_, s_t, terminate = env.step(a_t)
		log_agent.append( np.append(qscore, a_t))
	port = env.getPortfolio()
	file_pic, file_csv, gain = port.report(episode= "agent_run")	
	print("Compute gain: ", gain)								
	print("Generate a graph figure to files:", file_pic)
	print("Log portfolio to csv files: ", file_csv)	
	
	# create csv log file when runing on the best action	
	# for debug when run finally 
	import pandas as pd	
	df_csv = pd.read_csv(file_csv)	
	df_agent = pd.DataFrame(data=np.vstack(log_agent) ,columns=['sell_score', 'buy_score', 'best_action'])
	df_log = df_csv.join(df_agent, how="left")
	file_csv = os.path.join(LOG_PATH, "log_" + symbol + "_agent_run.csv")
	df_log.to_csv(file_csv) 

	# For debug only
	# create log html
	all_pic.append(file_pic)	
	dir_log = os.path.join("file:///" + os.path.dirname(os.path.abspath(__file__)), LOG_PATH)	
	file_log = createLogHTML(symbol, dir_log, all_pic)	
	# If your system set IE is default, it may be don't show logs on your webbrowser
	webbrowser.open('file://' + os.path.realpath(file_log))
	
if __name__ == "__main__" :
	# test for stock of thailand only
	symbol = 'PTT'
	startDate='2017-06-01'
	endDate = strftime("%Y-%m-%d", gmtime())
	training_agent(symbol, startDate, endDate, total_episode=10)