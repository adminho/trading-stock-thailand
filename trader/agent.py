import random
import numpy as np
from trader.replay_memory import ReplayMemory
from trader.qnetwork import QNetwork

class StockAgent:
	def __init__(self, qnetwork, mem_size = 10000, batch_size = 64, update_interval = 5, seed = 1412):		
		random.seed(seed)
		self.batch_size = batch_size
		self.update_interval = update_interval		
		self.losses = []
		self.qnetwork = qnetwork		
		# replay memory
		self.memory = ReplayMemory(mem_size)
		
	def reduce_epsilon(self, i, eps_start = 1.0, eps_end = 0.001, eps_decay = 0.999):
		eps = max(eps_start * (eps_decay ** i), eps_end)
		return(eps)
	
	def save_experience(self, state, action, reward, next_state, terminate, t_step):
		#add transition to replay memory
		self.memory.save(state, action, reward, next_state, terminate)
		# learn every self.t_step
		#self.t_step += 1
		if t_step % self.update_interval == 0:
			if len(self.memory) > self.batch_size:				
				transitions = self.memory.get_sample(self.batch_size)
				self.qnetwork.learn(transitions)

	def select_action(self, state, t_step):		
		epsilon = self.reduce_epsilon(t_step)		
		#epsilon greedy
		if random.random() > epsilon:
			action_values = self.qnetwork.output(state)
			return np.argmax(action_values)			
		else:
			return random.choice([0, 1, 2]) # 0, 1 or 2	