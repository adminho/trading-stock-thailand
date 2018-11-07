from collections import defaultdict, namedtuple, deque
import random
import numpy as np

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state','done'))
class ReplayMemory:
	def __init__(self, capacity, seed = 1412):		
		self.capacity = capacity
		self.memory = deque(maxlen=capacity) 
		#self.seed = random.seed(seed)
		random.seed(seed)	
	'''
	def save(self, *args):
		t = Transition(*args)
		self.memory.append(t)		
	'''
	def save(self, state, action, reward, next_state, terminate):
		tuple = (state, action, reward, next_state, terminate)
		self.memory.append( tuple )
	'''
	def sample(self, batch_size):
		ts = random.sample(self.memory, batch_size)		
		states = np.vstack([t.state for t in ts])
		actions = np.vstack([t.action for t in ts])
		rewards = np.vstack([t.reward for t in ts])
		next_states = np.vstack([t.next_state for t in ts])
		dones = np.vstack([t.done for t in ts])
		return (states,actions,rewards,next_states,dones)
	'''
	def get_sample(self, batch_size):
		return random.sample(self.memory, batch_size)				
		
	def __len__(self):
		return(len(self.memory))
		
		
class PrioritizedMemory:
	"""
	https://arxiv.org/pdf/1511.05952.pdf
	"""
	def __init__(self, capacity, alpha = 0.6):
		self.capacity = capacity
		self.memory = deque(maxlen=capacity)
		self.alpha = alpha
		self.priority = deque(maxlen=capacity)
		self.probs = np.zeros(capacity)
	def add(self, *args):
		max_prior = max(self.priority) if self.memory else 1.
		t = Transition(*args)
		self.memory.append(t)
		#give latest transition max priority for optimistic start
		self.priority.append(max_prior)
	def prior_to_prob(self):
		probs = np.array([i**self.alpha for i in self.priority]) #uniform sampling when alpha is 0
		self.probs[range(len(self.priority))] = probs
		self.probs /= np.sum(self.probs)
	def sample(self, batch_size, beta = 0.4):
		#calculate prob every time we will sample
		self.prior_to_prob()
		idx = np.random.choice(range(self.capacity), batch_size, replace=False, p=self.probs)
		ts = [self.memory[i] for i in idx]
		
		#stitch tuple
		states = np.vstack([t.state for t in ts])
		actions = np.vstack([t.action for t in ts])
		rewards = np.vstack([t.reward for t in ts])
		next_states = np.vstack([t.next_state for t in ts])
		dones = np.vstack([t.done for t in ts]).astype(np.uint8)
		
		#importance sampling weights
		sampling_weights = (len(self.memory)*self.probs[idx])**(-beta) #higher beta, higher compensation for prioritized sampling
		sampling_weights = sampling_weights / np.max(sampling_weights) #normalize by max weight to always scale down
		sampling_weights = sampling_weights
		
		return(states,actions,rewards,next_states,dones,idx,sampling_weights)
	def update_priority(self,idx,losses):
		for i, l in zip(idx, losses):
			self.priority[i] = l.data
		
	def __len__(self):
		return(len(self.memory))
