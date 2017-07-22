# reference code example from: https://github.com/martin-gorner/tensorflow-mnist-tutorial

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import numpy as np
import datetime

class Visualization:
	port = None	
	ispause = False
	am = None	
		
	def __init__(self, title, dpi=70):		
		fig, ax1 = plt.subplots()
		plt.gcf().canvas.set_window_title('Deep Q-learning')
		
		#fig = plt.figure()		
		#fig.set_facecolor('#FFFFFF')
		#ax1 = fig.add_subplot(1,2,1)
		#ax2 = fig.add_subplot(1,2,2)
		ax1.set_title(title)
		lineGreen, 	= ax1.plot([], [], 'go', linewidth=0.5)
		lineRed, 	= ax1.plot([], [], 'r^', linewidth=0.5)		
		lineClose, 	= ax1.plot([], [], 'c--', linewidth=0.5)		       
		plt.tight_layout()
	
		def init():
			#ax1.set_ylim(0, 1)	
			#ax1.set_xlim(0, 1)	
			"""
			ax1.set_title(title)		
			#margin = int( (min(datasetX) - max(datasetX))/10 )
			ax1.set_xlim(min(datasetX) , max(datasetX) )
			#margin = int( (min(datasetY) - max(datasetY))/10 )
			ax1.set_ylim(min(datasetY) , max(datasetY) )				
		
			ax2.set_title("Accuracy")
			ax2.set_xlim(0, len(accuracyList)) # initial value only, autoscaled after that
			ax2.set_ylim(0, max(accuracyList) +5) 			# not autoscaled
		
			ax3.set_title("Loss")
			ax3.set_xlim(0, len(lossList)) # initial value only, autoscaled after that
			ax3.set_ylim(0, max(lossList) + 0.1) # not autoscaled
			"""
			#ax2.set_xlim(0, 10)  # initial value only, autoscaled after that
			#plt.tight_layout()
			return lineGreen, lineRed, lineClose
	
		def update():
			# for ax1
			buy, sell, price, date  = self.port.getAllData() 
			margin = (max(price) - min(price) ) * 0.1
			ax1.set_ylim(min(price) - margin , max(price) +  margin)
			#ax1.set_ylim(min(price) , max(price))			
			ax1.set_xlim(date[0], date[-1:])   			
								
			lineRed.set_data(date, sell)			
			lineGreen.set_data(date, buy)				
			lineClose.set_data(date, price)
						
			"""
			if self.step == 0 :
				# clear all graph
				lineGreen.set_data([], [])				
				lineRed.set_data([], [])
				lineClose.set_data([], [])
				lineGain.set_data([], [])			
				return init()
			"""
			return lineGreen, lineRed, lineClose
			
		def key_event_handler(event):
			if len(event.key) == 0:
				return
			else:
				keycode = event.key

			# pause/resume with space bar
			if keycode == ' ':
				self.ispause = not self.ispause
				#if not self.ispause:
				#	update()
				return
			# other matplotlib keyboard shortcuts:
			# 'o' box zoom
			# 'p' mouse pan and zoom
			# 'h' or 'home' reset
			# 's' save
			# 'g' toggle grid (when mouse is over a plot)
			# 'k' toggle log/lin x axis
			# 'l' toggle log/lin y axis			
			
			#plt.draw()
			#update()

		fig.canvas.mpl_connect('key_press_event', key_event_handler)
		self._fig = fig
		self._init = init
		self._update = update
	
	def updatePortfolio(self, port):
		self.port = port
			
	def train(self, training_agent, total_episode, save_movie=False):
		def animate_func(step):		
			#self.step = step		
			if step > total_episode:				 
				print("Finish training ")				
				#plt.close()
			else:
				training_agent(episode=step, visual=self)
				plt.pause(1.001) # makes the UI a little more responsive 
				
			if not self.ispause:
				return self._update()

		self.am = animation.FuncAnimation(self._fig, animate_func, total_episode, init_func=self._init, interval=16, repeat=False, blit=False)
		if save_movie:
			mywriter = animation.FFMpegWriter(fps=24, codec='libx264', extra_args=['-pix_fmt', 'yuv420p', '-profile:v', 'high', '-tune', 'animation', '-crf', '18'])
			self.am.save("./video.mp4", writer=mywriter)
		else:
			plt.show(block=True)
	