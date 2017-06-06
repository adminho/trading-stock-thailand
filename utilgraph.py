import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, WeekdayLocator,DayLocator, MONDAY
from matplotlib.finance import candlestick_ohlc

import indicator as ind
import utility as utl
import utilmodel as utm

def plotStock(df, columns, start_index, end_index, title="Selected data"):
	"""Plot the desired columns over index values in the given range."""	
	df_plot = df.ix[start_index:end_index, columns]
	
	"""Plot stock prices with a custom title and meaningful axis labels."""
	ax = df_plot.plot(title=title, fontsize=12)
	ax.set_xlabel("Date")
	ax.set_ylabel("Price")
	plt.show()

# Borrowed code from : https://www.udacity.com/course/machine-learning-for-trading--ud501
def plotDailyHist(df, symbol, title="Selected data"):
	"""Plot the desired columns over index values in the given range."""		
	df_plot = ind.daily_returns(df)[symbol]
		
	"""Plot stock prices with a custom title and meaningful axis labels."""	
	df_plot.hist(bins=20)
	ax = plt.axvline(df_plot.mean(), color='w', linestyle='dashed',linewidth=2)
	std = df_plot.std()
	plt.axvline(std,  color='r', linestyle='dashed',linewidth=2)
	plt.axvline(-std, color='r', linestyle='dashed',linewidth=2)
		
	#ax.set_xlabel("Daily returns")
	#ax.set_ylabel("Frequency")
	plt.show()

# Borrowed code from : http://matplotlib.org/examples/pylab_examples/finance_demo.
def plotCandlestick(symbol, start_index, end_index, title="Selected data"):
	dates = pd.date_range(start_index, end_index)	
	quotes = utl.loadStockQuotes(symbol, dates)		
	
	mondays = WeekdayLocator(MONDAY)        # major ticks on the mondays
	alldays = DayLocator()              	# minor ticks on the days
	weekFormatter = DateFormatter('%b %d')  # e.g., Jan 12
	dayFormatter = DateFormatter('%d')      # e.g., 12

	fig, ax = plt.subplots()
	fig.subplots_adjust(bottom=0.2)
	ax.xaxis.set_major_locator(mondays)
	ax.xaxis.set_minor_locator(alldays)
	ax.xaxis.set_major_formatter(weekFormatter)
	#ax.xaxis.set_minor_formatter(dayFormatter)
 
	#plot_day_summary(ax, quotes, ticksize=3)
	candlestick_ohlc(ax, quotes, width=0.6)

	ax.xaxis_date()
	ax.autoscale_view()
	plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')

	plt.show()
	
def plotLine(data):	
	plt.figure(figsize=(6, 3))
	plt.plot(data)
	plt.ylabel('error')
	plt.xlabel('iteration')
	plt.title('training error')
	plt.show()

def _plotColorLine(close, Ydigit, ax):
	# sytles = ['-r', '-g']		
	green = close.where(Ydigit==1)	# new height
	red = close.where(Ydigit==0)	# not
		
	ax.plot(red.index, red.values, '-r',  green.index, green.values,'-g')	
	ax.legend(['Not', 'predict new high'])
			
def plot1ColLine(symbol, close, Ydigit, title):
	_plotColorLine(close, Ydigit, plt)	
	plt.title(title)
	plt.xlabel('Dates')
	plt.ylabel('Price')
	plt.show()
	
def plot2ColLine(symbol, X1, Y1, title1, X2, Y2, title2):
	fig = plt.figure()
	plt.gcf().canvas.set_window_title(symbol)
	fig.set_facecolor('#FFFFFF')
		
	ax1 = fig.add_subplot(1,2,1)
	ax1.set_title(title1)
	ax1.set_xlabel('Dates')
	ax1.set_ylabel('Price')
	ax1.get_xaxis().set_visible(False)
	
	ax2 = fig.add_subplot(1,2,2)
	ax2.set_title(title2)
	ax2.set_xlabel('Dates')
	ax2.set_ylabel('Price')
	ax2.get_xaxis().set_visible(False)
	
	_plotColorLine(X1, Y1, ax1)
	_plotColorLine(X2, Y2, ax2)
	
	PIC_PATH = 'debug'
	if os.path.exists(PIC_PATH) == False:
		os.makedirs(PIC_PATH)

	plt.savefig(os.path.join(PIC_PATH, '%s.jpg' % (symbol)))
	plt.show()
	
def plotPCA2d(Xpca, Ydigit):
	colors = ['red', 'green']	
	for number in range(0,2): # 0 to 1
		XY = Xpca[np.where(Ydigit == number)[0]]
		# seperate to x, y component
		x = XY[:, 0]	
		y = XY[:, 1]
		plt.scatter(x, y, c=colors[number])
		
	plt.legend(['Not', 'predict new high in 14 days'])
	plt.xlabel('First Principal Component')
	plt.ylabel('Second Principal Component')
	plt.show()

def plotPCA3d(Xpca, Ydigit):	
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')	
	colors = ['red', 'green']
	for number in range(0,2): # 0 to 1
		XYZ = Xpca[np.where(Ydigit == number)[0]]
		# seperate to x, y, z component
		x = XYZ[:, 0]	
		y = XYZ[:, 1]
		z = XYZ[:, 1]
		ax.scatter(x, y, z, c=colors[number])
	
	plt.legend(['Not', 'predict new high in 14 days'])
	ax.set_xlabel('First Principal Component')
	ax.set_ylabel('Second Principal Component')
	ax.set_zlabel('Third Principal Component')
	plt.show()
	
def plotPCA(X, Y):	
	# Visualize data
	# convert to 2 components
	Xpca = utm.getPCAvalues(X , 2)
	assert Xpca.shape[1] == 2
	plotPCA2d(Xpca, Y)
	
	# convert to 3 components
	Xpca = utm.getPCAvalues(X , 3)
	assert Xpca.shape[1] == 3
	plotPCA3d(Xpca, Y)

