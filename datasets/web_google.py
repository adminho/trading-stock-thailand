# -*- coding: utf-8 -*-

# How to install
# references: https://pypi.python.org/pypi/googlefinance
# pip install googlefinance

from googlefinance import getQuotes
import json

import urllib
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
from os.path import join, exists
from os import  remove, makedirs
import datetime

# http://www.google.com/finance/historical?q=AAPL
# http://www.google.com/finance/historical?q=BKK:BBL&start=0&num=30
def getTableData(symbol, page=1):
	url_string = "http://www.google.com/finance/historical?q=BKK:{0}".format(symbol)
	url_string += '&start={0}&num=30'.format((page-1)*30)
	
	page = urllib.request.urlopen(url_string).read()		   
	soup = BeautifulSoup(page, 'lxml')	
	table_element =soup.find('table', class_='gf-table historical_price')
	return table_element, url_string

def createDataFrame(table_element):   
	#Generate lists
	row_list =[]
	head_list = []
	
	if table_element is None:
		return None

	for row in table_element.findAll("tr"):
		head = row.findAll('th') #To store column head
		if len(head) != 0:
			for h in head:
				head_list.append(h.find(text=True).replace('\n',''))
			
		rows = row.findAll('td')
		if len(rows) != 0: #Only extract table body not heading				
			for r in rows:
				row_list.append(r.find(text=True).replace('\n',''))
	
	num_col = len(head_list)			
	shape_row = int(len(row_list)/num_col)
	row_list = np.reshape(row_list, (shape_row,num_col)) 
	df=pd.DataFrame(columns = head_list, data = row_list)
	return df

def getAllDataframe(symbol, total_page=1):
	# get stock data from set.or.th web (total_page)
	df = None
	for p in range(1, total_page+1):
		table_element, url_string = getTableData(symbol, page=p) 
		print(url_string)
		
		df_temp = createDataFrame(table_element)
		if df is None:
			df = df_temp
		else:		
			df = df.append(df_temp)
	
	#renew data structure of dataframe
	df_date = df['Date']
	# convert string to date
	date_list = [ datetime.datetime.strptime(str_date, '%b %d, %Y') for str_date in df_date.values]		
	columnNames = df.columns[1:] # skip 'Date' column
	df = df[columnNames]
	# change column names
	columnNames = [name.upper() for name in columnNames]
	df_new = pd.DataFrame(data=df.values, index=date_list, columns=columnNames)	
	# order current date to old date
	df_new = df_new.reindex(index=df_new.index[::-1])
	return df_new

def load_OHLCV(symbol, startDate, endDate):
	df = getAllDataframe(symbol, total_page=2)
	return df[startDate: endDate]

DIR_SEC_CSV = "sec_google_price"
def writeCSVFile(df, symbol, output_path=DIR_SEC_CSV, include_index = False):
	csv_file = "{}.csv".format(join(output_path, symbol))	
	df.to_csv(csv_file, index = include_index)	
		
def removeOldFile(symbol, output_path=DIR_SEC_CSV):
	csv_file = "{}.csv".format(join(output_path, symbol))	
	if exists(output_path) == False:
		makedirs(output_path)
	if exists(csv_file):			
		remove(csv_file)	

symbol = 'BBL'		
print(json.dumps(getQuotes('SET:' + symbol), indent=2)) # for stock thai

# get stock data from Google web
# Becareful: If you connect to web very frequency, Google will thik you that robot
df = getAllDataframe(symbol, total_page=2)
print("------------- tail -------------")
print(df.tail())

# save csv file (all stock data)
removeOldFile(symbol) # clear old 
writeCSVFile(df, symbol)

df_select = load_OHLCV('SET', '20170615', '20170619')
print("-------select some date --------")
print(df_select)
