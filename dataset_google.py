# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 12:31:19 2017

@author: Administrator
"""

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
    shape_row = len(row_list)/num_col
    row_list = np.reshape(row_list, (shape_row,num_col)) 
    df=pd.DataFrame(columns = head_list, data = row_list)
    return df

def create_all_data(symbol, total_page=1):
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
    return df

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
df = create_all_data(symbol, total_page=2)
print(df.head())

# save csv file (all stock data)
removeOldFile(symbol) # clear old 
writeCSVFile(df, symbol)
    
