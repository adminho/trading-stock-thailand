# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 12:31:19 2017

@author: Administrator
"""

import urllib
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
from os.path import join, exists
from os import  remove, makedirs


# https://www.set.or.th/set/factsheet.do?symbol=BBL&ssoPageId=3&language=en&country=US
def getTableData(symbol):
    url_string = "https://www.set.or.th/set/factsheet.do?symbol={0}&ssoPageId=3&language=en&country=US".format(symbol)
        
    page = urllib.request.urlopen(url_string).read()           
    soup = BeautifulSoup(page, 'lxml')    
    table_element =soup.findAll('table', class_='table-factsheet-padding0')        
    return table_element[19], url_string

def createDataFrame(table_element):   
    #Generate lists
    row_list =[]
    index_list = []
    if table_element is None:
        return None
    
    num_col = 6
    count =0
    for row in table_element.findAll("td"):        
        txt = row.find(text=True)        
        count = count+1
        if count % num_col == 1:
            txt = txt.replace('-', '*')
            index_list.append(txt)
        else:
            row_list.append(txt)
        
    index_list = index_list[1:len(index_list)-1] # skip 'Statement of Comprehensive Income (MB.) and 'more'
    
    num_col_df = num_col-1
    shape_row = len(row_list)/num_col_df
    row_list = np.reshape(row_list, (shape_row,num_col_df)) 
    
    all_head = row_list[0]
    all_row = row_list[1:]    
    df=pd.DataFrame(columns = all_head, index = index_list, data = all_row)
    return df

DIR_SEC_CSV = "sec_set_income"
def writeCSVFile(df, symbol, output_path=DIR_SEC_CSV, include_index = False):
    csv_file = "{}.csv".format(join(output_path, symbol))    
    df.to_csv(csv_file, index = include_index)    
        
def removeOldFile(symbol, output_path=DIR_SEC_CSV):
    csv_file = "{}.csv".format(join(output_path, symbol))    
    if exists(output_path) == False:
        makedirs(output_path)
    if exists(csv_file):			
        remove(csv_file)	

if __name__ == "__main__" :
    symbol_list= ['DCORP', 'MEGA', 'UEC', 'APURE', 'GOLD', 'MTLS', 'JMART', 'TWPC', 'BEAUTY', 'CPN']
    for symbol in symbol_list:
        table_element, url_string = getTableData(symbol)
        print('\n%s' % url_string)
        df = createDataFrame(table_element)
        print(df)
        
        removeOldFile(symbol) # clear old
        writeCSVFile(df, symbol, include_index = True)
