import os
from os import listdir, makedirs
from os.path import isfile, join, exists
import json 
import numpy as np
import pandas as pd
import shutil
from collections import namedtuple	

CSV_PATH = 'D:/MyProject/Big-datasets/data_stock/sec_csv/'
SEC_JSON_PATH = "sec_json"

def clearDir(dirPath=SEC_JSON_PATH):
	if exists(dirPath):			
		shutil.rmtree(dirPath)	
	makedirs(dirPath)	

def list_csv(dirPath=CSV_PATH, start=0, end=None):
	files_list = [f for f in listdir(dirPath) if isfile(join(dirPath, f))]
	if end == None :
		end = len(files_list)
	files_list = files_list[start:end] # select files (symbol.csv)
	return files_list

#convert csv to json file
def csv_to_json(file_json, files_csv, jsonPath=SEC_JSON_PATH, csvPath=CSV_PATH ):
	clearDir() # delete old files	
	with open(join(jsonPath, file_json), 'w') as f:
		for id, file in enumerate(files_csv):		
			df = pd.read_csv( join(csvPath, file))
			data= df.to_json(orient='records') # data = df.to_json(orient='index')
			symbol = file.replace('.csv','')		
			document = f'{{"_id": "{id}", "stock": "{ symbol }", "profile":"", "data":{data}}}' 				
			f.write('%s\n' % document)

stock = namedtuple('Stock', ['symbol','data'])
# convert json to dataframe in pandas
def json_to_df(file_json, jsonPath=SEC_JSON_PATH):
	with open(join(jsonPath, file_json)) as f:
		list_stock = []
		linelist = f.readlines()
		for document in linelist:			
			record = json.loads(document)
			data = record['data']
			df_record = pd.DataFrame.from_dict(data, orient='columns')
			s = stock(record['stock'], df_record)
			list_stock.append(s)
		return list_stock

if __name__ == "__main__" :		
	file = 'stock.json'
	files_csv = list_csv(end=50) # select 50 files
	csv_to_json(file, files_csv)
	stock_list = json_to_df(file)	
	print(stock_list[0][1].head())	
	assert len(stock_list) == len(files_csv)		