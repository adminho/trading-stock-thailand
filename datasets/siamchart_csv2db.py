import pandas as pd
import numpy as np
import json 
import sqlite3

from os import listdir 
from os.path import isfile, join 
from datetime import datetime
from tqdm import tqdm

CSV_PATH = 'D:/MyProject/Big-datasets/data_stock/sec_csv/'
DATABASE_PATH = 'db/stocks.db'

def drop_table(conn):
	c = conn.cursor()
	c.execute("DROP TABLE STOCKS")
	c.execute("DROP TABLE DATA")
	conn.commit()
	print("Drop table success")
	
def create_table(conn):	
	c = conn.cursor()
	c.execute('''CREATE TABLE STOCKS (
				ID interger primary key,
				SYMBOL text, 
				profile text)''')
	conn.commit()			
	c.execute('''CREATE TABLE DATA (
				ID_SYMBOL interger, 
				DATE text, 
				OPEN real, 
				HIGH  real, 
				LOW  real, 
				CLOSE  real, 
				VOLUME integer)''')	
	conn.commit()
	print("Create table success")
	
def covert_filed_date(record): # convert date format	
	# TEXT as ISO8601 strings ("YYYY-MM-DD HH:MM:SS.SSS").
	record[0] = datetime.strptime(str(int(record[0])), '%Y%m%d').strftime('%Y-%m-%d')    
	return record
	
def list_csv(dirPath=CSV_PATH, start=0, end=None):
	files_list = [f for f in listdir(dirPath) if isfile(join(dirPath, f))]
	if end == None :
		end = len(files_list)
	files_list = files_list[start:end] # select files (symbol.csv)
	return files_list

def insert_table(files_list, conn, dirPath=CSV_PATH):
	c = conn.cursor()
	data_stock = []
	for id, file in tqdm(enumerate(files_list)):		
		df = pd.read_csv(join(dirPath, file))
		#print(df.columns) # Index(['DATE', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME'], dtype='object')	
		symbol = file.replace('.csv','')
		data_stock.append((id, symbol , ""))		
		data = [ (id,) + tuple(covert_filed_date(record)) for record in df.values.tolist()]		
	c.executemany('INSERT INTO DATA VALUES (?,?,?,?,?,?,?)', data)	
	c.executemany('INSERT INTO STOCKS VALUES (?,?,?)', data_stock)
	conn.commit()

# We can also close the connection if we are done with it.
# Just be sure any changes have been committed or they will be lost.
# conn.close()
# https://docs.python.org/2/library/sqlite3.html
# https://www.w3schools.com/python/python_mongodb_getstarted.asp
# https://www.sqlite.org/docs.html

if __name__ == "__main__" :	
	conn = sqlite3.connect(DATABASE_PATH)
	drop_table(conn)
	create_table(conn)
	files_list = list_csv(end=5)
	insert_table(files_list, conn)
	
	c = conn.cursor()
	for row in c.execute(
		'''SELECT symbol, date, open, high, low, close, volume 
		FROM Stocks INNER JOIN Data ON stocks.id = data.id_symbol
		'''):	
		print(row)
	
	df = pd.read_csv(join(CSV_PATH, '!AGRO.csv'))
	for count in c.execute(
		'''SELECT count(date)
		FROM Stocks INNER JOIN Data ON stocks.id = data.id_symbol AND symbol like '!PROPCON'
		'''):
		print (count[0])
		
	assert len(df) == count[0]
	conn.close()
	