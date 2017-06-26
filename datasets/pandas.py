import pandas as pd
import numpy as np
from pandas_datareader import data as pdr

import fix_yahoo_finance as yf # <== that's all it takes :-)

def load_OHLCV(symbol, startdate, enddate):
    return pdr.get_data_yahoo(symbol+".BK", start=startdate, end=enddate)

def loadBigData(symbol_list, startdate, enddate):	
	datas = map (lambda symbol: load_OHLCV(symbol, startdate, enddate), symbol_list)	
	return(pd.concat(datas, keys=symbol_list, names=['Symbol', 'Date']))

def selectColumn(all_data, column_name):    
    all_symbol = set(all_data.index.get_level_values(0))    
    df = None
    for symbol in all_symbol:        
        df_col = all_data[[column_name]]        
        df_temp = df_col.iloc[df_col.index.get_level_values('Symbol') == symbol]
        df_temp.index = df_temp.index.droplevel('Symbol')        
        df_temp = df_temp.rename(columns={column_name: symbol}) # prevent column_name overlap
        
        if df is None:
            df = df_temp
        else:
        	  df = df.join(df_temp) # left join by default
    return df

    
def loadPriceData(symbol_list, startdate, enddate):
    all_data = loadBigData(symbol_list, startdate, enddate)     
    return selectColumn(all_data, 'Adj Close')

# Meaning Adjusted Closing Price
# http://www.investopedia.com/terms/a/adjusted_closing_price.asp
# https://test.set.or.th/th/products/index/files/2011-10-31-TRI_calculation_Method_Th.pdf
from time import gmtime, strftime
if __name__ == "__main__":    
    #startdate = datetime.datetime(2010, 10, 1)       
    startDate = '2017-03-01'
    endDate = strftime("%Y-%m-%d", gmtime())    
        
    df = load_OHLCV("PTT", startDate, endDate)
    print()
    print(df.tail())
    
    symbol_list = ["PTT", "AOT", "SCC", "CPALL"]
    all_data = loadBigData(symbol_list, startDate, endDate)    
    print(all_data)
    
    adj_close = all_data[['Adj Close']]
    print(adj_close)
    
    df = loadPriceData(symbol_list, startDate, endDate)    
    #df.fillna(0, inplace=True)
    print(df.tail())  
  