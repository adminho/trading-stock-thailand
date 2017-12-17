import indicator as ind
import dataset as data
import utility as utl
import utilgraph as utg

import numpy as np
import pandas as pd
import pickle


startDate = '2017-05-01'
endDate = utl.getCurrentDateStr()
dates = pd.date_range(startDate, endDate)	
symbolList = data.getAllSymbol()
df = utl.loadPriceData(symbolList, dates)

 
def max_sharpe_ratio(periods=5):	
	df_sr = ind.create_dataframe_SR(df, symbolList, window=periods)
	df_sr.fillna(0, inplace=True) # fill NaN to 0
	df_sr = df_sr.shift(-(periods-1))
	df_sr.dropna(0, inplace=True) # drop NaN at tail
	result = np.mean(df_sr)
	
     # Returns the indices that would sort an array
     # max values at last element
	maxIndexResult = np.argsort(result)
	nameSymbol = [ df_sr.columns[index] for index in reversed(maxIndexResult) if result[index] > 0 ]
	
	# save to dump file
	pickle.dump( nameSymbol, open( "symbol_list.p", "wb" ) )
	print("List max sharpe ratio\n",nameSymbol)	
	
	df_norm = utl.normalize_data(df)
	utg.plotStock(df_norm, nameSymbol[0:5], startDate, endDate, 'High sharpe ratio')

def plot_industry_roc(startDate, endDate):
    dates = pd.date_range(startDate, endDate)	
    symbolList = data.getSubIndustrySymbol()
    df = utl.loadPriceData(symbolList, dates)
    
    df_roc = ind.roc(df)    
    df_roc.fillna(0, inplace=True) # fill NaN to 0
    
    result = np.mean(df_roc)
    result = result.where(result>0)
    result = result.where(result>=result['SET'])
    
    maxIndexResult = np.argsort(result)
    nameSymbol = [ df_roc.columns[index] for index in reversed(maxIndexResult) if result[index] > 0 ]               
    print("List of Industry Group and Sector that ROC > SET (ROC)\n", nameSymbol)
    
    df_norm = utl.normalize_data(df)
    utg.plotStock(df_norm, nameSymbol, startDate, endDate, title='ROC: Industry Group and Sector')

def brek_new_high():        
    lasted = df.ix[-1:]
    n_period = 30
    previous = df.ix[-1*(n_period+1):-1]
    max_price = previous.max()
        
    result = (lasted-max_price)/max_price * 100
    result = result.values[0]
    maxIndexResult = np.argsort(result)
        
    nameSymbol = [ df.columns[index] for index in reversed(maxIndexResult) if result[index] > -0.5 ]
    print("List of break new high", nameSymbol)
    
    df_norm = utl.normalize_data(df)
    utg.plotStock(df_norm, nameSymbol[0:5], startDate, endDate, title='close to break new high')
    utg.plotStock(df_norm, nameSymbol[6:10], startDate, endDate, title='close to break new high')
    utg.plotStock(df_norm, nameSymbol[-6:-1], startDate, endDate, title='close to break new high')
    
# does't work    
def max_change(startDate, endDate):
    dates = pd.date_range(startDate, endDate)	
    symbolList = data.getAllSymbol()
    df = utl.loadPriceData(symbolList, dates)
    day4 = df.ix[-4:]       # select last dast in 4 days
    day4 = day4.iloc[::-1]  # reverse rows
    day4 = day4.pct_change() # calcuate percent change of prices
    day4 = day4[1:4]        # remove NaN in first rows
    day3 = day4.fillna(0)
    result = day3.mean()
    print(day3)
    maxIndexResult = np.argsort(result)
    nameSymbol = [ day3.columns[index] for index in reversed(maxIndexResult) if result[index] > 3 ]
    print("List of high percent change\n", nameSymbol)
    
    df_norm = utl.normalize_data(df)
    utg.plotStock(df_norm, nameSymbol[0:5], startDate, endDate, title='High percent change')

def plot_example():
    df_norm = utl.normalize_data(df)
    nameSymbol = ['TRUBB','SGP','JAS','ZMICO','BCP','VNT','SCG', 'DCON','PTTGC', 'IT', 'PTTEP']

    utg.plotStock(df_norm, nameSymbol, startDate, endDate, title='I am interested')

#plot_example()
#max_change('2017-03-01', curDateStr) 
brek_new_high() 
#plot_industry_roc('2016-12-01', '2017-01-15') 
max_sharpe_ratio() 