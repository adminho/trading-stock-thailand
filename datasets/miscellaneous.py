import sys

# First example
# How to install googlefinance package
# Reference: https://pypi.python.org/pypi/googlefinance
# pip install googlefinance
print("+++++googlefinance example+++++")
from googlefinance import getQuotes
import json
try:
    symbol = 'PTT'
    print(json.dumps(getQuotes('SET:' + symbol), indent=2)) # for Stock of Thailand, There is a prefix with 'SET:'
    print()
except:
    print("Error:", sys.exc_info()[0])
    print("Description:", sys.exc_info()[1])
#http://www.google.com/finance/company_news?output=json&q=GOOG&start=0&num=1000

# Second example
# How to install quandl package
# https://github.com/quandl/quandl-python
# pip install quandl
print("+++++quandl example+++++")
import quandl
try:
    quandl.ApiConfig.api_key = 'YOUR_API_KEY' #(must register at https://www.quandl.com/)
    print("THAISE index:")
    data = quandl.get("THAISE/INDEX")
    #data = quandl.get("THAISE/INDEX", authtoken="YOUR_API_KEY")
    print(data.head())
except:
    print("Error:", sys.exc_info()[0])
    print("Description:", sys.exc_info()[1])

try:
    print("\nAAPL:")
    data = quandl.get("WIKI/AAPL", start_date="2006-10-01", end_date="2012-01-01")
    print(data.head())
    print()
except:
    print("Error:", sys.exc_info()[0])
    print("Description:", sys.exc_info()[1])


# Third example
# How to install pandas-datareader package
# Reference: https://github.com/pydata/pandas-datareader
# pip install pandas-datareader

# But does't work, It has some bug
# Fix bug: https://github.com/ranaroussi/fix-yahoo-finance (The future, I think that the bug is solved)
# pip install fix_yahoo_finance --upgrade --no-cache-dir
# or pip install fix_yahoo_finance --upgrade
print("+++++pandas_datareader+++++")
from pandas_datareader import data as pdr
#import fix_yahoo_finance as yf # <== that's all it takes :-)
# For Stock of Thailand, the symbol must be followed with '.BK'
# download dataframe
try:
    ptt = pdr.get_data_yahoo("PTT.BK", start="2017-01-01", end="2017-04-30")
    print(ptt.tail())
    print()
except:
    print("Error:", sys.exc_info()[0])
    print("Description:", sys.exc_info()[1])

# Forth example 
# How to install yahoo_finance package
# Reference: https://pypi.python.org/pypi/yahoo-finance
# pip install yahoo_finance
from yahoo_finance import Share
try:
    yahoo = Share("YHOO")
    print("\n++++YHOO stock++++")
    print (yahoo.get_open())
    print (yahoo.get_price())
    print (yahoo.get_trade_datetime())
except:
    print("Error:", sys.exc_info()[0])
    print("Description:", sys.exc_info()[1])

from yahoo_finance import Currency
try:
    eur_pln = Currency('EURPLN')
    print("\n++++Currency of EURPLN++++")
    print (eur_pln.get_bid())
    print (eur_pln.get_ask())
    print (eur_pln.get_rate())
    print (eur_pln.get_trade_datetime())
except:
    print("Error:", sys.exc_info()[0])
    print("Description:", sys.exc_info()[1])