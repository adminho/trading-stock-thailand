# Trading example codes (not yet)

This repository collects my example codes in Python for studing in [The Stock Exchange of Thailand (SET)](http://www.set.or.th/set/mainpage.do) that is developing (__not yet__)

## Requirement

All examples are written in Python language, so you need to setup your environments as below 

* First, install [ANACONDA](https://www.continuum.io/downloads)
* For Deep learning, I used 2 library including TensorFlow and Keras

You can install TensorFlow from PyPI with the command

`pip install tensorflow`

And you can also install Keras from PyPI with the command

`pip install keras`

* Install pandas_datareader to get data from [https://finance.yahoo.com/](https://finance.yahoo.com/) and also install fix_yahoo_finance

`pip install pandas_datareader`

`pip install fix_yahoo_finance`

* Install tqdm to make my loops show a smart progress meter on console

`pip install tqdm`


Download datasets (EOD data files from SET)

* http://siamchart.com/stock/


## My source codes


### 1) [LSTM_predict_trend.py](LSTM_predict_trend.py) 

I'm trying to apply Deep Learning (LSTM network) to predict a stock trend (not complete)


### 2) [BOT_API_example.py](BOT_API_example.py)

Since Bank of Thailand (BOT) has provided [21 APIs](https://iapi.bot.or.th/Developer?lang=th) for query data including Exchange rate, Interest Rate and Debt securities auction so I would like to show examples howto use 2 APIs such as
- Daily Weighted-average Interbank Exchange Rate - THB / USD
- Daily Average Exchange Rate - THB / Foreign Currency

For example code in HTML/JavaScript, I shared at [here](https://gist.github.com/adminho/0159bb53c02bfdee1c4c31de3d8ecd92)


### 3) [indicator.py](indicator.py) 

There are my examples to compute the technical indicators for securities including

* [ROC](http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:rate_of_change_roc_and_momentum)
* [Bollinger Band (BBANDS)](http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:bollinger_bands)
* daily returns
* [SMA and EMA](http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:moving_averages)
* [MACD and signal](http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:moving_average_convergence_divergence_macd)
* [RSI](http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:relative_strength_index_rsi)
* [Sharpe ratio](http://www.investopedia.com/terms/s/sharperatio.asp?lgl=rira-baseline-vertical)
* [True Range (TR) and ATR](http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:average_true_range_atr)
* [Beta](http://www.investopedia.com/terms/b/beta.asp?lgl=rira-baseline-vertical)
* [K% and D%](http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:stochastic_oscillator_fast_slow_and_full)
* [OBV](http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:on_balance_volume_obv)
* compute gain 
* and etc


### 4) [DeepQ_trade.py](DeepQ_trade.py)

I'm trying to apply Deep Q-learning (Reinforcement Learning) to automatic trading (not complete)


## Thank you

I Borrowed some codes from

* https://www.udacity.com/course/machine-learning-for-trading--ud501
* http://matplotlib.org/examples/pylab_examples/finance_demo.

