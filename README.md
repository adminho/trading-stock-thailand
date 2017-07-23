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

`ROC, Bollinger Band (BBANDS), daily returns, SMA, EMA, MACD, RSI, Sharpe ratio,True Range (TR), ATR, Beta, OBV and etc`


### 4) [DeepQ_trade.py](DeepQ_trade.py)

I'm trying to apply Deep Q-learning (Reinforcement Learning) to automatic trading (not complete)


## Credit 

I Borrowed some codes from

* https://www.udacity.com/course/machine-learning-for-trading--ud501
* http://matplotlib.org/examples/pylab_examples/finance_demo.

