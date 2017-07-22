# Trading example codes (not yet)

This repository collects my example codes in Python for studing in [The Stock Exchange of Thailand (SET)](http://www.set.or.th/set/mainpage.do) that is developing (__not yet__)

In [indicator.py](indicator.py) file, there are my examples to compute the technical indicators for securities including

`ROC, Bollinger Band (BBANDS), daily returns, SMA, EMA, MACD, RSI, Sharpe ratio,True Range (TR), ATR, Beta, OBV and etc`

I Borrowed some code from

* https://www.udacity.com/course/machine-learning-for-trading--ud501
* http://matplotlib.org/examples/pylab_examples/finance_demo.

Download datasets (EOD data files from SET)

* http://siamchart.com/stock/

## Requirement

All examples are written in Python language, so you need to setup your environments as below 

* First, install [ANACONDA](https://www.continuum.io/downloads)
* For Deep learning, I used 2 library including TensorFlow and Keras

You can install TensorFlow from PyPI with the command

`pip install tensorflow`

And you can also install Keras from PyPI with the command

`pip install keras`

* Install tqdm to make my loops show a smart progress meter 

`pip install tqdm`

## My source codes

In [LSTM_predict_trend.py](LSTM_predict_trend.py) file, I'm trying to apply Deep Learning (LSTM network) to predict a stock trend (Not complete)
