import matplotlib
import numpy as np
import datetime as dt
import pandas_datareader as pdr
import math
import functions as fn
import matplotlib.pyplot as plt
import csv

"""Importing Point Process Data Set"""
fig = plt.figure()
stock_chart = fig.add_subplot(111)
start = dt.datetime(2001, 1, 1)
end = dt.datetime(2017, 1, 1)  # Manually set end of range
# end = dt.datetime.now().date()
present = dt.datetime.now()

print(start)
print(present)

apple = pdr.DataReader("AAPL", 'yahoo', start, end)  # Take not of the capitalization in DataReader
# google = pdr.DataReader("GOOGL", 'yahoo', start, end)
plt.plot(apple["Adj Close"], c='b')  # Plotting stock price wrt to time
# plt.plot(google["Adj Close"], c='r')
stock_chart.set_title("Apple Stock Price")
plt.axis([start, end, 0, 200])
plt.show()
