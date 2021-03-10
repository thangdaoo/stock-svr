import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# shows all plots on on chart
directory = r'./datasets/'
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        ticker = filename.split(".")
        ticker = ticker[0]
        print(ticker)
        ticker_df = pd.read_csv(os.path.join(directory, filename))
        x = ticker_df['Date']
        y = ticker_df['Adj Close']
        plt.plot(x, y, label=ticker)
        plt.legend(loc="upper left")
        plt.xlabel('Date')
        plt.ylabel('Adj Close')

plt.show()

# to show each individual stock move plt.show() to the end of the loop
# however, only one ticker can be shown at a time
