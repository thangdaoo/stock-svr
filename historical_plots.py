import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# shows all plots on on chart
directory = r'./datasets/'
index = 1
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        ticker = filename.split(".")
        ticker = ticker[0]
        print(ticker)
        ticker_df = pd.read_csv(os.path.join(directory, filename))
        x = ticker_df['Date']
        y = ticker_df['Adj Close']
        plt.figure(index)
        plt.plot(x, y, label=ticker)
        plt.legend(loc="upper left")
        plt.xlabel('Date')
        plt.ylabel('Adj Close ($)')
        index += 1

plt.show()