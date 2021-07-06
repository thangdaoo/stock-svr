import numpy as np
import pandas as pd
import os
import time
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit
from sklearn import metrics
from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf

# ticker = pd.read_csv("./datasets/AAPL.csv")
# ticker = pd.DataFrame(ticker)
# ticker = ticker.drop(columns=['Open','High','Low','Close','Volume'])
# date = ticker['Date']
# date = pd.to_datetime(ticker.Date)
# close = ticker['Adj Close']

# X_train, X_val, y_train, y_val = [], [], [], []

# tscv = TimeSeriesSplit(n_splits=5)
# for train_index, val_index in tscv.split(date):
#     X_train.append(date[train_index])
#     X_val.append(date[val_index])
#     y_train.append(close[train_index])
#     y_val.append(close[val_index])

# Xtrain, ytrain, Xval, yval = np.array(X_train[0]), np.array(y_train[0]), np.array(X_val[0]), np.array(y_val[0])

# lag = 180
# x = np.zeros((len(ytrain) - lag + 1,lag))
# rowNumber = 0

# for i in range(lag,len(ytrain)+1):
#     x[rowNumber,] = ytrain[(i-lag):(i)]
#     rowNumber += 1
# x = x[:-1]
# y = ytrain[179:-1]
# y_val = ytrain[180:]
# x_val = Xtrain[180:]

# linear_svr = SVR(kernel='linear', C=1e10)
# rbf_svr = SVR(kernel='rbf', C=1e10, gamma=0.1)
# y_lin = linear_svr.fit(x, y).predict(x)
# y_rbf = rbf_svr.fit(x, y).predict(x)

# rmse_lin = metrics.mean_squared_error(y, y_lin, squared=False)
# rmse_rbf = metrics.mean_squared_error(y, y_rbf, squared=False)
# mape_lin = metrics.mean_absolute_percentage_error(y, y_lin)
# mape_rbf = metrics.mean_absolute_percentage_error(y, y_rbf)

# plt.plot(x_val, y, color='red', label='Data')
# plt.plot(x_val, y_lin, color='green', linestyle='dotted', label='Linear Model')
# plt.plot(x_val, y_rbf, color='blue', linestyle='dotted', label='RBF Model')
# plt.legend(loc="upper left")
# plt.title("AAPL")
# plt.show()


directory = r'./datasets/'
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        ticker = filename.split(".")
        ticker = ticker[0]
        print()
        print(ticker)
        ticker_df = pd.read_csv(os.path.join(directory, filename))
        date = ticker_df['Date']
        date = pd.to_datetime(ticker_df.Date)
        close = ticker_df['Adj Close']

        X_train, X_val, y_train, y_val = [], [], [], []

        tscv = TimeSeriesSplit(n_splits=5)
        for train_index, val_index in tscv.split(date):
            X_train.append(date[train_index])
            X_val.append(date[val_index])
            y_train.append(close[train_index])
            y_val.append(close[val_index])

        plot = 1
        index = 1

        # plot_pacf(close, lags=90)
        # plt.title(ticker)
        # plt.xlabel('Lag')
        # plt.ylabel('Partial Autocorrelation')
        # plot += 1
        # plt.show()

        for split in range(5):
            start = time.time()
            Xtrain, ytrain, Xval, yval = np.array(X_train[split]), np.array(y_train[split]), np.array(X_val[split]), np.array(y_val[split])

            lag = 180
            x = np.zeros((len(ytrain) - lag + 1,lag))
            rowNumber = 0

            for i in range(lag,len(ytrain)+1):
                x[rowNumber,] = ytrain[(i-lag):(i)]
                rowNumber += 1
            x = x[:-1]
            y = ytrain[179:-1]
            y_val = ytrain[180:]
            x_val = Xtrain[180:]

            linear_svr = SVR(kernel='linear', C=1e10)
            rbf_svr = SVR(kernel='rbf', C=1e10, gamma=0.1)
            y_lin = linear_svr.fit(x, y).predict(x)
            y_rbf = rbf_svr.fit(x, y).predict(x)

            rmse_lin = metrics.mean_squared_error(y_val, y_lin, squared=False)
            rmse_rbf = metrics.mean_squared_error(y_val, y_rbf, squared=False)
            print('lin rmse:',rmse_lin)
            print('rbf rmse:',rmse_rbf)
            print()

            mape_lin = metrics.mean_absolute_percentage_error(y_val, y_lin)
            mape_rbf = metrics.mean_absolute_percentage_error(y_val, y_rbf)
            print('lin mape:',mape_lin)
            print('rbf mape:',mape_rbf)
            end = time.time()
            print()
            print('Time Elapsed:',end - start)
            print('--------------Split ' + str(index) + '-----------------')
            index += 1

            fig, ax = plt.subplots()
            # ax.axline([0,0],[1,1], color='black')
            # plt.scatter(y, y, color='red', label='Data')
            # plt.scatter(y, y_lin, color='green', label='Linear Model')
            # plt.scatter(y, y_rbf, color='blue', label='RBF Model')

            plt.plot(x_val, y, color='red', label='Data')
            plt.plot(x_val, y_lin, color='green', linestyle='dotted', label='Linear Model')
            plt.plot(x_val, y_rbf, color='blue', linestyle='dotted', label='RBF Model')

            plt.title(ticker + " " + str(plot))
            plt.legend(loc="upper left")
            plot += 1
            plt.show()
