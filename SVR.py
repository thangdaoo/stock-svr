import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit


# pd.set_option('display.max_rows', None)
aapl = pd.read_csv("./datasets/AAPL.csv")
aapl_df = pd.DataFrame(aapl)
aapl_df['Date'] = pd.to_datetime(aapl_df.Date)
X = aapl_df['Date']
y = aapl_df['Adj Close']

y_predictions = []
X_train = []
X_val = []
y_train = []
y_val = []
tscv = TimeSeriesSplit(n_splits=5, test_size=250)
for train_index, val_index in tscv.split(X):
    # print("TRAIN:", X[train_index])
    # print("TEST:", X[test_index])
    # print('SPLIT:', split)
    # print('-------------------------------------')
    X_train.append(X[train_index])
    X_val.append(X[val_index])
    y_train.append(y[train_index])
    y_val.append(y[val_index])
    # Xtrain_arr = X_train.values
    # ytrain_arr = y_train.values
    # ytest_arr = y_val.values
    # reshaped_Xtrain = Xtrain_arr.reshape(-1,1)
    # reshaped_ytrain = ytrain_arr.reshape(-1,1)
    # reshaped_ytest = ytest_arr.reshape(-1,1)
    # SVRModel = SVR(kernel='rbf')
    # SVRModel.fit(reshaped_Xtrain, reshaped_ytrain.ravel())
    # y_pred = SVRModel.predict(reshaped_ytest.ravel())
    # y_predictions.append(y_pred)
    # reg = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
    # reg.fit(X_train, y_train)

aapl_Xtrain, aapl_Ytrain = X_train[0], y_train[0]
aapl_Xtrain, aapl_Ytrain = np.array(aapl_Xtrain), np.array(aapl_Ytrain)
print(aapl_Xtrain)
linear_svr = SVR(kernel='linear', C=1000.0)
# linear_svr.fit(aapl_Xtrain, aapl_Ytrain)

