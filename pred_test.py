import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import yfinance as yf
import logging
import sys

stock_ticker = sys.argv[1]
start_date = sys.argv[2]
end_date = sys.argv[3]

logging.basicConfig(filename="std.log", format='%(message)s', filemode='w') 
logger=logging.getLogger() 
logger.setLevel(logging.INFO) 
logger.info("LSTM Stock Prediction Tool\n\n") 

stock_data = yf.download(tickers=stock_ticker, start=start_date, end=end_date)
stock_data = stock_data.iloc[:,0:4]
logger.info('Stock Data for %s from %s to %s\n', stock_ticker, start_date, end_date)
logger.info('{}\n\n'.format(stock_data))

#Feature Scaling
scaler = StandardScaler()
scaled_data = scaler.fit_transform(stock_data)
scaled_data = pd.DataFrame(columns=stock_data.columns, data=scaled_data, index=stock_data.index)
logger.info("SCALED DATA")
logger.info(scaled_data)

#Data Split
def lstm_split(data, n_steps):
    x, y = [], []
    for i in range(len(data)-n_steps+1):
        x.append(data[i:i+n_steps, :-1])
        y.append(data[i+n_steps-1, -1])
    return np.array(x), np.array(y)

#Train and Test
x1, y1 = lstm_split(scaled_data.values, n_steps=2)
# logger.info("X LSTM")
# logger.info(x1)
# logger.info("Y LSTM")
# logger.info(y1)
train_split = 0.9
split_idx = int(np.ceil(len(x1)*train_split))
date_index = scaled_data.index

x_train, x_test = x1[:split_idx], x1[split_idx:-1]
y_train, y_test = y1[1:split_idx+1], y1[split_idx+1:]
x_train_date, x_test_date = date_index[:split_idx+2], date_index[split_idx+2:]
logger.info("X TRAIN")
logger.info(x_train)
logger.info("X TEST")
logger.info(x_test)
logger.info("Y TRAIN")
logger.info(y_train)
logger.info("Y TEST")
logger.info(y_test)

#Build LSTM Model
lstm = Sequential()
lstm.add(LSTM(50, input_shape=(x_train.shape[1], x_train.shape[2]), activation='relu', return_sequences=True))
lstm.add(LSTM(50, activation='relu'))
lstm.add(Dense(1))
lstm.compile(loss='mean_squared_error', optimizer='adam')
lstm.summary()

#Fit Model to Training Data
history = lstm.fit(x_train, y_train, epochs=100, batch_size=4, verbose=2, shuffle=False)
y_pred = lstm.predict(x_test)
for i in range(103):
    print(y_test[i], ' ' , y_pred[i])

diff_test, diff_pred = [],[]
for i in range(1, 103):
    diff_test.append(y_test[i] - y_test[i-1])
    diff_pred.append(y_pred[i][0] - y_pred[i-1][0])

for i in range(102):
    print(diff_test[i], ' ', diff_pred[i])

count = 0
for i in range(102):
    if diff_test[i]*diff_pred[i]>0: 
        print("✅")
        count = count+1
    else: print("❌")

print("FINAL STATS: ", count/102)

rmse = mean_squared_error(y_test, y_pred, squared=False)
mape = mean_absolute_percentage_error(y_test, y_pred)
logger.info('RESULTS\n')
logger.info('---------------------------')
logger.info('| LSTM RMSE: %f', rmse)
logger.info('| LSTM MAPE: %f', mape)
logger.info('---------------------------')

plt.figure(figsize=(10, 5))
plt.plot(x_test_date, y_test, label='True Values', color='blue')
plt.plot(x_test_date, y_pred, label='Predictions', color='red')

# Add labels and title
plt.xlabel('Date', fontsize=15)
plt.ylabel('Stock Price', fontsize=15)
plt.title('True vs. Predicted Stock Prices', fontsize=18)
# Add legend
plt.legend()
plt.show()