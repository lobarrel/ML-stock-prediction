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


stock_data = pd.read_csv('./AAPL.csv', index_col='Date')
stock_data = stock_data.iloc[:,0:4]

#Feature Scaling
scaler = StandardScaler()
scaled_data = scaler.fit_transform(stock_data)
scaled_data = pd.DataFrame(columns=stock_data.columns, data=scaled_data, index=stock_data.index)

#Data Split
def lstm_split(data, n_steps):
    x, y = [], []
    for i in range(len(data)-n_steps+1):
        x.append(data[i:i+n_steps, :-1])
        y.append(data[i+n_steps-1, -1])
    return np.array(x), np.array(y)

#Train and Test
x1, y1 = lstm_split(scaled_data.values, n_steps=2)
train_split = 0.8
split_idx = int(np.ceil(len(x1)*train_split))
date_index = scaled_data.index

x_train, x_test = x1[:split_idx], x1[split_idx:]
y_train, y_test = y1[:split_idx], y1[split_idx:]
x_train_date, x_test_date = date_index[:split_idx+1], date_index[split_idx+1:]

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

rmse = mean_squared_error(y_test, y_pred, squared=False)
mape = mean_absolute_percentage_error(y_test, y_pred)
print('LSTM RMSE: ', rmse)
print('LSTM MAPE: ', mape)

#Simple Moving Average
train_split = 0.8
split_idx = int(np.ceil(len(scaled_data)*train_split))
test = scaled_data[['Close']].iloc[split_idx:]
sma_50 = scaled_data[['Close']].rolling(50).mean().iloc[split_idx:]
sma_200 = scaled_data[['Close']].rolling(200).mean().iloc[split_idx:]

rmse = mean_squared_error(test, sma_50, squared=False)
mape = mean_absolute_percentage_error(test, sma_50)
print('SMA_50 RMSE: ', rmse)
print('SMA_50 MAPE: ', mape)

rmse = mean_squared_error(test, sma_200, squared=False)
mape = mean_absolute_percentage_error(test, sma_200)
print('SMA_200 RMSE: ', rmse)
print('SMA_200 MAPE: ', mape)

#Plot Results
plt.figure(figsize=(10, 5))
plt.plot(x_test_date, y_test, label='True Values', color='blue')
plt.plot(x_test_date, y_pred, label='Predictions', color='red')
plt.plot(sma_50, label='SMA 50')
plt.plot(sma_200, label='SMA 200')
# Add labels and title
plt.xlabel('Date', fontsize=15)
plt.ylabel('Stock Price', fontsize=15)
plt.title('True vs. Predicted Stock Prices', fontsize=18)
# Add legend
plt.legend()
plt.show()

