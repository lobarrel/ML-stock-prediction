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

plt.figure(figsize=(13,8))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=60))
x_dates = [dt.datetime.strptime(d,'%Y-%m-%d').date() for d in stock_data.index.values]

plt.plot(x_dates, stock_data['High'], label='High')
plt.plot(x_dates, stock_data['Low'], label='Low')
plt.legend()
plt.gcf().autofmt_xdate()
#plt.show()

target_y = stock_data['Close']
x_feat = stock_data.iloc[:,0:3]

#Feature Scaling
sc = StandardScaler()
x_ft = sc.fit_transform(x_feat.values)
x_ft = pd.DataFrame(columns=x_feat.columns, data=x_ft, index=x_feat.index)

#Data Split
def lstm_split(data, n_steps):
    x, y = [], []
    for i in range(len(data)-n_steps+1):
        x.append(data[i:i+n_steps, :-1])
        y.append(data[i+n_steps-1, -1])
    return np.array(x), np.array(y)

#Train and Test
x1, y1 = lstm_split(x_ft.values, n_steps=2)
train_split = 0.8
split_idx = int(np.ceil(len(x1)*train_split))
date_index = x_ft.index

x_train, x_test = x1[:split_idx], x1[split_idx:]
y_train, y_test = y1[:split_idx], y1[split_idx:]
x_train_date, x_test_date = date_index[:split_idx], date_index[split_idx:]
print(x1.shape, x_train.shape, x_test.shape, y_test.shape)

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
print("RMSE: ", rmse)
print("MAPE: ", mape)

# Assuming you have y_pred (predicted stock prices) from your LSTM model
plt.figure(figsize=(10, 6))  # Adjust the figure size as needed

# Plot true stock values (y_test) in blue
plt.plot(x_test_date, y_test, label='True Values', color='blue')

# Plot predicted stock values (y_pred) in red
plt.plot(x_test_date, y_pred, label='Predictions', color='red')

# Add labels and title
plt.xlabel('Date', fontsize=15)
plt.ylabel('Stock Price', fontsize=15)
plt.title('True vs. Predicted Stock Prices', fontsize=18)

# Add legend
plt.legend()

# Show the plot
plt.show()