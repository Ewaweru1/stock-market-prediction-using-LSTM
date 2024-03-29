#%%
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from pandas_datareader import data as pdr
from datetime import date

start = "2010-01-01"
today = date.today().strftime("%Y-%m-%d")

yf.pdr_override()
df = pdr.get_data_yahoo('AAPL', start, today)
print(df.head())

print(df.tail())

df = df.reset_index()
print(df.head())

df = df.drop(['Date','Adj Close'], axis= 1)
print(df.head())

print(plt.plot(df.Close))
print(plt.show)

df

# Ensure 'Close' column is numeric
df['Close'] = pd.to_numeric(df['Close'], errors='coerce')

# Ensure dataframes are not empty and do not contain NaN values
assert not df['Close'].isnull().any(), "DataFrame contains NaN values"

#moving averages
ma40 = df.Close.rolling(40).mean()
ma40

ma10 = df.Close.rolling(10).mean()
ma10

plt.figure(figsize= (12,6))
plt.plot(df.Close)
plt.plot(ma40, color='red')
plt.plot(ma10, color='yellow')

print(df.shape)

#splitting Data into Training and Testing
data_training = pd.DataFrame(df['Close'][0:int(.7*len(df))])
data_testing = pd.DataFrame(df['Close'][int(.7*len(df)): int(len(df))])
                            
print(data_training.shape)
print(data_testing.shape)

data_training.head
data_testing.head

#scaling data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0,1))

data_training_array = scaler.fit_transform(data_training)
data_training_array
data_training_array.shape

#divide data into x train and y train
x_train = []
y_train = []


for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i - 100:i])
    y_train.append(data_training_array[i, 0])
    
	#convert to numpy arrays 
x_train, y_train = np.array(x_train), np.array(y_train)
print(x_train.shape)

#LSTM MODEL
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from keras.layers import Input

model = Sequential()
model.add(Input(shape=(x_train.shape[1], 1)))
model.add(LSTM(units = 50, activation = 'relu', return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(units = 60, activation = 'relu', return_sequences = True ))
model.add(Dropout(0.3))

model.add(LSTM(units = 80, activation = 'relu', return_sequences = True))
model.add(Dropout(0.4))

model.add(LSTM(units = 120, activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(units = 1))


model.summary()

#compile model
model.compile(optimizer = 'Adam', loss = 'mean_squared_error')
model.fit(x_train, y_train, epochs = 50)

#save model
model.save('LSTM_model.keras')

data_testing.head()
data_testing.tail(100)

past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
final_df.head()


#scaling
input_data = scaler.fit_transform(final_df)
print(input_data.shape)

# Convert input_data to a list for easier appending
input_data_list = input_data.tolist()

x_test = []
y_test = []

for i in range(100, len(input_data_list)):
    x_test.append(input_data_list[i-100: i])
    y_test.append(input_data_list[i])

# Convert back to numpy arrays
x_test, y_test = np.array(x_test), np.array(y_test)
print(x_test.shape)
print(y_test.shape)
    
#making predictions
y_predicted = model.predict(x_test)
y_predicted.shape

print(y_test)
print(y_predicted)

print("Scale factor:", scaler.scale_)

#to bring back to original units
scale_factor = 1/scaler.scale_[0] 
y_predicted = scale_factor * y_predicted
y_test =  y_test *scale_factor

#plot predicted and original values
plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'g', label="Predicted Price")
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()