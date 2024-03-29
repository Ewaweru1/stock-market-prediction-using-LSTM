import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from pandas_datareader import data as pdr
from datetime import date
import streamlit as st
from keras.models import load_model
import pymongo

# MongoDB connection
mongo_password = "z91oQGHUdHt2d5VH"
mongo_username = "ericwaweruewm"
mongo_uri = f"mongodb+srv://{mongo_username}:{mongo_password}@lstm.rdo1doh.mongodb.net/"
client = pymongo.MongoClient(mongo_uri)

# Database
db = client['LSTM']
collection = db['predictions']

# Function to insert data into MongoDB
def insert_data(data):
    collection.insert_one(data)

start = "2010-01-01"
today = date.today().strftime("%Y-%m-%d")

st.title("Money Market Prediction App")

yf.pdr_override()
user_input = st.text_input('Enter stock Ticker'  , 'AAPL')
df = pdr.get_data_yahoo(user_input, start, today)

# Describe data
st.subheader('Data from 2010 - Date')
st.write(df.describe())

# Visualizations
st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

# Visualizations with 40MA
st.subheader('Closing Price vs Time chart with  40 & 10 Moving Average')
ma40 = df.Close.rolling(40).mean()
ma10 = df.Close.rolling(10).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma40, 'r')
plt.plot(ma10,  'g')
plt.plot(df.Close  , 'b')
st.pyplot(fig)

# Splitting Data into Training and Testing
data_training = pd.DataFrame(df['Close'][0:int(.7*len(df))])
data_testing = pd.DataFrame(df['Close'][int(.7*len(df)): int(len(df))])

# Scaling data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

data_training_array = scaler.fit_transform(data_training)

# Load model
model = load_model('LSTM_model.keras')

# Testing
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)

# Scaling
input_data = scaler.fit_transform(final_df)

# Convert input_data to a list for easier appending
input_data_list = input_data.tolist()

x_test = []
y_test = []

for i in range(100, len(input_data_list)):
    x_test.append(input_data_list[i-100: i])
    y_test.append(input_data_list[i])

# Convert back to numpy arrays
x_test, y_test = np.array(x_test), np.array(y_test)

# Making predictions
y_predicted = model.predict(x_test)

# To bring back to original units
scale_factor = 1/scaler.scale_[0] 
y_predicted = scale_factor * y_predicted
y_test =  y_test * scale_factor

# Plot predicted and original values
st.subheader('Predictions vs original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label="Predicted Price")
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

# Inserting data into MongoDB
insert_data({
    "stock_ticker": user_input,
    "original_prices": y_test.tolist(),
    "predicted_prices": y_predicted.tolist()
})

# Download button
def download_button(filename, label):
    with open(filename, 'rb') as f:
        data = f.read()
    st.download_button(label, data, file_name=filename)

# Add download button for the generated image
download_button("predicted_vs_original_prices.png", "Download Predicted vs Original Prices Plot")
