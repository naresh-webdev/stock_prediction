import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import pandas_datareader.data as data
import datetime
import yfinance as yf
from keras.models import load_model
import streamlit as st


start = datetime.datetime(2013, 1, 1)
end = datetime.datetime(2023, 2, 11)

st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')

df = yf.download(user_input,start, end)
df = df.reset_index()
df = df.drop(['Date','Adj Close'], axis = 1)

#Describing Data
st.subheader('Data from 2013 - 2023')
st.write(df.describe())

#visualisations
st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize= (12,6))
plt.plot(df.Close, 'royalblue')
st.pyplot(fig)


#MOVING AVG
st.subheader('Closing Price vs Time chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize= (12,6))
plt.plot(ma100, 'g')
plt.plot(df.Close, 'royalblue')
st.pyplot(fig)


st.subheader('Closing Price vs Time chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize= (12,6))
plt.plot(ma100 ,'g')
plt.plot(ma200, 'r')
plt.plot(df.Close, 'royalblue')
st.pyplot(fig)


#splitting data into training and testing

data_training = pd.DataFrame(df['Close'][0:int(len(df)*.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*.70):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range= (0,1))
data_training_array = scaler.fit_transform(data_training)


#Load my model
model = load_model('keras_model.h5')

#Testing part
past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index = True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test, y_test = np.array(x_test), np.array(y_test) 

#making predictions

y_predicted = model.predict(x_test)
scaler = scaler.scale_

scale_factor = 1 / scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

#final graph
st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)




# # own customisation
# print(len(y_predicted))
# print(len(y_test))
if y_predicted[-1] > y_predicted[-2]:
  st.subheader('THE STOCK MAY GO UP  💹')
else:
  st.subheader('THE STOCK MAY GO DOWN 📈')
