import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

#  Load Data
company = 'AAPL'

start = dt.datetime(2017, 1, 1)
end = dt.datetime(2021, 1, 1)

data = web.DataReader(company, 'yahoo', start, end)
open_data = data['Open'].values
close_data = data['Close'].values
change_data = []
scalar = 0
for day, value in enumerate(open_data):
    if day != 0:
        change = open_data[day]-close_data[day-1]
        change_data.append(change)
        if abs(change) > scalar:
            scalar = abs(change)

for i in range(len(change_data)):
    change_data[i] = change_data[i] / scalar

# Prepare Data
#  scalar = MinMaxScaler(feature_range=(0, 1))
#  scaled_data = scalar.fit_transform(data['Open'].values.reshape(-1, 1))

prediction_days = 60

x_train = []
y_train = []

for x in range(prediction_days, len(change_data)):
    x_train.append(change_data[x-prediction_days:x])
    y_train.append(change_data[x])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Build the model
model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))  # Prediction of the next overnight swing

model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(x_train, y_train, epochs=25, batch_size=32)

''' Test the Model Accuracy on Existing Data'''

# Load Test Data
test_start = dt.datetime(2021, 1, 1)
test_end = dt.datetime.now()

test_data = web.DataReader(company, 'yahoo', test_start, test_end)
open_data = test_data['Open'].values
close_data = test_data['Close'].values
change_data2 = []
for day, value in enumerate(open_data):
    if day != 0:
        change = open_data[day]-close_data[day-1]
        change_data2.append(change)

for i in range(len(change_data2)):
    change_data2[i] = change_data2[i] / scalar


actual_prices = test_data['Open'].values
new_prices = []
for day, price in enumerate(actual_prices):
    if day != 0:
        new_prices.append(price-actual_prices[day-1])
actual_prices = new_prices

total_dataset = change_data + change_data2


model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:]

for i in range(len(model_inputs)):
    model_inputs[i] = model_inputs[i] / scalar

# Make Predictions on Test Data

x_test = []

for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x-prediction_days:x])  # change to be overnight swing

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_prices = model.predict(x_test)
# predicted_prices = scalar.inverse_transform(predicted_prices)
for i in range(len(predicted_prices)):
    predicted_prices[i] = predicted_prices[i] * scalar
print(predicted_prices)

# Plot the Test Predictions
plt.plot(actual_prices, color="black", label=f"Actual {company} Price")
plt.plot(predicted_prices, color="green", label=f"Predicted {company} Price")
plt.title(f"{company} Share Price")
plt.xlabel('Time')
plt.ylabel(f'{company} Share Price')
plt.legend()
plt.show()
