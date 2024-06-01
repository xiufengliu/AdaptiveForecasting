import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from keras.metrics import RootMeanSquaredError
from statsmodels.tsa.arima.model import ARIMA
from gan_model import GANModel
from dataprocessor import DataPreprocessor

# Load and preprocess data
data = pd.read_csv('energy_data.csv')
preprocessor = DataPreprocessor(data)
data = preprocessor.preprocess_data()

# Split data into training and testing sets
train_data = data[:int(len(data)*0.8)]
test_data = data[int(len(data)*0.8):]

# Scale data
scaler = MinMaxScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

# Define input shape for GAN model
input_shape = (train_data.shape[1],)

# Train GAN model
gan_model = GANModel(input_shape)
gan_model.train_gan_model(train_data, num_epochs=100, batch_size=32)

# Evaluate GAN model
y_pred = gan_model.generator.predict(test_data)
y_true = test_data[:, -1]
mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
print(f'GAN model - MAE: {mae}, MSE: {mse}, RMSE: {rmse}')

# Train baseline models
lstm_model = Sequential()
lstm_model.add(LSTM(50, activation='relu', input_shape=input_shape))
lstm_model.add(Dense(1))
lstm_model.compile(loss=MeanSquaredError(), optimizer=Adam(), metrics=[RootMeanSquaredError()])
lstm_model.fit(train_data, train_data[:, -1], epochs=100, batch_size=32, verbose=0)

arima_model = ARIMA(train_data[:, -1], order=(5,1,0))
arima_model_fit = arima_model.fit(disp=0)

# Evaluate baseline models
y_pred_lstm = lstm_model.predict(test_data)
mae_lstm = mean_absolute_error(y_true, y_pred_lstm)
mse_lstm = mean_squared_error(y_true, y_pred_lstm)
rmse_lstm = np.sqrt(mse_lstm)
print(f'LSTM model - MAE: {mae_lstm}, MSE: {mse_lstm}, RMSE: {rmse_lstm}')

y_pred_arima = arima_model_fit.predict(start=len(train_data), end=len(data)-1)
mae_arima = mean_absolute_error(y_true, y_pred_arima)
mse_arima = mean_squared_error(y_true, y_pred_arima)
rmse_arima = np.sqrt(mse_arima)
print(f'ARIMA model - MAE: {mae_arima}, MSE: {mse_arima}, RMSE: {rmse_arima}')
