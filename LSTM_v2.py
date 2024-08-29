import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import math
import yfinance as yf
import time
import os
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# Fetch historical data from Yahoo Finance
def get_historical_data_yahoo(symbol, start_date, end_date, interval='1m'):
    df = yf.download(symbol, start=start_date, end=end_date, interval=interval)
    df = df[['Close']]
    df.rename(columns={'Close': 'close'}, inplace=True)
    return df

def fetch_data_in_chunks(symbol, start_date, end_date, interval='1m', chunk_days=7):
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    delta = timedelta(days=chunk_days)
    
    all_data = pd.DataFrame()
    
    while start < end:
        chunk_end = min(start + delta, end)
        chunk_data = get_historical_data_yahoo(symbol, start.strftime('%Y-%m-%d'), chunk_end.strftime('%Y-%m-%d'), interval)
        all_data = pd.concat([all_data, chunk_data])
        start = chunk_end
    
    return all_data

def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

def create_dataset(data, time_step=60):
    X, Y = [], []
    for i in range(len(data) - time_step):
        a = data[i:(i + time_step), 0]
        X.append(a)
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

def split_data(data, train_ratio=0.65):
    train_size = int(len(data) * train_ratio)
    train_data = data[:train_size]
    test_data = data[train_size:]
    return train_data, test_data

def train_model(X, Y):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        Dropout(0.15),  # Dropout layer for regularization
        LSTM(50, return_sequences=False),
        Dropout(0.15),  # Dropout layer for regularization
        Dense(25),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
    
    model.fit(X, Y, 
              validation_split=0.1,  # Split data for validation
              batch_size=32, 
              epochs=50, 
              callbacks=[early_stopping], 
              verbose=1)
    
    return model

def evaluate_and_save_model(model, X_train, Y_train, X_test, Y_test, scaler):
    # Predictions on training and testing data
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    # Inverse transform predictions and actual values
    train_predict_inverse = scaler.inverse_transform(train_predict)
    test_predict_inverse = scaler.inverse_transform(test_predict)
    Y_train_inverse = scaler.inverse_transform(Y_train.reshape(-1, 1))
    Y_test_inverse = scaler.inverse_transform(Y_test.reshape(-1, 1))

    # Calculate RMSE on the original scale
    train_rmse = math.sqrt(mean_squared_error(Y_train_inverse, train_predict_inverse))
    test_rmse = math.sqrt(mean_squared_error(Y_test_inverse, test_predict_inverse))

    # Calculate MAE on the original scale
    train_mae = mean_absolute_error(Y_train_inverse, train_predict_inverse)
    test_mae = mean_absolute_error(Y_test_inverse, test_predict_inverse)

    # Calculate MAPE on the original scale
    train_mape = np.mean(np.abs((Y_train_inverse - train_predict_inverse) / Y_train_inverse)) * 100
    test_mape = np.mean(np.abs((Y_test_inverse - test_predict_inverse) / Y_test_inverse)) * 100

    # Calculate R² on the original scale
    train_r2 = r2_score(Y_train_inverse, train_predict_inverse)
    test_r2 = r2_score(Y_test_inverse, test_predict_inverse)

    print(f"Train RMSE: {train_rmse}")
    print(f"Test RMSE:  {test_rmse}\n")
    print(f"Train MAE: {train_mae}")
    print(f"Test MAE:  {test_mae}\n")
    print(f"Train MAPE: {train_mape}")
    print(f"Test MAPE:  {test_mape}\n")
    print(f"Train R-squared: {train_r2}")
    print(f"Test R-squared:  {test_r2}\n")

    # Save the model only if it meets the criteria
    if ((test_rmse < 90 and test_mae < train_mae and test_mape < train_mape and test_rmse < train_rmse and test_r2 > 0.997) or
        (test_rmse < 70 and test_mae < 50 and test_mape < 0.07 and test_r2 > 0.997)):
        model.save('bitcoin_lstm_model.h5')
        print("Model saved successfully.\n")
        return True, train_predict_inverse, test_predict_inverse
    else:
        print("Bad model, retrain again.\n")
        return False, None, None

def plot_results(scaled_data, train_predict, test_predict, scaler, time_step, count):
    look_back = time_step

    # Initialize arrays
    trainPredictPlot = np.empty_like(scaled_data)
    trainPredictPlot[:, :] = np.nan
    testPredictPlot = np.empty_like(scaled_data)
    testPredictPlot[:, :] = np.nan

    # Plot train predictions
    trainPredictPlot[look_back:len(train_predict) + look_back, :] = train_predict

    # Plot test predictions
    test_start_index = len(train_predict) + (look_back * 2)
    test_end_index = test_start_index + len(test_predict)
    
    if test_end_index > len(scaled_data):
        test_end_index = len(scaled_data) - 1

    # Print shapes and indices for debugging
    print(f"test_predict shape: {test_predict.shape}")
    print(f"trainPredictPlot shape: {trainPredictPlot.shape}")
    print(f"testPredictPlot shape: {testPredictPlot.shape}")
    print(f"test_start_index: {test_start_index}")
    print(f"test_end_index: {test_end_index}")

    testPredictPlot[test_start_index:test_end_index, :] = test_predict[:test_end_index - test_start_index]

    # Plot the results
    plt.figure(figsize=(10, 8))
    plt.plot(scaler.inverse_transform(scaled_data), label='Actual Data')
    plt.plot(trainPredictPlot, label='Train Predictions')
    plt.plot(testPredictPlot, label='Test Predictions')
    plt.legend()
    plt.title('LSTM Model Backtesting')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.savefig(f'{os.getenv('PLOTS_DOWNLOAD_PATH')}_{count}.png')
    plt.close()

def main():
    symbol = 'BTC-USD'
    count = 1

    while True:
        # Set the current start and end date
        start_date = (datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d')
        end_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')  # Add 1 day to fetch newest data on current date
        
        historical_data = fetch_data_in_chunks(symbol, start_date, end_date)
        scaled_data, scaler = preprocess_data(historical_data)
        
        train_data, test_data = split_data(scaled_data)
        X_train, Y_train = create_dataset(train_data)
        X_test, Y_test = create_dataset(test_data)
        
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        model = train_model(X_train, Y_train)
        
        # Evaluate and save the model
        is_valid, train_predict_inverse, test_predict_inverse = evaluate_and_save_model(
            model, X_train, Y_train, X_test, Y_test, scaler
        )

        # If a valid model is found, plot the results
        if is_valid:
            plot_results(scaled_data, train_predict_inverse, test_predict_inverse, scaler, time_step=60, count=count)
            count += 1
        else:
            continue

        # Countdown timer using tqdm
        print("\nWaiting for the next retraining...")
        for _ in tqdm(range(3600)):
            time.sleep(1)

if __name__ == "__main__":
    main()
