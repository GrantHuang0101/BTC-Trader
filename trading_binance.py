import numpy as np
from binance.client import Client
from binance.enums import *
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from dotenv import load_dotenv
import os
import time

load_dotenv()  # take environment variables from .env.

# Initialize Binance API client
api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_API_SECRET')
client = Client(api_key, api_secret, testnet=True)

# Load the trained LSTM model and scaler
model = load_model('bitcoin_lstm_model.h5')
scaler = MinMaxScaler(feature_range=(0, 1))

# Fetch live data from Binance
def fetch_live_data(symbol, interval='1m', limit=60):
    klines = client.futures_klines(symbol=symbol, interval=interval, limit=limit)
    data = [float(kline[4]) for kline in klines]  # Closing price is at index 4
    return np.array(data).reshape(-1, 1)

# Make prediction using the LSTM model
def make_prediction(model, live_data, scaler, time_step=60):
    # scaler.fit(live_data)
    scaled_data = scaler.fit_transform(live_data)
    X = []
    X.append(scaled_data[-time_step:, 0])
    X = np.array(X)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    prediction = model.predict(X)
    predicted_price = scaler.inverse_transform(prediction)
    return predicted_price[0, 0]

# Place an order on Binance Futures
def place_order(symbol, quantity, order_type=ORDER_TYPE_MARKET, side=SIDE_BUY):
    try:
        order = client.futures_create_order(
            symbol=symbol,
            type=order_type,
            side=side,
            quantity=quantity,
            test=True
        )
        print(f"Order placed: {order['side']} {order['origQty']}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Execute trading strategy
def execute_trading_strategy(symbol, model, scaler, time_step=60):
    live_data = fetch_live_data(symbol)
    predicted_price = make_prediction(model, live_data, scaler, time_step)
    
    current_price = live_data[-1, 0]
    print(f"Predicted Price: {predicted_price}, Current Price: {current_price}")
    
    # Define the thresholds
    # buy_threshold = current_price * 1.005
    # sell_threshold = current_price * 0.995

    # Check current position
    position_info = client.futures_position_information(symbol=symbol)
    current_position = float(position_info[0]['positionAmt'])  # Positive for long, negative for short
    balance = float(client.futures_account_balance()[4]['availableBalance'])  # Available balance in USDT

    # Trading logic
    if predicted_price > current_price:
        # Buy only if we don't have a long position
        place_order(symbol, quantity=0.01, side=SIDE_BUY)
    elif predicted_price < current_price and current_position > 0:
        # Sell only if we have a long position
        place_order(symbol, quantity=0.01, side=SIDE_SELL)
    
    position_info = client.futures_position_information(symbol=symbol)
    current_position = float(position_info[0]['positionAmt'])
    balance = float(client.futures_account_balance()[4]['availableBalance'])
    print(f"Current position: {current_position}, USDT Balance: {balance}")

# Loop for continuous trading
def start_trading(symbol, model, scaler, interval='1m'):
    while True:
        execute_trading_strategy(symbol, model, scaler)
        time.sleep(60)  # Wait for the next interval

if __name__ == "__main__":
    symbol = 'BTCUSDT'
    start_trading(symbol, model, scaler)