import numpy as np
from binance.client import Client
from binance.enums import *
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from dotenv import load_dotenv
import os
import time
from tqdm import tqdm

load_dotenv()  # take environment variables from .env.

# Initialize Binance API client
api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_API_SECRET')
client = Client(api_key, api_secret, testnet=True)

# Load the trained LSTM model and scaler
# model = load_model('bitcoin_lstm_model.h5')
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

# Global variables
initial_buy_price = None
average_cost = None
buy_times = 0

# Execute trading strategy
def execute_trading_strategy(symbol, scaler, time_step=60):
    global initial_buy_price, average_cost, buy_times

    # Load the latest model
    model = load_model('bitcoin_lstm_model.h5')
    # model.compile(optimizer='adam', loss='mean_squared_error')

    live_data = fetch_live_data(symbol)
    predicted_price = make_prediction(model, live_data, scaler, time_step)
    
    current_price = live_data[-1, 0]
    print(f"Predicted Price: {predicted_price}, Current Price: {current_price}")

    # Check current position
    position_info = client.futures_position_information(symbol=symbol)
    current_position = float(position_info[0]['positionAmt'])  # Positive for long, negative for short
    balance = float(client.futures_account_balance()[4]['availableBalance'])  # Available balance in USDT

    # Trading logic
    if current_position == 0 and predicted_price > current_price + 10:
        initial_buy_price = current_price
        average_cost = current_price
        buy_times = 0
        place_order(symbol, quantity=0.05, side=SIDE_BUY)

    elif current_position > 0 and predicted_price > current_price:
        drop_threshold = initial_buy_price * (0.98 ** (buy_times + 1))
        
        if current_price <= drop_threshold and buy_times < 7:
            buy_times += 1
            quantity = 0.05 * (1.2 ** buy_times)
            average_cost = (average_cost * (1.2 ** (buy_times - 1)) + current_price * quantity) / (1.2 ** buy_times)
            place_order(symbol, quantity=quantity, side=SIDE_BUY)

    elif average_cost and average_cost > 0 and current_price >= average_cost * 1.2:
        place_order(symbol, quantity=current_position, side=SIDE_SELL)
        initial_buy_price = None
        average_cost = None
        buy_times = 0
    
    position_info = client.futures_position_information(symbol=symbol)
    current_position = float(position_info[0]['positionAmt'])
    balance = float(client.futures_account_balance()[4]['availableBalance'])
    print(f"Current position: {current_position}, USDT Balance: {balance}")
    print(f"Average Cost: {average_cost}, Buy times: {buy_times}")

# Loop for continuous trading
def start_trading(symbol, scaler, interval='1m'):
    while True:
        execute_trading_strategy(symbol, scaler)

        for _ in tqdm(range(60), desc="Waiting..."):
            time.sleep(1)

if __name__ == "__main__":
    symbol = 'BTCUSDT'
    start_trading(symbol, scaler)