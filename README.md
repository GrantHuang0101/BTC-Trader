# BTC-Trader: A Bitcoin Trading Bot

BTC_Trader is a Python-based project that includes two main components:

1. **LSTM Model Training**: A script that fetches historical Bitcoin data, trains an LSTM model to predict future prices, and evaluates the model for potential use in trading.

2. **Real-Time Trading**: A script that utilizes the trained LSTM model to make real-time trading decisions on Binance Futures, following a martingale-based strategy.

## Required Python Packages

You can install the required packages by running:

```bash
pip install -r requirements.txt
```

## Environment Variables

Ensure you have a .env file with the following keys:

1. BINANCE_API_KEY: Your Binance API key.
2. BINANCE_API_SECRET: Your Binance API secret.
3. PLOTS_DOWNLOAD_PATH: Path where you want to save the plots.

## Usage

### 1. Training the LSTM Model

To train the LSTM model using historical Bitcoin data, run the following command:

```bash
python3 LSTM_v2.py
```

This script fetches data from Yahoo Finance, preprocesses it, trains an LSTM model, and saves the model if it meets certain criteria.

### 2. Real-Time Trading

To start real-time trading on Binance using the trained LSTM model, run:

```bash
python3 trading_binance.py
```

This script fetches live data from Binance, uses the LSTM model to predict prices, and executes trades based on a martingale strategy. The bot continuously monitors the market and adjusts its strategy in real-time.

## Notes

1. Ensure the .env file is correctly configured with your Binance API credentials and the path for saving plots.
2. The bot operates in a loop, retraining the model every hour and making real-time trading decisions.

## Disclaimer

Trading cryptocurrencies involves risk, and this bot should be used for educational purposes only. Use it at your own risk.
