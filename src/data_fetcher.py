import requests
import pandas as pd
import time

BASE_URL = 'https://api.binance.com/api/v3/klines'

def fetch_historical_data(symbol='BTCUSDT', interval='1h', start_time=None, end_time=None):
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit':1000
    }
    if start_time:
        params['startTime'] = start_time
    if end_time:
        params['endTime'] = end_time

    response = requests.get(BASE_URL, params=params)
    data = response.json()

    if isinstance(data, dict) and data.get("code"):
        raise Exception(f"Error fetching data: {data}")

    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    df = df.astype({'open':'float', 'high':'float', 'low':'float', 'close':'float', 'volume':'float'})

    return df
