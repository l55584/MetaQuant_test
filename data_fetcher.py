# data_fetcher.py
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import yfinance as yf
import logging
import time
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import random

logger = logging.getLogger(__name__)

# Pre-filtered universe of 20 high-quality, liquid assets
HIGH_QUALITY_ASSETS = [
    "BTC-USD",
    "ETH-USD",
    "SOL-USD",
    "BNB-USD",
    "XRP-USD",
    "ADA-USD",
    "AVAX-USD",
    "DOT-USD",
    "LINK-USD",
    "LTC-USD",
    "ZEC-USD",
    "ATOM-USD",
    "ETC-USD",
    "XLM-USD",
    "ALGO-USD",
    "UNI-USD",
    "AAVE-USD",
    "FIL-USD",
    "EOS-USD",
    "XTZ-USD",
    "ICP-USD",
    "DOGE-USD",
]

HORUS_KEY = "c2ee3103d6057c2932b984b488c19db0df1d79d00775e5c88816d5e88d994afb"

#para interval can take: 1h, 15min, 1d
def get_history_market_data(ticker, interval="1h", duration=100):
    """Get historical market data for a single ticker"""
    clean_ticker = ticker.replace("-USD", "")

    # Horus API call
    url = "https://api-horus.com/market/price"
    extend = {"1h": 3600, "15min": 60 * 15,'1d':24*60*60}
    duration_seconds = duration * extend[interval]
    start = int(time.time()) - duration_seconds
    end = int(time.time())

    try:
        response = requests.get(
            url,
            headers={"X-API-Key": HORUS_KEY},
            params={
                "asset": clean_ticker,
                "interval": interval,
                "start": start,
                "end": end,
            },
            timeout=10,
        )

        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                prices = [item["price"] for item in data if "price" in item]
                logger.info(f"✓ {ticker}: {len(prices)} points from Horus")
                return prices
    except Exception as e:
        logger.warning(f"Horus API failed for {ticker}: {e}")

    # Yahoo Finance fallback
    try:
        end = datetime.now(timezone.utc)
        if(interval[-1]=='h'):
            start = end - timedelta(hours=duration)
        if(interval[-1]=='d'):
            start = end- timedelta(days=duration)

        df = yf.download(
            tickers=ticker, interval=interval, start=start, end=end, progress=False, auto_adjust=False
        )

        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                close_prices = df[("Close", ticker)].tolist()
            else:
                close_prices = df["Close"].tolist()

            close_prices = [x for x in close_prices if not np.isnan(x)]
            logger.info(f"✓ {ticker}: {len(close_prices)} points from Yahoo")
            return close_prices
    except Exception as e:
        logger.error(f"Yahoo Finance failed for {ticker}: {e}")

    return []


def get_all_market_data(interval="1h", duration=100):
    """Get market data for all high-quality assets"""
    logger.info(f"🔄 Fetching data for {len(HIGH_QUALITY_ASSETS)} high-quality assets")

    df = {}
    successful_assets = []

    for asset in HIGH_QUALITY_ASSETS:
        #print(asset)
        prices = get_history_market_data(asset, interval=interval, duration=duration)
        if prices:
            df[asset] = prices
            successful_assets.append(asset)
    # Ensure equal length
    min_length = min(len(df[asset]) for asset in successful_assets)
    for asset in successful_assets:
        df[asset] = df[asset][-min_length:]

    price_df = pd.DataFrame(df)
    logger.info(
        f"✅ Data fetched: {price_df.shape[1]} assets, {price_df.shape[0]} periods"
    )
    return price_df, successful_assets

address=['address_count','address_percentage','balance_value','supply_percentage','whale_supply_share','whale_net_flow','whale_inflow_count']
defi=['tvl','chain_tvl','chain_tvl_dominance','protocol_tvl','protocal_tvl_by_chain','protocal_tvl_by_asset']
#data can only be on a daily basis
#when para "timestamp" is true, it also recores timestamp in an independent column
def get_horus_sentiment(sentiments=[],interval='1d',duration=50,timestamp=False):
    sentiment_scores = {}
    duration+=2
    duration = duration * 24*60*60
    start=int(time.time())-duration
    end=int(time.time())
    timestamp_rec=[]
    url=''
    for sentiment in sentiments:
        if(sentiment in address):
            url = f"https://api-horus.com/addresses/{sentiment}"
        elif(sentiment in defi):
            url = f"https://api-horus.com/defi/{sentiment}"
        else:
            logger.warning('sentiment name invalid')
            return
        response = requests.get(
            url,
            headers={"X-API-Key": HORUS_KEY},
            params={
                "chain": "bitcoin",
                'interval':interval,
                'start' : start,
                'end' : end
                },
            timeout=10,
        )
        if response.status_code == 200:
            data = response.json()
            sentiment_scores[sentiment] = [item.get(sentiment,item.get('address_distribution',0)) for item in data]
            timestamp_rec=[item['timestamp'] for item in data]
        else:
            logger.warning(f"Sentiment failed for {sentiment}: {response.text}")
    #length-set:
    min_length = min(len(sentiment_scores[asset]) for asset in sentiment_scores.keys())
    for asset in sentiment_scores.keys():
        sentiment_scores[asset] = sentiment_scores[asset][-min_length:]
    #data-clean:
    if('whale_supply_share' in sentiment_scores.keys()):
       for index in range(len(sentiment_scores['whale_supply_share'])):
            if(sentiment_scores['whale_supply_share'][index]>1 and index):
                sentiment_scores['whale_supply_share'][index]=sentiment_scores['whale_supply_share'][index-1]
    if(timestamp==False):
        return pd.DataFrame(sentiment_scores)
    else:
        sentiment_scores['timestamp']=timestamp_rec
        return pd.DataFrame(sentiment_scores)
def get_sentiment_score(sentiments={'whale_net_flow','whale_inflow_count','chain_tvl'}):
    try:
        df = get_horus_sentiment(sentiments, interval = '1d', duration = 50, timestamp=False)
        if df is None or df.empty:
            # Return mock sentiment data
            return pd.DataFrame({'sentiment_score': [random.uniform(-1, 1) for _ in range(50)]})
        
        df = df.dropna()
        features = df.columns

        scaler = StandardScaler()
        scaled = scaler.fit_transform(df[features])

        pca = PCA(n_components=1)
        sentiment_raw = pca.fit_transform(scaled)

        score =-1+2*MinMaxScaler().fit_transform(sentiment_raw)

        df['sentiment_score'] = score
        return df[['sentiment_score']]
    except Exception as e:
        logger.error(f"Sentiment failed, using mock: {e}")
        return pd.DataFrame({'sentiment_score': [random.uniform(-1, 1) for _ in range(50)]})