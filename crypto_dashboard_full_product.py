# crypto_dashboard_full_product.py
"""
Vollständiges Krypto-Dashboard (BTC/ETH/SOL)
Features:
 - Technische Indikatoren: SMA20/50/200, RSI, MACD, ATR
 - Candlestick-Charts (Plotly)
 - Crypto Fear & Greed Index
 - Krypto-News (CoinTelegraph RSS)
 - Makro-Kalender (TradingEconomics, High-Impact) + Prognose
 - CoinAnk Liquidation Heatmap (Dummy oder API-Key)
"""

import os
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import ta
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ---------------- CONFIG ----------------
st.set_page_config(layout="wide", page_title="Krypto Profi-Dashboard")

WATCHLIST = ["BTC-USD", "ETH-USD", "SOL-USD"]

# ---------------- Helpers ----------------
@st.cache_data(ttl=300)
def fetch_ohlc(symbol, months=12):
    try:
        df = yf.download(symbol, period=f"{months}mo", interval="1d", progress=False)
        if df is None or df.empty:
            return None
        df.index = pd.to_datetime(df.index)
        return df
    except Exception:
        return None

def safe_series_from_df(df, col):
    """Extrahiert eine Spalte robust als 1D Series."""
    try:
        s = df[col]
        if isinstance(s, pd.DataFrame):  # falls DataFrame statt Series
            s = s.iloc[:, 0]
        return pd.Series(s.values, index=df.index).astype(float)
    except Exception:
        return pd.Series(dtype=float, index=df.index)

# ---------------- Technical indicators ----------------
def compute_indicators(df):
    close = safe_series_from_df(df, "Close")
    high = safe_series_from_df(df, "High")
    low = safe_series_from_df(df, "Low")

    df["SMA20"] = close.rolling(20, min_periods=1).mean()
    df["SMA50"] = close.rolling(50, min_periods=1).mean()
    df["SMA200"] = close.rolling(200, min_periods=1).mean()

    try:
        df["RSI"] = ta.momentum.RSIIndicator(close, window=14).rsi()
    except Exception:
        df["RSI"] = np.nan

    try:
        macd = ta.trend.MACD(close)
        df["MACD"] = macd.macd()
        df["MACD_SIGNAL"] = macd.macd_signal()
    except Exception:
        df["MACD"], df["MACD_SIGNAL"] = np.nan, np.nan

    try:
        atr = ta.volatility.AverageTrueRange(high, low, close, window=14)
        df["ATR"] = atr.average_true_range()
    except Exception:
        df["ATR"] = close.pct_change().rolling(14).std() * close

    return df

# ---------------- Fear & Greed ----------------
@st.cache_data(ttl=300)
def fetch_fear_greed():
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
        js = r.json()
        return int(js["data"][0]["value"]), js["data"][0]["value_classification"]
    except Exception:
        return None, None

# ---------------- News ----------------
@st.cache_data(ttl=600)
def fetch_news(limit=8):
    url = "https://cointelegraph.com/rss"
    try:
        r = requests.get(url, timeout=10)
        soup = BeautifulSoup(r.content, "xml")
        return [{"title": it.title.text, "link": it.link.text} for it in soup.find_all("item")[:limit]]
    except Exception:
        return []

# ---------------- Makro-Daten ----------------
@st.cache_data(ttl=600)
def fetch_tradingeconomics_calendar(start_date, end_date):
    try:
        url = f"https://api.tradingeconomics.com/calendar?start_date={start_date}&end_date={end_date}&c=guest:guest"
        r = requests.get(url, timeout=15)
        return [ev for ev in r.json() if ev.get("importance") == 3]
    except Exception:
        return []

# ---------------- Prognose ----------------
def train_monthly_model(symbol):
    df = fetch_ohlc(symbol, months=14)
    if df is None:
        return None, None, None, None
    monthly = df["Close"].resample("M").last().pct_change().dropna()
    if len(monthly) < 4:
        return None, None, None, None
    X = np.arange(len(monthly)).reshape(-1,1)
    y = monthly.values
    model = LinearRegression()
    model.fit(X,y)
    return model, X, y, r2_score(y, model.predict(X))

def predict_next_month(symbol):
    model, X, y, r2 = train_monthly_model(symbol)
    if model is None:
        return None, None
    try:
        pred = model.predict([[len(X)]])[0]
        return pred, r2
    except Exception:
        return None, None

# ---------------- UI ----------------
st.title("📊 Profi Krypto-Dashboard (BTC / ETH / SOL)")
tabs = st.tabs(["Übersicht", "Charts & Indikatoren", "Makro-Analyse", "Heatmap"])

# --- Tab Übersicht
with tabs[0]:
    fgi, fgi_text = fetch_fear_greed()
    st.metric("Fear & Greed", fgi if fgi else "n/a", fgi_text if fgi_text else "n/a")

    st.subheader("Krypto News")
    for n in fetch_news(5):
        st.write(f"- [{n['title']}]({n['link']})")

    st.subheader("High-Impact Events (30 Tage)")
    today = datetime.utcnow().date()
    events = fetch_tradingeconomics_calendar(today.strftime("%Y-%m-%d"), (today+timedelta(days=30)).strftime("%Y-%m-%d"))
    for ev in events[:5]:
        st.write(f"- {ev.get('date')} {ev.get('country')}: {ev.get('event')}")

# --- Tab Charts
with tabs[1]:
    for ticker in WATCHLIST:
        st.subheader(ticker)
        df = fetch_ohlc(ticker, months=12)
        if df is None:
            st.write("Keine Daten verfügbar.")
            continue
        df = compute_indicators(df)
        latest = df.iloc[-1]

        latest_close = float(latest["Close"]) if "Close" in latest else np.nan
        latest_rsi = float(latest["RSI"]) if "RSI" in latest else np.nan

        st.write(f"Close: {latest_close:.2f} | RSI: {latest_rsi:.1f}")

        fig = go.Figure(data=[go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"]
        )])
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA20"], name="SMA20"))
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA50"], name="SMA50"))
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA200"], name="SMA200"))
        st.plotly_chart(fig, use_container_width=True)

# --- Tab Makro
with tabs[2]:
    st.header("Makro-Analyse & Prognose")
    for ticker in WATCHLIST:
        pred, r2 = predict_next_month(ticker)
        if pred is not None and not np.isnan(pred):
            r2_val = f"{r2:.2f}" if r2 is not None else "n/a"
            st.write(f"{ticker}: Prognose nächster Monat {pred*100:.2f}% (R²={r2_val})")
        else:
            st.write(f"{ticker}: Keine Prognose möglich")

# --- Tab Heatmap
with tabs[3]:
    st.header("Liquidation Heatmap (Dummy-Daten Beispiel)")
    coins = ["BTC", "ETH", "SOL"]
    data = []
    for c in coins:
        for p in np.linspace(0.9, 1.1, 40):
            data.append({"Coin": c, "price": p, "long_liq": np.random.poisson(20), "short_liq": np.random.poisson(15)})
    dfh = pd.DataFrame(data)
    fig = px.density_heatmap(dfh, x="price", y="Coin", z="long_liq", color_continuous_scale="Reds")
    st.plotly_chart(fig, use_container_width=True)
