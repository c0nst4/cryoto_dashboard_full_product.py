# crypto_dashboard_full_product.py
"""
Krypto Dashboard – BTC, ETH, SOL
--------------------------------
✅ SMA20 / SMA50 / SMA200
✅ EMA20 / EMA50 / EMA200
✅ RSI, MACD
✅ Fear & Greed Index
✅ News (CoinTelegraph + Investing.com)
✅ Makro-Trendprognose (BTC vs DXY + VIX)
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import ta
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# =================== Einstellungen ===================
st.set_page_config(layout="wide", page_title="Krypto Dashboard")
WATCHLIST = ["BTC-USD", "ETH-USD", "SOL-USD"]

# =================== Kursdaten abrufen ===================
@st.cache_data(ttl=600)
def fetch_ohlc(symbol, months=12):
    df = yf.download(symbol, period=f"{months}mo", interval="1d", progress=False)
    if df.empty:
        return None
    df.index = pd.to_datetime(df.index)
    return df

def safe_series(df, col):
    try:
        s = df[col]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        return pd.Series(s.values, index=df.index).astype(float)
    except Exception:
        return pd.Series(dtype=float, index=df.index)

# =================== Technische Analyse ===================
def compute_indicators(df):
    close = safe_series(df, "Close")

    # SMA
    df["SMA20"] = close.rolling(20).mean()
    df["SMA50"] = close.rolling(50).mean()
    df["SMA200"] = close.rolling(200).mean()

    # EMA
    df["EMA20"] = close.ewm(span=20, adjust=False).mean()
    df["EMA50"] = close.ewm(span=50, adjust=False).mean()
    df["EMA200"] = close.ewm(span=200, adjust=False).mean()

    # RSI
    try:
        df["RSI"] = ta.momentum.RSIIndicator(close, window=14).rsi()
    except Exception:
        df["RSI"] = np.nan

    # MACD
    try:
        macd = ta.trend.MACD(close)
        df["MACD"] = macd.macd()
        df["MACD_SIGNAL"] = macd.macd_signal()
    except Exception:
        df["MACD"], df["MACD_SIGNAL"] = np.nan, np.nan

    return df

# =================== Fear & Greed Index ===================
@st.cache_data(ttl=3600)
def fetch_fear_greed():
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
        js = r.json()
        return int(js["data"][0]["value"]), js["data"][0]["value_classification"]
    except Exception:
        return None, None

# =================== News abrufen ===================
@st.cache_data(ttl=900)
def fetch_news(limit=8):
    news = []
    try:
        # CoinTelegraph
        r = requests.get("https://cointelegraph.com/rss", timeout=10)
        soup = BeautifulSoup(r.content, "xml")
        items = soup.find_all("item")[:limit // 2]
        for i in items:
            news.append({"source": "CoinTelegraph", "title": i.title.text, "link": i.link.text})
    except Exception:
        pass
    try:
        # Investing.com
        r = requests.get("https://www.investing.com/rss/news_25.rss", timeout=10)
        soup = BeautifulSoup(r.content, "xml")
        items = soup.find_all("item")[:limit // 2]
        for i in items:
            news.append({"source": "Investing.com", "title": i.title.text, "link": i.link.text})
    except Exception:
        pass
    return news

# =================== Makro-Analyse ===================
@st.cache_data(ttl=3600)
def fetch_macro_data():
    try:
        vix = yf.download("^VIX", period="2y", interval="1d")["Close"].resample("M").last()
        dxy = yf.download("DX-Y.NYB", period="2y", interval="1d")["Close"].resample("M").last()
        btc = yf.download("BTC-USD", period="2y", interval="1d")["Close"].resample("M").last()
        df = pd.concat({"BTC": btc.pct_change(), "DXY": dxy.pct_change(), "VIX": vix.pct_change()}, axis=1).dropna()
        return df
    except Exception:
        return pd.DataFrame()

def predict_macro_trend():
    df = fetch_macro_data()
    if df.empty or len(df) < 6:
        return None, None
    X = df[["DXY", "VIX"]]
    y = df["BTC"]
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    model = LinearRegression().fit(X_scaled, y)
    r2 = model.score(X_scaled, y)
    pred = model.predict(X_scaled[-1].reshape(1, -1))[0]
    return pred, r2

# =================== Streamlit Layout ===================
st.title("📊 Profi Krypto-Dashboard (BTC / ETH / SOL)")

tabs = st.tabs(["Übersicht", "Charts", "Makro-Analyse"])

# --- Übersicht ---
with tabs[0]:
    fgi, txt = fetch_fear_greed()
    st.metric("Fear & Greed Index", fgi if fgi else "n/a", txt if txt else "n/a")

    st.subheader("📰 Aktuelle News")
    for n in fetch_news():
        st.markdown(f"- **[{n['source']}]** [{n['title']}]({n['link']})")

# --- Charts ---
with tabs[1]:
    for ticker in WATCHLIST:
        st.subheader(ticker)
        df = fetch_ohlc(ticker)
        if df is None:
            st.warning("Keine Daten gefunden.")
            continue

        df = compute_indicators(df)
        latest = df.iloc[-1]

        st.write(f"Close: {latest['Close']:.2f} USD | RSI: {latest['RSI']:.1f}")

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
            name="Preis"
        ))

        # SMA Linien
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA20"], name="SMA20", line=dict(width=1)))
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA50"], name="SMA50", line=dict(width=1)))
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA200"], name="SMA200", line=dict(width=1)))

        # EMA Linien
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"], name="EMA20", line=dict(dash="dot")))
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA50"], name="EMA50", line=dict(dash="dot")))
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA200"], name="EMA200", line=dict(dash="dot")))

        fig.update_layout(height=400, xaxis_title="Datum", yaxis_title="Preis (USD)")
        st.plotly_chart(fig, use_container_width=True)

# --- Makro ---
with tabs[2]:
    st.header("🌍 Makro-Analyse & Trendprognose")
    pred, r2 = predict_macro_trend()
    if pred is None:
        st.warning("Keine Makro-Prognose verfügbar.")
    else:
        trend = "Bullish" if pred > 0.02 else "Bearish" if pred < -0.02 else "Neutral"
        symbol = "🟢" if trend == "Bullish" else "🔴" if trend == "Bearish" else "⚪"
        st.write(f"{symbol} Prognose: {trend} (R²={r2:.2f})")

    st.dataframe(fetch_macro_data().tail(6).round(3))
