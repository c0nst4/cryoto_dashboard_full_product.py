# crypto_dashboard_full_product.py
"""
Profi Krypto Dashboard – BTC, ETH, SOL
---------------------------------------
Features:
✅ Technische Analyse (SMA, RSI, MACD, ATR)
✅ Fear & Greed Index
✅ News (CoinTelegraph)
✅ Makro-Analyse mit realen Wirtschaftsdaten:
   - US CPI, Arbeitslosenquote, Zinsen, DXY, VIX
✅ Monatsprognose (bullish/neutral/bearish)
✅ Liquidation Heatmap (Simuliert)
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
from sklearn.preprocessing import StandardScaler

# ============= GRUNDEINSTELLUNGEN =============
st.set_page_config(layout="wide", page_title="Krypto Dashboard")

WATCHLIST = ["BTC-USD", "ETH-USD", "SOL-USD"]

# ============= DATEN LADEN =============
@st.cache_data(ttl=600)
def fetch_ohlc(symbol, months=12):
    try:
        df = yf.download(symbol, period=f"{months}mo", interval="1d", progress=False)
        if df.empty:
            return None
        df.index = pd.to_datetime(df.index)
        return df
    except Exception:
        return None

def safe_series(df, col):
    try:
        s = df[col]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        return pd.Series(s.values, index=df.index).astype(float)
    except Exception:
        return pd.Series(dtype=float, index=df.index)

# ============= TECHNISCHE ANALYSE =============
def compute_indicators(df):
    close = safe_series(df, "Close")
    high = safe_series(df, "High")
    low = safe_series(df, "Low")

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

# ============= FEAR & GREED =============
@st.cache_data(ttl=600)
def fetch_fear_greed():
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
        js = r.json()
        return int(js["data"][0]["value"]), js["data"][0]["value_classification"]
    except Exception:
        return None, None

# ============= NEWS =============
@st.cache_data(ttl=600)
def fetch_news(limit=6):
    try:
        r = requests.get("https://cointelegraph.com/rss", timeout=10)
        soup = BeautifulSoup(r.content, "xml")
        items = soup.find_all("item")[:limit]
        return [{"title": i.title.text, "link": i.link.text} for i in items]
    except Exception:
        return []

# ============= MAKRODATEN (echte Quellen) =============
@st.cache_data(ttl=3600)
def fetch_macro_data():
    # Holt CPI, Unemployment, Fed Funds, DXY, VIX
    try:
        macro = {}
        macro["CPI"] = yf.download("^CPI", period="2y", interval="1mo")["Close"]  # US Inflation (geschätzt)
    except Exception:
        macro["CPI"] = pd.Series(dtype=float)

    try:
        macro["UNEMP"] = yf.download("UNRATE", period="2y", interval="1mo")["Close"]  # Arbeitslosenquote
    except Exception:
        macro["UNEMP"] = pd.Series(dtype=float)

    try:
        macro["FEDFUNDS"] = yf.download("FEDFUNDS", period="2y", interval="1mo")["Close"]  # Zins
    except Exception:
        macro["FEDFUNDS"] = pd.Series(dtype=float)

    try:
        macro["DXY"] = yf.download("DX-Y.NYB", period="2y", interval="1d")["Close"].resample("M").last()
    except Exception:
        macro["DXY"] = pd.Series(dtype=float)

    try:
        macro["VIX"] = yf.download("^VIX", period="2y", interval="1d")["Close"].resample("M").last()
    except Exception:
        macro["VIX"] = pd.Series(dtype=float)

    # Kombinieren in DataFrame
    df = pd.concat(macro, axis=1)
    df = df.dropna(how="all")
    return df

def train_macro_model(symbol):
    df_price = yf.download(symbol, period="2y", interval="1d", progress=False)
    if df_price.empty:
        return None, None

    df_price = df_price["Close"].resample("M").last().pct_change().dropna()
    df_macro = fetch_macro_data()
    df = pd.concat([df_price.rename("RET"), df_macro], axis=1).dropna()
    if len(df) < 6:
        return None, None

    X = df[["CPI", "UNEMP", "FEDFUNDS", "DXY", "VIX"]]
    y = df["RET"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(X_scaled, y)
    r2 = model.score(X_scaled, y)

    return (model, scaler, X.iloc[-1:].values, r2)

def predict_next_month(symbol):
    trained = train_macro_model(symbol)
    if trained is None or trained[0] is None:
        return None, None
    model, scaler, lastX, r2 = trained
    try:
        pred = model.predict(scaler.transform(lastX))[0]
        return pred, r2
    except Exception:
        return None, None

# ============= STREAMLIT UI =============
st.title("📊 Profi Krypto-Dashboard (BTC / ETH / SOL)")
tabs = st.tabs(["Übersicht", "Charts", "Makro-Analyse", "Heatmap"])

# --- Übersicht ---
with tabs[0]:
    fgi, txt = fetch_fear_greed()
    st.metric("Fear & Greed Index", fgi if fgi else "n/a", txt if txt else "n/a")
    st.subheader("📰 Aktuelle Krypto-News")
    for n in fetch_news():
        st.markdown(f"- [{n['title']}]({n['link']})")

# --- Charts ---
with tabs[1]:
    for ticker in WATCHLIST:
        df = fetch_ohlc(ticker, months=12)
        if df is None:
            st.warning(f"Keine Daten für {ticker}")
            continue
        df = compute_indicators(df)
        latest = df.iloc[-1]
        latest_close = float(latest["Close"]) if "Close" in latest else np.nan
        latest_rsi = float(latest["RSI"]) if "RSI" in latest else np.nan
        st.write(f"{ticker}: Close={latest_close:.2f} USD | RSI={latest_rsi:.1f}")

        fig = go.Figure(data=[go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"]
        )])
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA20"], name="SMA20"))
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA50"], name="SMA50"))
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA200"], name="SMA200"))
        st.plotly_chart(fig, use_container_width=True)

# --- Makro-Analyse ---
with tabs[2]:
    st.header("🌍 Makro-Analyse & Prognose auf Basis realer Wirtschaftsdaten")
    st.write("Modelle basieren auf CPI, Zinsen, Arbeitsmarkt, USD-Index und VIX.")
    df_macro = fetch_macro_data()
    if df_macro.empty:
        st.error("Makrodaten konnten nicht geladen werden.")
    else:
        st.write("Aktuelle Makro-Daten:")
        st.dataframe(df_macro.tail(6).round(3))

    for ticker in WATCHLIST:
        pred, r2 = predict_next_month(ticker)
        if pred is None:
            st.warning(f"{ticker}: Keine Prognose möglich (unzureichende Daten).")
            continue

        trend = "Bullish" if pred > 0.02 else ("Bearish" if pred < -0.02 else "Neutral")
        color = "🟢" if trend == "Bullish" else "🔴" if trend == "Bearish" else "⚪"
        st.write(f"{ticker}: {color} Prognose nächster Monat {pred*100:.2f}% (R²={r2:.2f}) → **{trend}**")

# --- Heatmap ---
with tabs[3]:
    st.header("🔥 Liquidation Heatmap (Simulation)")
    coins = ["BTC", "ETH", "SOL"]
    data = []
    for c in coins:
        for p in np.linspace(0.9, 1.1, 40):
            data.append({
                "Coin": c,
                "price": p,
                "long_liq": np.random.poisson(20),
                "short_liq": np.random.poisson(15)
            })
    dfh = pd.DataFrame(data)
    fig = px.density_heatmap(dfh, x="price", y="Coin", z="long_liq", color_continuous_scale="Reds")
    st.plotly_chart(fig, use_container_width=True)
