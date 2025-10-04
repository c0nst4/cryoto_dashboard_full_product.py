# crypto_dashboard_full_product.py
"""
Profi-Krypto-Dashboard: BTC / ETH / SOL
------------------------------------------------
Features:
✅ Technische Indikatoren (SMA20/50/200, RSI, MACD, ATR)
✅ Fear & Greed Index (alternative.me)
✅ News (CoinTelegraph RSS)
✅ Makro-Analyse mit Prognose (TradingEconomics)
✅ Liquidation Heatmap (Simuliert)
✅ Robuste Fehlerbehandlung (keine Abstürze)
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

# ===================== Einstellungen =====================
st.set_page_config(layout="wide", page_title="Krypto Dashboard")

WATCHLIST = ["BTC-USD", "ETH-USD", "SOL-USD"]

# ===================== Helper Funktionen =====================
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

def safe_series(df, col):
    try:
        s = df[col]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        return pd.Series(s.values, index=df.index).astype(float)
    except Exception:
        return pd.Series(dtype=float, index=df.index)

# ===================== Technische Analyse =====================
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

# ===================== Fear & Greed Index =====================
@st.cache_data(ttl=600)
def fetch_fear_greed():
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
        data = r.json()
        return int(data["data"][0]["value"]), data["data"][0]["value_classification"]
    except Exception:
        return None, None

# ===================== News =====================
@st.cache_data(ttl=600)
def fetch_news(limit=6):
    try:
        r = requests.get("https://cointelegraph.com/rss", timeout=10)
        soup = BeautifulSoup(r.content, "xml")
        items = soup.find_all("item")[:limit]
        return [{"title": i.title.text, "link": i.link.text} for i in items]
    except Exception:
        return []

# ===================== Makro Analyse =====================
@st.cache_data(ttl=600)
def fetch_tradingeconomics_calendar(start_date, end_date):
    key = os.getenv("TRADINGECONOMICS_API_KEY", "guest:guest")
    try:
        url = f"https://api.tradingeconomics.com/calendar?start_date={start_date}&end_date={end_date}&c={key}"
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return []
        data = r.json()
        return [ev for ev in data if ev.get("importance") == 3]
    except Exception:
        return []

def train_monthly_model(symbol):
    df = fetch_ohlc(symbol, months=14)
    if df is None or df.empty:
        return None, None
    monthly = df["Close"].resample("M").last().pct_change().dropna()
    if len(monthly) < 6:
        return None, None
    X = np.arange(len(monthly)).reshape(-1, 1)
    y = monthly.values
    model = LinearRegression().fit(X, y)
    return model, r2_score(y, model.predict(X))

def predict_next_month(symbol):
    model, r2 = train_monthly_model(symbol)
    if model is None:
        return None, None
    X_next = np.array([[len(model.coef_) + 1]])
    pred = model.predict(X_next)[0]
    return pred, r2

# ===================== Streamlit Layout =====================
st.title("📊 Krypto Dashboard: BTC / ETH / SOL")
tabs = st.tabs(["Übersicht", "Charts", "Makro-Analyse", "Heatmap"])

# ---------- Übersicht ----------
with tabs[0]:
    st.header("📈 Marktübersicht")
    fgi, fgi_text = fetch_fear_greed()
    st.metric("Fear & Greed Index", fgi if fgi else "n/a", fgi_text if fgi_text else "n/a")

    st.subheader("📰 Aktuelle Krypto-News")
    for news in fetch_news():
        st.markdown(f"- [{news['title']}]({news['link']})")

    st.subheader("📅 Kommende High-Impact Events")
    today = datetime.utcnow().date()
    events = fetch_tradingeconomics_calendar(today.strftime("%Y-%m-%d"), (today + timedelta(days=30)).strftime("%Y-%m-%d"))
    if events:
        for ev in events[:5]:
            st.write(f"- {ev.get('date')} {ev.get('country')}: {ev.get('event')}")
    else:
        st.info("Keine High-Impact Events verfügbar (API-Limit oder keine Daten).")

# ---------- Charts ----------
with tabs[1]:
    st.header("📊 Charts & Technische Signale")
    for ticker in WATCHLIST:
        st.subheader(ticker)
        df = fetch_ohlc(ticker, months=12)
        if df is None:
            st.warning("Keine Daten verfügbar.")
            continue

        df = compute_indicators(df)
        latest = df.iloc[-1]
        latest_close = float(latest["Close"]) if "Close" in latest else np.nan
        latest_rsi = float(latest["RSI"]) if "RSI" in latest else np.nan
        st.write(f"Close: {latest_close:.2f} USD | RSI: {latest_rsi:.1f}")

        fig = go.Figure(data=[go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"]
        )])
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA20"], name="SMA20"))
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA50"], name="SMA50"))
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA200"], name="SMA200"))
        fig.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

# ---------- Makro ----------
with tabs[2]:
    st.header("🌍 Makro-Analyse & Monatsprognose")
    for ticker in WATCHLIST:
        pred, r2 = predict_next_month(ticker)

        if pred is not None and isinstance(pred, (float, int)) and not np.isnan(pred):
            pred_pct = f"{pred * 100:.2f}%"
        else:
            pred_pct = "n/a"

        if r2 is not None and isinstance(r2, (float, int)) and not np.isnan(r2):
            r2_val = f"{r2:.2f}"
        else:
            r2_val = "n/a"

        st.write(f"{ticker}: Prognose nächster Monat = {pred_pct} (R²={r2_val})")

# ---------- Heatmap ----------
with tabs[3]:
    st.header("🔥 Liquidation Heatmap (Simuliert)")
    coins = ["BTC", "ETH", "SOL"]
    data = []
    for c in coins:
        for p in np.linspace(0.9, 1.1, 40):
            data.append({"Coin": c, "price": p, "long_liq": np.random.poisson(20), "short_liq": np.random.poisson(15)})
    df_heat = pd.DataFrame(data)
    fig = px.density_heatmap(df_heat, x="price", y="Coin", z="long_liq", color_continuous_scale="Reds")
    st.plotly_chart(fig, use_container_width=True)
