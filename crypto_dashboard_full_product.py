# crypto_dashboard_full_product.py
"""
Krypto Profi-Dashboard (robuste Vollversion)
-------------------------------------------------
✅ BTC / ETH / SOL — Multi-Timeframe Analyse (4h, D, W)
✅ SMA, EMA, MACD, RSI, ATR
✅ Fear & Greed Index + News + Wirtschaftskalender
✅ Technische & Makro-Prognosen (Tag, Woche, Monat)
✅ Swing-Signale (EMA/SMA/MACD/RSI)
Hinweis: Keine Anlageberatung.
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
from sklearn.metrics import r2_score
from datetime import datetime
import traceback

# ----------------------------------------------------
# CONFIG
# ----------------------------------------------------
st.set_page_config(layout="wide", page_title="📊 Krypto Profi-Dashboard")
WATCHLIST = ["BTC-USD", "ETH-USD", "SOL-USD"]
TIMEFRAMES = {"4h": {"interval": "4h", "period": "90d"},
              "1d": {"interval": "1d", "period": "1y"},
              "1wk": {"interval": "1wk", "period": "2y"}}

# ----------------------------------------------------
# HELPER-FUNKTIONEN
# ----------------------------------------------------
def safe_series(df, col):
    if df is None or df.empty: 
        return pd.Series(dtype=float)
    if col in df.columns:
        return df[col]
    for c in df.columns:
        if col.lower() in str(c).lower():
            return df[c]
    return pd.Series(dtype=float, index=df.index)

def fetch_ohlc(symbol, interval, period):
    try:
        df = yf.download(symbol, interval=interval, period=period, progress=False)
        df.index = pd.to_datetime(df.index)
        return df
    except Exception:
        return None

def compute_indicators(df):
    if df is None or df.empty: 
        return df
    close = safe_series(df, "Close")
    high = safe_series(df, "High")
    low = safe_series(df, "Low")
    for p in (20, 50, 200):
        df[f"SMA{p}"] = close.rolling(p).mean()
        df[f"EMA{p}"] = close.ewm(span=p, adjust=False).mean()
    try:
        df["RSI"] = ta.momentum.RSIIndicator(close, window=14).rsi()
    except Exception:
        df["RSI"] = np.nan
    try:
        macd = ta.trend.MACD(close)
        df["MACD"] = macd.macd()
        df["MACD_SIGNAL"] = macd.macd_signal()
        df["MACD_DIFF"] = df["MACD"] - df["MACD_SIGNAL"]
    except Exception:
        df["MACD"], df["MACD_SIGNAL"], df["MACD_DIFF"] = np.nan, np.nan, np.nan
    return df

def detect_signals(df):
    alerts = []
    if df is None or df.empty:
        return alerts
    try:
        if len(df.dropna()) < 2:
            return alerts
        prev, cur = df.iloc[-2], df.iloc[-1]
        if prev["EMA20"] < prev["EMA50"] and cur["EMA20"] > cur["EMA50"]:
            alerts.append("🚀 EMA20 kreuzt über EMA50 (bullish)")
        if prev["EMA20"] > prev["EMA50"] and cur["EMA20"] < cur["EMA50"]:
            alerts.append("🔻 EMA20 kreuzt unter EMA50 (bearish)")
        if prev["SMA50"] < prev["SMA200"] and cur["SMA50"] > cur["SMA200"]:
            alerts.append("📈 SMA50 kreuzt über SMA200 (bullish)")
        if prev["SMA50"] > prev["SMA200"] and cur["SMA50"] < cur["SMA200"]:
            alerts.append("📉 SMA50 kreuzt unter SMA200 (bearish)")
        if prev["MACD"] < prev["MACD_SIGNAL"] and cur["MACD"] > cur["MACD_SIGNAL"]:
            alerts.append("📊 MACD bullish crossover")
        if prev["MACD"] > prev["MACD_SIGNAL"] and cur["MACD"] < cur["MACD_SIGNAL"]:
            alerts.append("📊 MACD bearish crossover")
        if cur["RSI"] > 70:
            alerts.append(f"⚠️ RSI überkauft ({cur['RSI']:.1f})")
        if cur["RSI"] < 30:
            alerts.append(f"⚠️ RSI überverkauft ({cur['RSI']:.1f})")
    except Exception:
        pass
    return alerts

# ----------------------------------------------------
# DATENQUELLEN
# ----------------------------------------------------
@st.cache_data(ttl=1800)
def fetch_fear_greed():
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
        js = r.json()
        return int(js["data"][0]["value"]), js["data"][0]["value_classification"]
    except Exception:
        return None, None

@st.cache_data(ttl=900)
def fetch_news(limit=10):
    feeds = [
        ("CoinTelegraph", "https://cointelegraph.com/rss"),
        ("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
        ("Bitcoin.com", "https://news.bitcoin.com/feed/")
    ]
    out = []
    for name, url in feeds:
        try:
            r = requests.get(url, timeout=10)
            soup = BeautifulSoup(r.content, "xml")
            for item in soup.find_all("item")[:limit//len(feeds)+1]:
                title = item.title.text if item.title else ""
                link = item.link.text if item.link else ""
                out.append({"source": name, "title": title, "link": link})
        except Exception:
            continue
    if not out:
        return [{"source": "System", "title": "Keine aktuellen News gefunden", "link": ""}]
    return out[:limit]

@st.cache_data(ttl=3600)
def fetch_macro_data():
    try:
        btc = yf.download("BTC-USD", period="2y", interval="1d", progress=False)["Close"].resample("M").last().pct_change()
        dxy = yf.download("DX-Y.NYB", period="2y", interval="1d", progress=False)["Close"].resample("M").last().pct_change()
        vix = yf.download("^VIX", period="2y", interval="1d", progress=False)["Close"].resample("M").last().pct_change()
        df = pd.concat([btc, dxy, vix], axis=1)
        df.columns = ["BTC", "DXY", "VIX"]
        return df.dropna()
    except Exception:
        return pd.DataFrame()

def predict_macro_trend():
    df = fetch_macro_data()
    if df.empty:
        return None, None
    try:
        X = df[["DXY", "VIX"]].values
        y = df["BTC"].values
        Xs = StandardScaler().fit_transform(X)
        model = LinearRegression().fit(Xs, y)
        pred = model.predict(Xs[-1].reshape(1, -1))[0]
        r2 = r2_score(y, model.predict(Xs))
        return pred, r2
    except Exception:
        return None, None

@st.cache_data(ttl=3600)
def fetch_econ_calendar():
    try:
        r = requests.get("https://api.tradingeconomics.com/calendar?c=guest:guest", timeout=10)
        data = r.json()
        events = [{"country": d.get("country", ""), "event": d.get("event", ""), "impact": d.get("impact", "")}
                  for d in data if str(d.get("impact", "")).lower() == "high"]
        return events[:10]
    except Exception:
        return []

# ----------------------------------------------------
# VISUALISIERUNG
# ----------------------------------------------------
def plot_chart(df, symbol, mode="EMA"):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="Kurs"
    ))
    if mode == "SMA":
        for p in (20, 50, 200):
            fig.add_trace(go.Scatter(x=df.index, y=df[f"SMA{p}"], name=f"SMA{p}", line=dict(width=2)))
    else:
        for p, color in zip((20, 50, 200), ["blue", "orange", "red"]):
            fig.add_trace(go.Scatter(x=df.index, y=df[f"EMA{p}"], name=f"EMA{p}", line=dict(width=2, color=color)))
    fig.update_layout(title=f"{symbol} — {mode}", height=400)
    st.plotly_chart(fig, use_container_width=True)

def plot_rsi_macd(df):
    if "RSI" in df.columns:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI", line=dict(color="gold", width=2)))
        fig.add_hrect(y0=30, y1=70, fillcolor="lightgray", opacity=0.2)
        fig.update_layout(title="RSI", height=200)
        st.plotly_chart(fig, use_container_width=True)
    if "MACD" in df.columns:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", line=dict(color="green", width=2)))
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD_SIGNAL"], name="Signal", line=dict(color="red", width=2)))
        fig.update_layout(title="MACD", height=200)
        st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------
# UI
# ----------------------------------------------------
st.title("📈 Krypto Profi-Dashboard (robust)")

tabs = st.tabs(["Übersicht", "Charts & Indikatoren", "Makro & Prognose", "Wirtschaftskalender"])

# Übersicht
with tabs[0]:
    st.header("Schnellübersicht")
    val, label = fetch_fear_greed()
    st.metric("Fear & Greed Index", val if val else "n/a", label if label else "")
    st.subheader("Aktuelle Krypto-News")
    for n in fetch_news(10):
        if n["link"]:
            st.markdown(f"- **[{n['source']}]** [{n['title']}]({n['link']})")
        else:
            st.markdown(f"- **[{n['source']}]** {n['title']}")

# Charts
with tabs[1]:
    for symbol in WATCHLIST:
        st.header(f"{symbol}")
        for tf_name, tf in TIMEFRAMES.items():
            st.subheader(f"{tf_name}-Timeframe")
            df = fetch_ohlc(symbol, tf["interval"], tf["period"])
            df = compute_indicators(df)
            if df is None or df.empty:
                st.warning(f"Keine Daten für {symbol} ({tf_name})")
                continue
            plot_chart(df, symbol, mode="EMA")
            plot_rsi_macd(df)
            for alert in detect_signals(df):
                st.warning(alert)
            st.divider()

# Makro
with tabs[2]:
    st.header("Makro-Analyse & Prognosen")
    pred, r2 = predict_macro_trend()
    if pred is None:
        st.info("Makro-Prognose nicht möglich.")
    else:
        trend = "Bullish" if pred > 0.02 else "Bearish" if pred < -0.02 else "Neutral"
        st.metric("Makro Trend", trend, f"R²={r2:.2f}")
    dfm = fetch_macro_data()
    if not dfm.empty:
        st.dataframe(dfm.tail(12).round(4))
    else:
        st.info("Keine Makrodaten verfügbar.")

# Wirtschaftskalender
with tabs[3]:
    st.header("Wichtige Wirtschaftstermine (Monat)")
    events = fetch_econ_calendar()
    if not events:
        st.info("Keine High-Impact-Events gefunden.")
    else:
        for e in events:
            st.markdown(f"- **{e['country']}**: {e['event']} *(Impact: {e['impact']})*")

st.markdown("---")
st.caption("⚠️ Indikative technische Analyse – keine Anlageberatung.")
