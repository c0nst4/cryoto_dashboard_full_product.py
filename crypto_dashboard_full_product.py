import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import ta
from datetime import datetime
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# ==============================
# APP SETUP
# ==============================
st.set_page_config(layout="wide", page_title="Krypto Dashboard – BTC/ETH/SOL")

WATCHLIST = ["BTC-USD", "ETH-USD", "SOL-USD"]

# ==============================
# DATA HELPERS
# ==============================
@st.cache_data(ttl=600)
def fetch_ohlc(symbol: str, months: int = 12):
    try:
        df = yf.download(symbol, period=f"{months}mo", interval="1d", progress=False)
        if df.empty:
            return None
        df.index = pd.to_datetime(df.index)
        return df
    except Exception:
        return None

@st.cache_data(ttl=3600)
def fetch_fear_greed():
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
        data = r.json()
        val = int(data["data"][0]["value"])
        label = data["data"][0]["value_classification"]
        return val, label
    except Exception:
        return None, None

@st.cache_data(ttl=900)
def fetch_news(limit=10):
    feeds = [
        ("CoinTelegraph", "https://cointelegraph.com/rss"),
        ("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
        ("Bitcoin.com", "https://news.bitcoin.com/feed/"),
        ("Google", "https://news.google.com/rss/search?q=crypto OR bitcoin OR ethereum&hl=en-US&gl=US&ceid=US:en")
    ]
    out = []
    for src, url in feeds:
        try:
            r = requests.get(url, timeout=10)
            soup = BeautifulSoup(r.content, "xml")
            for item in soup.find_all("item")[: limit // len(feeds)]:
                out.append({
                    "source": src,
                    "title": item.title.text,
                    "link": item.link.text,
                    "date": item.pubDate.text if item.pubDate else ""
                })
        except Exception:
            continue
    return out if out else [{"source": "System", "title": "Keine News gefunden", "link": "", "date": ""}]

# ==============================
# TECHNISCHE INDIKATOREN
# ==============================
def compute_indicators(df):
    close = df["Close"]
    df["SMA20"] = close.rolling(20).mean()
    df["SMA50"] = close.rolling(50).mean()
    df["SMA200"] = close.rolling(200).mean()
    df["EMA20"] = close.ewm(span=20, adjust=False).mean()
    df["EMA50"] = close.ewm(span=50, adjust=False).mean()
    df["EMA200"] = close.ewm(span=200, adjust=False).mean()
    df["RSI"] = ta.momentum.RSIIndicator(close, window=14).rsi()
    macd = ta.trend.MACD(close)
    df["MACD"] = macd.macd()
    df["MACD_SIGNAL"] = macd.macd_signal()
    df["ATR"] = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"], 14).average_true_range()
    return df

def detect_swing_signals(df):
    alerts = []
    if df is None or df.empty:
        return alerts
    last = df.iloc[-1]
    prev = df.iloc[-2]
    if last["EMA20"] > last["EMA50"] and prev["EMA20"] <= prev["EMA50"]:
        alerts.append("🚀 EMA20 kreuzt über EMA50 → Bullish Signal")
    if last["EMA20"] < last["EMA50"] and prev["EMA20"] >= prev["EMA50"]:
        alerts.append("🔻 EMA20 kreuzt unter EMA50 → Bearish Signal")
    if last["RSI"] > 70:
        alerts.append(f"⚠️ RSI überkauft ({last['RSI']:.1f})")
    if last["RSI"] < 30:
        alerts.append(f"⚠️ RSI überverkauft ({last['RSI']:.1f})")
    if last["MACD"] > last["MACD_SIGNAL"] and prev["MACD"] <= prev["MACD_SIGNAL"]:
        alerts.append("📈 MACD Bullish Crossover")
    if last["MACD"] < last["MACD_SIGNAL"] and prev["MACD"] >= prev["MACD_SIGNAL"]:
        alerts.append("📉 MACD Bearish Crossover")
    return alerts

# ==============================
# MAKROANALYSE
# ==============================
@st.cache_data(ttl=3600)
def fetch_macro_data():
    try:
        btc = yf.download("BTC-USD", period="2y", interval="1d")["Close"].resample("M").last().pct_change()
        dxy = yf.download("DX-Y.NYB", period="2y", interval="1d")["Close"].resample("M").last().pct_change()
        vix = yf.download("^VIX", period="2y", interval="1d")["Close"].resample("M").last().pct_change()
        return pd.concat({"BTC": btc, "DXY": dxy, "VIX": vix}, axis=1).dropna()
    except Exception:
        return pd.DataFrame()

def hybrid_forecast(symbol):
    df_macro = fetch_macro_data()
    df = yf.download(symbol, period="1y", interval="1d")["Close"].resample("M").last().pct_change().dropna()
    if df_macro.empty or df.empty:
        return None, None
    merged = pd.concat([df, df_macro], axis=1).dropna()
    y = merged.iloc[:, 0]
    X = merged.iloc[:, 1:]
    model = LinearRegression().fit(X, y)
    r2 = r2_score(y, model.predict(X))
    pred = model.predict([X.iloc[-1]])[0]
    return pred, r2

# ==============================
# WIRTSCHAFTSKALENDER
# ==============================
@st.cache_data(ttl=1800)
def fetch_economic_calendar(limit=10):
    try:
        r = requests.get("https://api.tradingeconomics.com/calendar", timeout=8)
        events = r.json()[:limit]
        return [{"Date": e.get("Date", ""), "Country": e.get("Country", ""), "Event": e.get("Event", "")} for e in events]
    except Exception:
        return [{"Date": "", "Country": "-", "Event": "Keine High-Impact Events gefunden (API-Limit oder keine Daten)."}]

# ==============================
# CHART
# ==============================
def plot_chart(df, ticker, mode="EMA"):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Kurs"))
    for p in (20, 50, 200):
        key = f"{mode}{p}"
        if key in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[key], name=key, line=dict(width=2)))
    fig.update_layout(title=f"{ticker} ({mode})", height=400)
    st.plotly_chart(fig, use_container_width=True)

# ==============================
# UI
# ==============================
st.title("📊 Profi Krypto Dashboard – BTC / ETH / SOL")

tabs = st.tabs(["Übersicht", "Indikatoren", "Makroanalyse", "Wirtschaftskalender"])

# Übersicht
with tabs[0]:
    st.header("Übersicht & Marktstimmung")
    val, label = fetch_fear_greed()
    st.metric("Fear & Greed Index", val if val else "n/a", label if label else "")
    st.subheader("Krypto-News")
    for n in fetch_news(10):
        st.markdown(f"- **[{n['source']}]** [{n['title']}]({n['link']})")

# Indikatoren
with tabs[1]:
    st.header("Technische Indikatoren & Signale")
    for ticker in WATCHLIST:
        df = fetch_ohlc(ticker)
        if df is None:
            st.warning(f"Keine Daten für {ticker}")
            continue
        df = compute_indicators(df)
        plot_chart(df, ticker, "EMA")
        alerts = detect_swing_signals(df)
        if alerts:
            for a in alerts:
                st.warning(a)
        else:
            st.info("Keine akuten Swing-Signale erkannt.")
        st.dataframe(df[["Close", "EMA20", "EMA50", "EMA200", "RSI", "MACD"]].tail(5))

# Makroanalyse
with tabs[2]:
    st.header("Makroanalyse & Preisprognose")
    for ticker in WATCHLIST:
        pred, r2 = hybrid_forecast(ticker)
        if pred is None:
            st.warning(f"{ticker}: Keine Prognose möglich")
            continue
        trend = "Bullish" if pred > 0.02 else ("Bearish" if pred < -0.02 else "Neutral")
        emoji = "🟢" if trend == "Bullish" else ("🔴" if trend == "Bearish" else "⚪")
        st.write(f"{emoji} {ticker}: Erwartete Monatsrendite **{pred*100:.2f}%** (R²={r2:.2f}) → {trend}")

    st.subheader("Makro-Daten (monatliche Returns)")
    dfm = fetch_macro_data()
    st.dataframe(dfm.tail(12).round(4))

# Wirtschaftskalender
with tabs[3]:
    st.header("Wichtige Termine")
    st.dataframe(pd.DataFrame(fetch_economic_calendar(10)))

st.markdown("---")
st.caption("⚠️ Keine Anlageberatung – nur Informationszwecke. Daten ohne Gewähr.")
