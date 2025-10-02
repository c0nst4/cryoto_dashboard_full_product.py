# crypto_dashboard_full_product.py
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

# =================== Einstellungen ===================
WATCHLIST = ["BTC-USD", "ETH-USD", "SOL-USD"]
COINANK_API_KEY = os.getenv("COINANK_API_KEY")  # optional
TRADINGECONOMICS_API_KEY = os.getenv("TRADINGECONOMICS_API_KEY") or "guest:guest"
# =====================================================

st.set_page_config(layout="wide", page_title="Krypto Dashboard")
st.title("📊 Krypto Dashboard: BTC / ETH / SOL")
st.caption("Technische Signale, Charts, Fear & Greed, News, Makro-Daten, Prognose & Heatmap")

# ----------------- Technische Signale -----------------
def get_technical_signals(ticker, period_months=12):
    df = yf.download(ticker, period=f"{period_months}mo", interval="1d", progress=False)
    if df.empty:
        return None, None, ["Daten fehlen"]

    close = df["Close"].squeeze().astype(float)

    df["SMA20"] = close.rolling(20).mean()
    df["SMA50"] = close.rolling(50).mean()
    df["SMA200"] = close.rolling(200).mean()
    df["RSI"] = ta.momentum.RSIIndicator(close, window=14).rsi()
    macd = ta.trend.MACD(close)
    df["MACD"] = macd.macd()
    df["ATR"] = ta.volatility.AverageTrueRange(df["High"], df["Low"], close).average_true_range()

    latest = df.iloc[-1]
    alerts = []
    if latest["RSI"] > 70:
        alerts.append("RSI überkauft")
    elif latest["RSI"] < 30:
        alerts.append("RSI überverkauft")
    if latest["Close"] > latest["SMA50"]:
        alerts.append("über SMA50")
    else:
        alerts.append("unter SMA50")

    return df, latest, alerts

# ----------------- Fear & Greed -----------------
def get_fear_greed():
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
        data = r.json()
        return data["data"][0]["value"], data["data"][0]["value_classification"]
    except:
        return None, None

# ----------------- News -----------------
def get_news():
    try:
        r = requests.get("https://cointelegraph.com/rss", timeout=10)
        soup = BeautifulSoup(r.content, "xml")
        items = soup.find_all("item")[:6]
        return [item.title.text for item in items]
    except:
        return []

# ----------------- Charts -----------------
def plot_chart(df, ticker):
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name="Candles"
    )])
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], mode='lines', name='SMA20'))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], mode='lines', name='SMA50'))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA200'], mode='lines', name='SMA200'))
    fig.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

# ----------------- Makro-Daten -----------------
def get_macro_events(start_date, end_date):
    try:
        url = f"https://api.tradingeconomics.com/calendar?start_date={start_date}&end_date={end_date}&c={TRADINGECONOMICS_API_KEY}"
        r = requests.get(url, timeout=10)
        data = r.json()
        return [ev for ev in data if ev.get("importance") == 3]
    except:
        return []

def forecast_next_month(symbol):
    df = yf.download(symbol, period="12mo", interval="1d", progress=False)
    if df.empty:
        return None, None
    monthly = df['Close'].resample('M').last()
    monthly_returns = monthly.pct_change().dropna()
    if len(monthly_returns) < 3:
        return None, None
    X = np.arange(len(monthly_returns)).reshape(-1,1)
    y = monthly_returns.values
    model = LinearRegression()
    model.fit(X,y)
    pred = model.predict([[len(X)]])[0]
    return pred, model.score(X,y)

# ----------------- CoinAnk Heatmap -----------------
def get_coinank_heatmap(symbol):
    if not COINANK_API_KEY:
        return []
    url = f"https://api.coinank.com/v1/liquidation/heatmap?symbol={symbol}&timeframe=1d&apikey={COINANK_API_KEY}"
    try:
        r = requests.get(url, timeout=10)
        return r.json().get("heatmap", [])
    except:
        return []

def plot_coinank_heatmap(heatmap_data, coin_name):
    if not heatmap_data:
        st.write(f"Keine Heatmap-Daten für {coin_name}")
        return
    df = pd.DataFrame(heatmap_data)
    fig_long = px.imshow([df['long_liq'].values], x=df['price'].values, y=[coin_name],
                         aspect="auto", color_continuous_scale="Reds", title=f"{coin_name} Long Liqs")
    fig_short = px.imshow([df['short_liq'].values], x=df['price'].values, y=[coin_name],
                          aspect="auto", color_continuous_scale="Blues", title=f"{coin_name} Short Liqs")
    st.plotly_chart(fig_long, use_container_width=True)
    st.plotly_chart(fig_short, use_container_width=True)

# ----------------- Layout -----------------
st.header("📈 Technische Signale & Charts")
for ticker in WATCHLIST:
    df, latest, alerts = get_technical_signals(ticker)
    if df is not None:
        st.subheader(ticker)
        st.write(f"Close: {latest['Close']:.2f}$ | RSI: {latest['RSI']:.1f}")
        st.write("Alerts: " + ", ".join(alerts))
        plot_chart(df, ticker)
    else:
        st.write(f"⚠️ Keine Daten für {ticker}")

st.header("📊 Fear & Greed Index")
fgi, fgi_text = get_fear_greed()
if fgi:
    st.metric("Fear & Greed", fgi, fgi_text)
else:
    st.write("Keine Daten verfügbar")

st.header("📰 Krypto News")
news = get_news()
if news:
    for n in news:
        st.write("- " + n)
else:
    st.write("Keine News verfügbar")

st.header("📅 Makro & Prognose")
today = datetime.utcnow()
start = today.strftime("%Y-%m-%d")
end = (today + timedelta(days=30)).strftime("%Y-%m-%d")
events = get_macro_events(start, end)
if events:
    st.write("High-Impact Events (nächste 30 Tage):")
    for ev in events[:5]:
        st.write(f"- {ev.get('date')} {ev.get('country')}: {ev.get('event')} (Forecast: {ev.get('forecast')})")
else:
    st.write("Keine Makro-Daten verfügbar")

for ticker in WATCHLIST:
    pred, r2 = forecast_next_month(ticker)
    if pred is not None:
        st.write(f"{ticker}: Prognose nächster Monat = {pred*100:.2f}% (R²={r2:.2f})")
    else:
        st.write(f"{ticker}: Keine Prognose möglich")

st.header("🔥 CoinAnk Liquidation Heatmap")
for coin in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
    st.subheader(coin.replace("USDT",""))
    heatmap_data = get_coinank_heatmap(coin)
    plot_coinank_heatmap(heatmap_data, coin.replace("USDT",""))

st.markdown("---")
st.write("⚠️ Prognosen und Signale sind nur Hilfen, keine Anlageberatung.")
