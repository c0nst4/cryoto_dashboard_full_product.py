# crypto_dashboard_full_product.py
"""
Vollständiges Krypto-Dashboard (BTC/ETH/SOL)
Features:
 - Technische Indikatoren: SMA20/50/200, RSI, MACD, ATR
 - Candlestick-Charts (Plotly)
 - Crypto Fear & Greed Index
 - Krypto-News (CoinTelegraph RSS)
 - Makro-Kalender (TradingEconomics, High-Impact) + Monats-Zusammenfassung
 - Event-Impact-Analyse (letzte 12 Monate): Reaktion des Preises nach Events
 - Einfaches Vorhersagemodell (Lineare Regression) zur Prognose der nächsten ~4 Wochen
 - CoinAnk Liquidation Heatmap (optional, per API-Key)
 - Optional: Telegram Alerts (wenn BOT_TOKEN + CHAT_ID in Secrets gesetzt)
Hinweis: Setze API-Keys in Streamlit Secrets oder Umgebungsvariablen:
 - TRADINGECONOMICS_API_KEY (format 'user:pass' or use guest:guest)
 - COINANK_API_KEY (optional)
 - TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID (optional)
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
import json
import time

# ---------------- CONFIG ----------------
st.set_page_config(layout="wide", page_title="Krypto Profi-Dashboard")

WATCHLIST = ["BTC-USD", "ETH-USD", "SOL-USD"]
COIN_SYMBOLS_FOR_HEATMAP = {"BTC-USD": "BTCUSDT", "ETH-USD": "ETHUSDT", "SOL-USD": "SOLUSDT"}

# read secrets / env
TRADINGECONOMICS_API_KEY = os.getenv("TRADINGECONOMICS_API_KEY") or st.secrets.get("TRADINGECONOMICS_API_KEY", None) or "guest:guest"
COINANK_API_KEY = os.getenv("COINANK_API_KEY") or st.secrets.get("COINANK_API_KEY", None)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or st.secrets.get("TELEGRAM_BOT_TOKEN", None)
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID") or st.secrets.get("TELEGRAM_CHAT_ID", None)

# ---------------- Helpers: Data fetch ----------------
@st.cache_data(ttl=300)
def fetch_ohlc(symbol, months=12):
    """Holt OHLC-Daten via yfinance. Gibt DataFrame oder None."""
    try:
        df = yf.download(symbol, period=f"{months}mo", interval="1d", progress=False)
        if df is None or df.empty:
            return None
        # ensure proper index tz naive / date
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        return None

def safe_series_from_df(df, col):
    """Wandelt df[col] in eine 1D pd.Series mit Index (robust gegen shape issues)."""
    s = pd.Series(df[col].values, index=df.index).astype(float)
    return s

# ---------------- Technical indicators ----------------
def compute_indicators(df):
    """Addiert Indikatoren in-place: SMA20/50/200, RSI, MACD, ATR."""
    close = safe_series_from_df(df, "Close")
    high = safe_series_from_df(df, "High")
    low = safe_series_from_df(df, "Low")

    df["SMA20"] = close.rolling(20, min_periods=1).mean()
    df["SMA50"] = close.rolling(50, min_periods=1).mean()
    df["SMA200"] = close.rolling(200, min_periods=1).mean()

    # RSI
    try:
        df["RSI"] = ta.momentum.RSIIndicator(close, window=14).rsi()
    except Exception:
        # fallback manual RSI
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = -delta.clip(upper=0).rolling(14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    try:
        macd = ta.trend.MACD(close)
        df["MACD"] = macd.macd()
        df["MACD_SIGNAL"] = macd.macd_signal()
    except Exception:
        df["MACD"] = np.nan
        df["MACD_SIGNAL"] = np.nan

    # ATR
    try:
        atr = ta.volatility.AverageTrueRange(high, low, close, window=14)
        df["ATR"] = atr.average_true_range()
    except Exception:
        df["ATR"] = close.pct_change().rolling(14).std() * close  # rough fallback

    return df

# ---------------- Fear & Greed ----------------
@st.cache_data(ttl=300)
def fetch_fear_greed():
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
        js = r.json()
        val = js["data"][0]["value"]
        text = js["data"][0]["value_classification"]
        return int(val), text
    except Exception:
        return None, None

# ---------------- News ----------------
@st.cache_data(ttl=600)
def fetch_news(limit=8):
    sources = [
        "https://cointelegraph.com/rss",
        "https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml"
    ]
    items = []
    for url in sources:
        try:
            r = requests.get(url, timeout=8)
            soup = BeautifulSoup(r.content, "xml")
            for it in soup.find_all("item")[:limit]:
                title = it.title.text if it.title else ""
                link = it.link.text if it.link else ""
                pub = it.pubDate.text if it.pubDate else ""
                items.append({"title": title, "link": link, "pubDate": pub})
        except Exception:
            continue
    return items[:limit]

# ---------------- TradingEconomics Calendar (High-Impact) ----------------
@st.cache_data(ttl=600)
def fetch_tradingeconomics_calendar(start_date, end_date):
    """Holt Events zwischen start_date und end_date. nutzt guest:guest falls kein key."""
    auth = TRADINGECONOMICS_API_KEY or "guest:guest"
    url = f"https://api.tradingeconomics.com/calendar?start_date={start_date}&end_date={end_date}&c={auth}"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()
        # filter high impact (importance==3)
        events = []
        for ev in data:
            if ev.get("importance") == 3:
                events.append(ev)
        return events
    except Exception:
        return []

# ---------------- Event impact analysis ----------------
def compute_event_price_reaction(events, price_series, window_days=7):
    """
    Für eine Liste von events (mit 'date'), berechne die price return from event_date to event_date+window_days.
    price_series: pd.Series indexed by datetime (close prices).
    Gibt DataFrame mit columns ['event_date','country','event','return_1d','return_7d',...,'forecast','actual'].
    """
    rows = []
    for ev in events:
        date_str = ev.get("date")
        try:
            evt_dt = pd.to_datetime(date_str)
        except Exception:
            continue
        # find closest market date >= evt_dt (market may be closed)
        future_idx = price_series.index.searchsorted(evt_dt)
        if future_idx >= len(price_series):
            continue
        idx0 = future_idx
        base_price = price_series.iloc[idx0]
        # compute returns at 1d,3d,7d
        ret = {}
        for d in [1,3,7,14]:
            idxf = idx0 + d
            if idxf < len(price_series):
                ret[f"ret_{d}d"] = (price_series.iloc[idxf] / base_price - 1.0)
            else:
                ret[f"ret_{d}d"] = np.nan
        rows.append({
            "event_date": evt_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "country": ev.get("country"),
            "event": ev.get("event") or ev.get("category") or ev.get("topic"),
            "actual": ev.get("actual"),
            "forecast": ev.get("forecast"),
            **ret
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)

def build_monthly_aggregate_from_events(events_df):
    """
    Aggregiert events_df pro Monat: count_high_impact, avg_surprise (actual-forecast), avg_ret_7d.
    """
    if events_df.empty:
        return pd.DataFrame()
    # convert event_date to datetime
    events_df["event_date_dt"] = pd.to_datetime(events_df["event_date"])
    events_df["year_month"] = events_df["event_date_dt"].dt.to_period("M")
    def surprise(x):
        vals = []
        for a,f in zip(x["actual"], x["forecast"]):
            try:
                if a is not None and f is not None and str(a) != "" and str(f) != "":
                    vals.append(float(a) - float(f))
            except Exception:
                continue
        return np.mean(vals) if vals else 0.0
    grouped = events_df.groupby("year_month").agg(
        count_hi = ("event", "count"),
        avg_surprise = (lambda df: None, lambda x: 0)  # placeholder
    )
    # compute avg_surprise and avg_ret_7d properly
    months = []
    rows = []
    for name, grp in events_df.groupby("year_month"):
        cnt = grp.shape[0]
        # avg surprise
        surprises = []
        for a,f in zip(grp["actual"], grp["forecast"]):
            try:
                if a not in (None,"") and f not in (None,""):
                    surprises.append(float(a)-float(f))
            except Exception:
                continue
        avg_surp = float(np.mean(surprises)) if surprises else 0.0
        # avg ret_7d
        r7 = grp["ret_7d"].dropna()
        avg_r7 = float(r7.mean()) if not r7.empty else 0.0
        rows.append({"year_month": str(name), "count_hi": cnt, "avg_surprise": avg_surp, "avg_ret_7d": avg_r7})
    return pd.DataFrame(rows).set_index("year_month")

# ---------------- Forecast model (monthly) ----------------
def train_monthly_model(symbol):
    """
    Trainiert ein einfaches Modell auf Monatsdaten:
    monthly_return ~ count_hi + avg_surprise + avg_ret_7d (lag)
    Nutzlast: benutzt letzte 12 Monatsdaten wenn möglich.
    Gibt model, X_df, y_series, r2
    """
    # get price series 400d
    df = fetch_ohlc(symbol, months=14)
    if df is None:
        return None, None, None, None
    price = df["Close"]
    # collect events for past 14 months
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=400)
    evs = fetch_tradingeconomics_calendar(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    # compute event reactions
    evdf = compute_event_price_reaction(evs, price, window_days=7)
    # aggregate monthly
    monthly_agg = build_monthly_aggregate_from_events(evdf)
    # build monthly returns
    monthly_close = price.resample("M").last()
    monthly_ret = monthly_close.pct_change().dropna()
    # align months (strings like '2024-09')
    mr = monthly_ret[-12:]
    if mr.empty:
        return None, None, None, None
    X_rows = []
    y = []
    months = []
    for dt, r in mr.items():
        ym = pd.Period(dt, freq='M')
        key = str(ym)
        row = monthly_agg.loc[key] if key in monthly_agg.index else pd.Series({"count_hi":0,"avg_surprise":0.0,"avg_ret_7d":0.0})
        X_rows.append([row.get("count_hi",0), row.get("avg_surprise",0.0), row.get("avg_ret_7d",0.0)])
        y.append(r)
        months.append(key)
    X = np.array(X_rows)
    y = np.array(y)
    if len(y) < 4:
        return None, None, None, None
    model = LinearRegression()
    model.fit(X,y)
    r2 = float(r2_score(y, model.predict(X)))
    X_df = pd.DataFrame(X, columns=["count_hi","avg_surprise","avg_ret_7d"], index=months)
    y_series = pd.Series(y, index=months)
    return model, X_df, y_series, r2

def prepare_next_month_features():
    """Ermittelt High-Impact Events im nächsten Monat und erzeugt feature vector [count_hi, avg_surprise, avg_ret_7d=0 placeholder]."""
    today = datetime.utcnow().date()
    if today.month == 12:
        nm_start = datetime(today.year+1,1,1)
    else:
        nm_start = datetime(today.year, today.month+1, 1)
    if nm_start.month == 12:
        nm_end = datetime(nm_start.year+1, 1, 1) - timedelta(days=1)
    else:
        nm_end = datetime(nm_start.year, nm_start.month+1,1) - timedelta(days=1)
    evs = fetch_tradingeconomics_calendar(nm_start.strftime("%Y-%m-%d"), nm_end.strftime("%Y-%m-%d"))
    count_hi = len(evs)
    surprises = []
    for ev in evs:
        try:
            a = ev.get("actual"); f = ev.get("forecast")
            if a not in (None,"") and f not in (None,""):
                surprises.append(float(a)-float(f))
        except Exception:
            continue
    avg_surp = float(np.mean(surprises)) if surprises else 0.0
    # avg_ret_7d unknown -> set 0 as placeholder (model trained on months with real avg_ret_7d from history)
    return np.array([count_hi, avg_surp, 0.0]), evs, nm_start

# ---------------- CoinAnk heatmap ----------------
@st.cache_data(ttl=300)
def fetch_coinank_heatmap(symbol):
    if not COINANK_API_KEY:
        return []
    url = f"https://api.coinank.com/v1/liquidation/heatmap?symbol={symbol}&timeframe=1d&apikey={COINANK_API_KEY}"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        js = r.json()
        return js.get("heatmap", [])
    except Exception:
        return []

# ---------------- Telegram notifications ----------------
def send_telegram_message(text):
    if not (TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID):
        return False, "Telegram not configured"
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        r = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=10)
        r.raise_for_status()
        return True, "sent"
    except Exception as e:
        return False, str(e)

# ---------------- UI ----------------
st.title("📈 Profi Krypto-Dashboard (BTC / ETH / SOL)")
tabs = st.tabs(["Übersicht", "Charts & Indikatoren", "Makro-Analyse", "Heatmap", "Einstellungen"])

# --- Tab 1: Übersicht
with tabs[0]:
    st.header("Schnellübersicht")
    col1, col2, col3 = st.columns(3)
    # Fear & Greed
    fgi, fgi_text = fetch_fear_greed()
    col1.metric("Fear & Greed", fgi if fgi is not None else "n/a", fgi_text if fgi_text else "")
    # top news
    news = fetch_news(5)
    col2.markdown("**Top Krypto News**")
    if news:
        for n in news[:3]:
            col2.write(f"- {n['title']}" if isinstance(n, dict) else f"- {n}")
    else:
        col2.write("Keine News verfügbar")
    # next macro events
    today = datetime.utcnow().date()
    start = today.strftime("%Y-%m-%d")
    end = (today + timedelta(days=30)).strftime("%Y-%m-%d")
    events_next30 = fetch_tradingeconomics_calendar(start, end)
    col3.markdown("**High-Impact (nächste 30d)**")
    if events_next30:
        for ev in events_next30[:5]:
            col3.write(f"- {ev.get('date')} {ev.get('country')}: {ev.get('event')}")
    else:
        col3.write("Keine High-Impact Events gefunden")

    st.markdown("---")
    st.write("Alerts (live-check):")
    alert_lines = []
    for ticker in WATCHLIST:
        df = fetch_ohlc(ticker, months=6)
        if df is None:
            alert_lines.append(f"{ticker}: Daten fehlen")
            continue
        df = compute_indicators(df)
        latest = df.iloc[-1]
        if latest.get("RSI", np.nan) > 70:
            alert_lines.append(f"{ticker}: RSI überkauft ({latest['RSI']:.1f})")
        if latest.get("RSI", np.nan) < 30:
            alert_lines.append(f"{ticker}: RSI überverkauft ({latest['RSI']:.1f})")
        # SMA50 cross
        if latest["Close"] < latest.get("SMA50", latest["Close"]):
            alert_lines.append(f"{ticker}: Close unter SMA50 ({latest['Close']:.2f} < {latest['SMA50']:.2f})")
    if alert_lines:
        for a in alert_lines:
            st.warning(a)
    else:
        st.success("Keine akuten Alerts gefunden")

# --- Tab 2: Charts & Indikatoren
with tabs[1]:
    st.header("Charts & Indikatoren")
    for ticker in WATCHLIST:
        st.subheader(ticker)
        df = fetch_ohlc(ticker, months=12)
        if df is None:
            st.write("Daten fehlen")
            continue
        df = compute_indicators(df)
        # show key metrics
        latest = df.iloc[-1]
        cols = st.columns(4)
        cols[0].metric("Close", f"{latest['Close']:.2f}")
        cols[1].metric("RSI", f"{latest.get('RSI', np.nan):.1f}")
        cols[2].metric("ATR", f"{latest.get('ATR', np.nan):.4f}")
        cols[3].metric("SMA50", f"{latest.get('SMA50', np.nan):.2f}")
        # candlestick + overlays
        fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Candles")])
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], mode='lines', name='SMA20'))
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], mode='lines', name='SMA50'))
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA200'], mode='lines', name='SMA200'))
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Indicator Table (letzte 5 Tage)**")
        st.dataframe(df[["Close","SMA20","SMA50","RSI","MACD","ATR"]].tail(5))

# --- Tab 3: Makro Analyse & Prognose
with tabs[2]:
    st.header("Makro-Analyse (Historical Event Impact & Forecast)")

    st.markdown("### 1) Historische Event-Reaktionen (letzte 12 Monate)")
    # For each coin build event impact DF (we focus on BTC for macro effect baseline)
    base_symbol = "BTC-USD"
    df_price = fetch_ohlc(base_symbol, months=14)
    if df_price is None:
        st.write("Preisdaten für BTC fehlen; Makro-Analyse nicht möglich")
    else:
        # fetch events last 14 months
        end_date = datetime.utcnow().date()
        start_date = end_date - timedelta(days=400)
        evs = fetch_tradingeconomics_calendar(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        evdf = compute_event_price_reaction(evs, df_price["Close"], window_days=7)
        if evdf.empty:
            st.write("Keine Events / Reaktionen gefunden")
        else:
            st.write("Einige Event-Reaktionen (Beispiel):")
            st.dataframe(evdf[["event_date","country","event","ret_1d","ret_7d"]].sort_values("event_date", ascending=False).head(15))
            # monthly aggregation
            monthly_agg = build_monthly_aggregate_from_events(evdf)
            if not monthly_agg.empty:
                st.write("Monatliche Aggregation (count_hi, avg_surprise, avg_ret_7d):")
                st.dataframe(monthly_agg.tail(12))
            else:
                st.write("Keine Monats-Aggregation möglich")

    st.markdown("### 2) Trainiertes Monatsmodell (letzte 12 Monate)")
    model_info = train_monthly_model(base_symbol)
    if model_info[0] is None:
        st.write("Nicht genügend Daten, um ein Monatsmodell zu trainieren.")
    else:
        model, Xdf, yser, r2 = model_info
        st.write(f"Modell trainiert. R² = {r2:.3f}")
        st.dataframe(pd.concat([Xdf, yser.rename("monthly_return")], axis=1))

        st.markdown("### 3) Prognose für die nächsten ~4 Wochen")
        feat, next_evs, nm_start = prepare_next_month_features()
        st.write(f"Vorhersage Monat ab: {nm_start.strftime('%Y-%m-%d')}. Geplante High-Impact Events: {feat[0]}. Avg surprise: {feat[1]:.3f}")
        # predict
        pred = None
        try:
            pred = model.predict(feat.reshape(1,-1))[0]
            pred_pct = pred*100
            st.write(f"Prognose (vereinfachte lineare Regression): erwartete Rendite nächste ~4 Wochen: {pred_pct:.2f}%")
            if pred > 0.02:
                st.success("Gesamtsignale: Bullish (Positiver Ausblick)")
            elif pred < -0.02:
                st.error("Gesamtsignale: Bearish (Negativer Ausblick)")
            else:
                st.info("Gesamtsignale: Neutral / Seitwärts")
        except Exception as e:
            st.write("Prognose konnte nicht berechnet werden:", str(e))
        if next_evs:
            st.markdown("Geplante High-Impact Events (nächster Monat):")
            st.dataframe(pd.DataFrame(next_evs)[["date","country","event","actual","forecast"]].head(12))

# --- Tab 4: Heatmap
with tabs[3]:
    st.header("Liquidation Heatmap (CoinAnk)")
    if not COINANK_API_KEY:
        st.info("CoinAnk API-Key nicht gesetzt. Setze COINANK_API_KEY in Secrets, um echte Heatmaps zu sehen.")
        # show dummy example
        st.write("Beispiel Heatmap (Dummy):")
        coins = ["BTC","ETH","SOL"]
        price_levels = np.linspace(0.9,1.1,40)
        data = []
        for c in coins:
            for p in price_levels:
                data.append({"Coin": c, "price": p, "long_liq": np.random.poisson(20), "short_liq": np.random.poisson(15)})
        dfh = pd.DataFrame(data)
        fig_long = px.density_heatmap(dfh, x="price", y="Coin", z="long_liq", color_continuous_scale="Reds")
        st.plotly_chart(fig_long, use_container_width=True)
    else:
        for t in WATCHLIST:
            sym = COIN_SYMBOLS_FOR_HEATMAP.get(t, None)
            st.subheader(t.replace("-USD",""))
            hm = fetch_coinank_heatmap(sym)
            if not hm:
                st.write("Keine Heatmap-Daten oder API-Fehler")
            else:
                dfhm = pd.DataFrame(hm)
                fig_long = px.imshow([dfhm['long_liq'].values], x=dfhm['price'].values, y=[t.replace("-USD","")], aspect="auto", color_continuous_scale="Reds", title=f"{t} Longs")
                fig_short = px.imshow([dfhm['short_liq'].values], x=dfhm['price'].values, y=[t.replace("-USD","")], aspect="auto", color_continuous_scale="Blues", title=f"{t} Shorts")
                st.plotly_chart(fig_long, use_container_width=True)
                st.plotly_chart(fig_short, use_container_width=True)

# --- Tab 5: Settings / Alerts
with tabs[4]:
    st.header("Einstellungen & Alerts")
    st.write("API-Keys werden aus Streamlit Secrets oder Umgebungsvariablen gelesen.")
    st.write("Setze TRADINGECONOMICS_API_KEY, COINANK_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID in Settings -> Secrets.")

    st.markdown("### Alerts manuell prüfen & senden")
    if st.button("Alerts prüfen & Telegram (falls konfiguriert)"):
        lines = []
        for ticker in WATCHLIST:
            d = fetch_ohlc(ticker, months=6)
            if d is None:
                lines.append(f"{ticker}: Daten fehlen")
                continue
            d = compute_indicators(d)
            lt = d.iloc[-1]
            if lt.get("RSI", np.nan) > 70:
                lines.append(f"{ticker}: RSI überkauft ({lt['RSI']:.1f})")
            if lt.get("RSI", np.nan) < 30:
                lines.append(f"{ticker}: RSI überverkauft ({lt['RSI']:.1f})")
        if not lines:
            st.success("Keine Alerts")
        else:
            st.warning("\n".join(lines))
            ok, msg = send_telegram_message("\n".join(lines))
            if ok:
                st.success("Telegram Nachricht gesendet")
            else:
                st.info(f"Telegram nicht gesendet: {msg}")

st.markdown("---")
st.write("⚠️ Hinweis: Prognosen sind nur statistische Hinweise und keine Anlageberatung.")
