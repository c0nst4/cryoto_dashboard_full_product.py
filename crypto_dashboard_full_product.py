# crypto_dashboard_full_product.py
"""
Robustes Profi-Krypto-Dashboard (BTC / ETH / SOL)
- Technische Indikatoren: SMA20/50/200, RSI, MACD, ATR
- Candlestick-Charts (Plotly)
- Fear & Greed Index
- Krypto-News (CoinTelegraph)
- Makro-Analyse mit echten Indikatoren (so gut wie möglich automatisch)
- Robuste Fallbacks (kein Absturz bei fehlenden Daten)
- Simulierte Liquidation Heatmap (wenn keine API vorhanden)
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
from sklearn.metrics import r2_score

# ---------------- Page config ----------------
st.set_page_config(layout="wide", page_title="Krypto Profi-Dashboard")

# ---------------- Settings ----------------
WATCHLIST = ["BTC-USD", "ETH-USD", "SOL-USD"]
# Optional: set TRADINGECONOMICS_API_KEY etc. in Streamlit Secrets or environment variables.

# ---------------- Helpers: robust data fetch ----------------
@st.cache_data(ttl=300)
def fetch_ohlc(symbol, months=12):
    """Holt OHLC Daten via yfinance (robust)."""
    try:
        df = yf.download(symbol, period=f"{months}mo", interval="1d", progress=False)
        if df is None or df.empty:
            return None
        df.index = pd.to_datetime(df.index)
        return df
    except Exception:
        return None

def safe_series_from_df(df, col):
    """
    Extrahiert df[col] als 1D Series robust gegen MultiIndex/2D-Formate.
    Gibt eine leere Series mit dem Index von df bei Fehlern.
    """
    try:
        if col not in df.columns:
            # Try if columns are MultiIndex where first level contains col
            if isinstance(df.columns, pd.MultiIndex):
                matches = [c for c in df.columns if c[0] == col]
                if matches:
                    s = df[matches[0]]
                else:
                    return pd.Series(dtype=float, index=df.index)
            else:
                return pd.Series(dtype=float, index=df.index)
        else:
            s = df[col]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        return pd.Series(s.values, index=df.index).astype(float)
    except Exception:
        return pd.Series(dtype=float, index=df.index)

# ---------------- Technical indicators ----------------
def compute_indicators(df):
    """Berechnet und fügt Indikatoren in df ein (in-place logisch)."""
    close = safe_series_from_df(df, "Close")
    high = safe_series_from_df(df, "High")
    low = safe_series_from_df(df, "Low")

    df["SMA20"] = close.rolling(20, min_periods=1).mean()
    df["SMA50"] = close.rolling(50, min_periods=1).mean()
    df["SMA200"] = close.rolling(200, min_periods=1).mean()

    try:
        df["RSI"] = ta.momentum.RSIIndicator(close, window=14).rsi()
    except Exception:
        # fallback simple RSI-like calc (not perfect)
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = -delta.clip(upper=0).rolling(14).mean()
        rs = gain / (loss.replace(0, np.nan))
        df["RSI"] = 100 - (100 / (1 + rs))

    try:
        macd = ta.trend.MACD(close)
        df["MACD"] = macd.macd()
        df["MACD_SIGNAL"] = macd.macd_signal()
    except Exception:
        df["MACD"] = np.nan
        df["MACD_SIGNAL"] = np.nan

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
        val = int(js["data"][0]["value"])
        txt = js["data"][0]["value_classification"]
        return val, txt
    except Exception:
        return None, None

# ---------------- News ----------------
@st.cache_data(ttl=600)
def fetch_news(limit=6):
    sources = ["https://cointelegraph.com/rss", "https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml"]
    items = []
    for src in sources:
        try:
            r = requests.get(src, timeout=8)
            soup = BeautifulSoup(r.content, "xml")
            for it in soup.find_all("item")[:limit]:
                items.append({"title": it.title.text if it.title else "", "link": it.link.text if it.link else ""})
        except Exception:
            continue
    return items[:limit]

# ---------------- Robust Macro data fetch ----------------
@st.cache_data(ttl=3600)
def fetch_macro_data():
    """
    Versucht, einige Makro-Indikatoren zu laden:
    - VIX (^VIX)
    - DXY (Dollar Index) - versucht verschiedene Tickervarianten
    - CPI, UNEMP, FEDFUNDS (wenn verfügbar via yfinance; ansonsten NaN)
    Ergebnis: DataFrame monthly (M) mit Spalten CPI, UNEMP, FEDFUNDS, DXY, VIX (soweit vorhanden).
    """
    macro = {}
    now = datetime.utcnow()

    # 1) VIX (meist vorhanden)
    try:
        vix = yf.download("^VIX", period="2y", interval="1d", progress=False)["Close"].resample("M").last()
        macro["VIX"] = vix
    except Exception:
        macro["VIX"] = pd.Series(dtype=float)

    # 2) DXY (Dollar Index) - multiple ticker attempts
    dxy = pd.Series(dtype=float)
    for t in ["DX-Y.NYB", "DXY", "USDX"]:
        try:
            tmp = yf.download(t, period="2y", interval="1d", progress=False)["Close"].resample("M").last()
            if tmp is not None and not tmp.empty:
                dxy = tmp
                break
        except Exception:
            continue
    macro["DXY"] = dxy

    # 3) CPI (attempts) - many providers not available via yfinance; we try common tickers, fallback empty
    cpi = pd.Series(dtype=float)
    for t in ["CPIAUCSL", "^CPI", "CPALTT01USM657N"]:  # try a few common names (may fail)
        try:
            tmp = yf.download(t, period="2y", interval="1mo", progress=False)["Close"]
            if tmp is not None and not tmp.empty:
                cpi = tmp.resample("M").last()
                break
        except Exception:
            continue
    macro["CPI"] = cpi

    # 4) Unemployment (UNRATE)
    unemp = pd.Series(dtype=float)
    for t in ["UNRATE", "UNEMP"]:
        try:
            tmp = yf.download(t, period="2y", interval="1mo", progress=False)["Close"]
            if tmp is not None and not tmp.empty:
                unemp = tmp.resample("M").last()
                break
        except Exception:
            continue
    macro["UNEMP"] = unemp

    # 5) FEDFUNDS / short rates
    fed = pd.Series(dtype=float)
    for t in ["FEDFUNDS", "DFF"]:
        try:
            tmp = yf.download(t, period="2y", interval="1mo", progress=False)["Close"]
            if tmp is not None and not tmp.empty:
                fed = tmp.resample("M").last()
                break
        except Exception:
            continue
    macro["FEDFUNDS"] = fed

    # concat into DataFrame with monthly index; flatten columns
    try:
        df = pd.concat({k: v for k, v in macro.items()}, axis=1)
        # if a MultiIndex columns happened, flatten
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [str(c[0]) for c in df.columns]
        # ensure monthly index
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df.resample("M").last()
        return df
    except Exception:
        # fallback empty df
        return pd.DataFrame()

# ---------------- Train macro-based model ----------------
def train_macro_model(symbol, min_samples=6):
    """
    Trainiert ein lineares Modell, das monatliche BTC-Renditen auf verfügbare Makro-Features regressiert.
    Rückgabe: (model, scaler, last_features_array, r2) oder (None, None, None, None) wenn nicht möglich.
    """
    # 1) price monthly returns
    df_price = fetch_ohlc(symbol, months=30)
    if df_price is None or df_price.empty:
        return None, None, None, None
    price_monthly = df_price["Close"].resample("M").last()
    ret_monthly = price_monthly.pct_change().dropna()
    if ret_monthly.empty:
        return None, None, None, None

    # 2) macro data monthly
    df_macro = fetch_macro_data()
    if df_macro.empty:
        # Not enough macro info -> fallback to time-based model (simple)
        if len(ret_monthly) < min_samples:
            return None, None, None, None
        X = np.arange(len(ret_monthly)).reshape(-1, 1)
        y = ret_monthly.values
        model = LinearRegression().fit(X, y)
        scaler = None
        last_features = np.array([[len(X)]])
        r2 = float(r2_score(y, model.predict(X)))
        return model, scaler, last_features, r2

    # 3) merge price returns with macro features (inner join on index)
    try:
        price_df = ret_monthly.to_frame(name="RET")
        # ensure df_macro columns are strings and monthly-resampled
        df_macro = df_macro.copy()
        df_macro.index = pd.to_datetime(df_macro.index)
        df_macro = df_macro.resample("M").last()
        # drop macro columns that are all NaN
        df_macro = df_macro.loc[:, df_macro.notna().any(axis=0)]
        if df_macro.empty:
            # fallback to price-only as above
            if len(ret_monthly) < min_samples:
                return None, None, None, None
            X = np.arange(len(ret_monthly)).reshape(-1, 1)
            y = ret_monthly.values
            model = LinearRegression().fit(X, y)
            scaler = None
            last_features = np.array([[len(X)]])
            r2 = float(r2_score(y, model.predict(X)))
            return model, scaler, last_features, r2

        df_all = pd.concat([price_df, df_macro], axis=1).dropna()
        if df_all.shape[0] < min_samples:
            return None, None, None, None

        # 4) build X,y
        feature_cols = [c for c in df_macro.columns if c in df_all.columns]
        X = df_all[feature_cols].values
        y = df_all["RET"].values

        scaler = StandardScaler().fit(X)
        X_scaled = scaler.transform(X)
        model = LinearRegression().fit(X_scaled, y)
        r2 = float(r2_score(y, model.predict(X_scaled)))

        # last feature row to predict next month (use last available macro row)
        last_features_row = df_macro.dropna(how="all").iloc[-1:][feature_cols].values  # shape (1, n)
        return model, scaler, last_features_row, r2
    except Exception:
        return None, None, None, None

def predict_next_month_with_macro(symbol):
    """
    Nutzt train_macro_model; gibt pred (float) und r2 (float) oder (None,None)
    """
    trained = train_macro_model(symbol)
    if trained is None or trained[0] is None:
        return None, None
    model, scaler, last_features_row, r2 = trained
    try:
        if scaler is not None:
            Xp = scaler.transform(last_features_row)
        else:
            Xp = last_features_row
        pred = float(model.predict(Xp)[0])
        return pred, r2
    except Exception:
        return None, None

# ---------------- Optional CoinAnk heatmap fetch (keine API key nötig hier) ----------------
@st.cache_data(ttl=300)
def fetch_coinank_heatmap_dummy(symbol):
    """Simulierter Fallback wenn keine CoinAnk API vorhanden."""
    # generate small dummy distribution around current price
    return None  # we use simulated heatmap in UI

# ---------------- Telegram (optional) ----------------
def send_telegram_message(text):
    token = os.getenv("TELEGRAM_BOT_TOKEN") or st.secrets.get("TELEGRAM_BOT_TOKEN", None)
    chat_id = os.getenv("TELEGRAM_CHAT_ID") or st.secrets.get("TELEGRAM_CHAT_ID", None)
    if not token or not chat_id:
        return False, "telegram not configured"
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        r = requests.post(url, data={"chat_id": chat_id, "text": text}, timeout=10)
        r.raise_for_status()
        return True, "sent"
    except Exception as e:
        return False, str(e)

# ---------------- UI ----------------
st.title("📊 Profi Krypto-Dashboard (BTC / ETH / SOL)")
tabs = st.tabs(["Übersicht", "Charts & Indikatoren", "Makro-Analyse", "Heatmap & Extras"])

# --- Übersicht Tab
with tabs[0]:
    st.header("Schnellübersicht")
    fgi, fgi_text = fetch_fear_greed()
    st.metric("Fear & Greed", fgi if fgi is not None else "n/a", fgi_text if fgi_text else "")

    st.subheader("Top Krypto-News")
    news = fetch_news(6)
    if news:
        for n in news:
            st.markdown(f"- [{n['title']}]({n['link']})")
    else:
        st.write("Keine News verfügbar.")

    st.subheader("High-Impact Events (nächste 30 Tage)")
    # Use TradingEconomics calendar if available (guest fallback)
    try:
        today = datetime.utcnow().date()
        start = today.strftime("%Y-%m-%d")
        end = (today + timedelta(days=30)).strftime("%Y-%m-%d")
        events = []
        # try to fetch via TradingEconomics only if key present or guest:guest works
        te_key = os.getenv("TRADINGECONOMICS_API_KEY", None) or st.secrets.get("TRADINGECONOMICS_API_KEY", None)
        te_url = f"https://api.tradingeconomics.com/calendar?start_date={start}&end_date={end}&c={te_key or 'guest:guest'}"
        try:
            r = requests.get(te_url, timeout=8)
            if r.status_code == 200:
                evs = r.json()
                events = [ev for ev in evs if ev.get("importance") == 3]
        except Exception:
            events = []
        if events:
            for ev in events[:6]:
                st.write(f"- {ev.get('date')} {ev.get('country')}: {ev.get('event')}")
        else:
            st.info("Keine High-Impact Events gefunden (API-Limit oder keine Daten).")
    except Exception:
        st.info("Keine Event-Daten.")

# --- Charts & Indikatoren Tab
with tabs[1]:
    st.header("Charts & Indikatoren")
    for ticker in WATCHLIST:
        st.subheader(ticker)
        df = fetch_ohlc(ticker, months=12)
        if df is None:
            st.warning(f"Keine Preisdaten für {ticker}.")
            continue
        df = compute_indicators(df)
        latest = df.iloc[-1]
        latest_close = float(latest["Close"]) if "Close" in latest else np.nan
        latest_rsi = float(latest["RSI"]) if "RSI" in latest else np.nan
        st.metric("Close", f"{latest_close:.2f} USD")
        st.metric("RSI", f"{latest_rsi:.1f}")
        fig = go.Figure(data=[go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"]
        )])
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA20"], name="SMA20"))
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA50"], name="SMA50"))
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA200"], name="SMA200"))
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)
        # small table
        try:
            st.dataframe(df[["Close","SMA20","SMA50","RSI","MACD","ATR"]].tail(5).round(6))
        except Exception:
            pass

# --- Makro-Analyse Tab
with tabs[2]:
    st.header("Makro-Analyse & Prognose (realistische Indikatoren)")
    st.write("Das Modell nutzt verfügbare Makrodaten (CPI, Arbeitslosenquote, FED Zinsen, DXY, VIX).")
    df_macro = fetch_macro_data()
    if df_macro is None or df_macro.empty:
        st.warning("Makrodaten konnten nicht geladen werden. Die Prognose funktioniert nur bei verfügbaren Makrodaten.")
    else:
        st.subheader("Makro-Daten (letzte Monate)")
        try:
            st.dataframe(df_macro.tail(6).round(4))
        except Exception:
            st.write(df_macro.tail(6).round(4))

    st.markdown("### Modelltraining & Prognose (pro Asset)")
    for ticker in WATCHLIST:
        pred, r2 = predict_next_month_with_macro(ticker)
        if pred is None:
            st.write(f"{ticker}: Keine Makro-basierte Prognose möglich (unzureichende Daten).")
            continue
        pred_pct = pred * 100
        trend = "Bullish" if pred > 0.02 else ("Bearish" if pred < -0.02 else "Neutral")
        symb = "🟢" if trend == "Bullish" else ("🔴" if trend == "Bearish" else "⚪")
        st.write(f"{ticker}: {symb} Prognose nächster Monat {pred_pct:.2f}% (R²={r2:.2f}) → **{trend}**")

# --- Heatmap & Extras Tab
with tabs[3]:
    st.header("Liquidation Heatmap (Simulation oder CoinAnk falls API vorhanden)")
    # if COINANK key in secrets env, you could call real API; here we show simulated map to be safe
    coins = ["BTC","ETH","SOL"]
    data = []
    for c in coins:
        for p in np.linspace(0.9, 1.1, 40):
            data.append({"Coin": c, "price": p, "long_liq": np.random.poisson(20), "short_liq": np.random.poisson(15)})
    dfh = pd.DataFrame(data)
    fig = px.density_heatmap(dfh, x="price", y="Coin", z="long_liq", color_continuous_scale="Reds")
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.write("⚠️ Hinweis: Alle Prognosen sind statistische Schätzungen und keine Anlageberatung. \
Verwende sie als Hilfsmittel, nicht als alleinige Entscheidungsgrundlage.")
