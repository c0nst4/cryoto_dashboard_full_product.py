# crypto_dashboard_full_product.py
"""
Vollständiges Krypto-Dashboard (Teil 1/3)
- Multi-TF Indikatoren (4h, D, W)
- Robuste Datenbeschaffung (yfinance + RSS + TradingEconomics fallback)
- Hybrid Forecast: technische + makro Features
- Sicheres Handling von Pandas/ta-Ausgaben (vermeidet 1D-Fehler)
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
from datetime import datetime, timedelta
import time
import math
import typing

# ---------------- Page config ----------------
st.set_page_config(layout="wide", page_title="Krypto Hybrid-Dashboard (BTC/ETH/SOL)")

# ---------------- Settings ----------------
WATCHLIST = ["BTC-USD", "ETH-USD", "SOL-USD"]
# Timeframes definitions
TIMEFRAMES = {
    "4h": {"interval": "4h", "period_days": 90},
    "D":  {"interval": "1d", "period_days": 365},
    "W":  {"interval": "1wk", "period_days": 730}
}
# Minimal data length for forecasts
MIN_DATA_POINTS_FORECAST = 60

# ---------------- Robust helper functions ----------------
def _first_numeric_column_from_df(df: pd.DataFrame) -> pd.Series:
    """Wenn DataFrame mehrere Spalten liefert, gib erste numerische Spalte zurück."""
    if df is None or df.empty:
        return pd.Series(dtype=float)
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            return s
    try:
        return df.iloc[:, 0].astype(float)
    except Exception:
        return pd.Series(dtype=float, index=df.index)

def safe_series_from_df(df: pd.DataFrame, col_name: str) -> pd.Series:
    """
    Robust: extrahiere 1D pd.Series für Spalte col_name.
    Handhabt MultiIndex-Spalten, DataFrame-in-Series, etc.
    """
    if df is None:
        return pd.Series(dtype=float)
    try:
        if col_name in df.columns:
            s = df[col_name]
        else:
            cols = df.columns
            if isinstance(cols, pd.MultiIndex):
                match = None
                for c in cols:
                    if col_name == c[-1] or col_name == c[0] or col_name.lower() in str(c[-1]).lower():
                        match = c
                        break
                s = df[match] if match else pd.Series(dtype=float, index=df.index)
            else:
                match = [c for c in cols if col_name.lower() in str(c).lower()]
                s = df[match[0]] if match else pd.Series(dtype=float, index=df.index)
        if isinstance(s, pd.DataFrame):
            s = _first_numeric_column_from_df(s)
        arr = np.asarray(s)
        if arr.ndim > 1:
            # handle shapes like (n,1)
            arr = arr[:, 0]
        return pd.Series(arr, index=df.index).astype(float)
    except Exception:
        return pd.Series(dtype=float, index=df.index)

def safe_get_latest_value(latest_row, col: str) -> float:
    """Sichere Extraktion eines Wertes aus einer Series/Zeile oder dict. Rückgabe float oder np.nan."""
    try:
        if latest_row is None:
            return np.nan
        if isinstance(latest_row, pd.Series):
            val = latest_row.get(col, np.nan)
        elif isinstance(latest_row, dict):
            val = latest_row.get(col, np.nan)
        else:
            return np.nan
        if isinstance(val, (list, np.ndarray, pd.Series)):
            val = np.array(val).flatten()[-1] if len(val) > 0 else np.nan
        return float(val) if not pd.isna(val) else np.nan
    except Exception:
        return np.nan

# ---------------- OHLC fetching (robust) ----------------
@st.cache_data(ttl=300)
def fetch_ohlc_yf(symbol: str, interval: str, period_days: int) -> typing.Optional[pd.DataFrame]:
    """
    Holt OHLC Daten über yfinance.
    Unterstützt: 4h via 60m resample, daily, weekly.
    """
    try:
        if interval == "4h":
            # yfinance oft keine 4h direkt; hole 60m und resample
            per = f"{max(30, min(period_days, 120))}d"
            df = yf.download(symbol, period=per, interval="60m", progress=False)
            if df is None or df.empty:
                return None
            try:
                df_4h = df.resample("4H").agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"})
                df_4h.dropna(how="any", inplace=True)
                return df_4h
            except Exception:
                return df
        elif interval in ("1d","1wk","1w"):
            if interval in ("1wk","1w"):
                df = yf.download(symbol, period=f"{period_days*2}d", interval="1wk", progress=False)
            else:
                df = yf.download(symbol, period=f"{period_days}d", interval="1d", progress=False)
            if df is None or df.empty:
                return None
            df.index = pd.to_datetime(df.index)
            return df
        else:
            # fallback
            df = yf.download(symbol, period=f"{period_days}d", interval=interval, progress=False)
            if df is None or df.empty:
                return None
            df.index = pd.to_datetime(df.index)
            return df
    except Exception:
        return None

# ---------------- Indicator computations ----------------
def compute_indicators_for_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Auf einem OHLC-DataFrame: berechne SMA/EMA/RSI/MACD/ATR.
    Schreibt Spalten in df und gibt es zurück.
    """
    if df is None or df.empty:
        return df
    close = safe_series_from_df(df, "Close")
    high = safe_series_from_df(df, "High")
    low = safe_series_from_df(df, "Low")
    if close.empty:
        return df

    # SMA
    df["SMA20"] = close.rolling(20, min_periods=1).mean()
    df["SMA50"] = close.rolling(50, min_periods=1).mean()
    df["SMA200"] = close.rolling(200, min_periods=1).mean()
    # EMA
    df["EMA20"] = close.ewm(span=20, adjust=False).mean()
    df["EMA50"] = close.ewm(span=50, adjust=False).mean()
    df["EMA200"] = close.ewm(span=200, adjust=False).mean()
    # RSI
    try:
        df["RSI"] = ta.momentum.RSIIndicator(close, window=14).rsi()
    except Exception:
        delta = close.diff()
        up = delta.clip(lower=0).rolling(14, min_periods=1).mean()
        down = -delta.clip(upper=0).rolling(14, min_periods=1).mean()
        rs = up / down.replace(0, np.nan)
        df["RSI"] = 100 - (100 / (1 + rs))
    # MACD
    try:
        macd = ta.trend.MACD(close)
        df["MACD"] = macd.macd()
        df["MACD_SIGNAL"] = macd.macd_signal()
        df["MACD_DIFF"] = df["MACD"] - df["MACD_SIGNAL"]
    except Exception:
        df["MACD"] = df["MACD_SIGNAL"] = df["MACD_DIFF"] = np.nan
    # ATR
    try:
        atr = ta.volatility.AverageTrueRange(high, low, close, window=14)
        df["ATR"] = atr.average_true_range()
    except Exception:
        df["ATR"] = close.pct_change().rolling(14).std() * close
    return df

# ---------------- Multi-timeframe wrapper ----------------
@st.cache_data(ttl=300)
def get_tf_indicators(symbol: str) -> dict:
    """
    Holt für jede TF OHLC und berechnet Indikatoren.
    Rückgabe dict: {tf: df_with_indicators}
    """
    out = {}
    for tf, meta in TIMEFRAMES.items():
        df = fetch_ohlc_yf(symbol, meta["interval"], meta["period_days"])
        df = compute_indicators_for_df(df) if df is not None else None
        out[tf] = df
    return out

# ---------------- Signals multi-timeframe ----------------
def detect_signals_multi_tf(tf_dfs: dict) -> dict:
    """
    Liefert pro TF Liste von Signal-Strings (EMA/SMA/MACD/RSI).
    """
    res = {}
    for tf, df in tf_dfs.items():
        arr = []
        if df is None or df.empty:
            res[tf] = arr; continue
        need = ["EMA20","EMA50","EMA200","SMA50","SMA200","MACD","MACD_SIGNAL","RSI"]
        sub = df[need].dropna(how="any")
        if sub.shape[0] < 2:
            res[tf] = arr; continue
        prev = sub.iloc[-2]; cur = sub.iloc[-1]
        try:
            # EMA cross
            if prev["EMA20"] < prev["EMA50"] and cur["EMA20"] > cur["EMA50"]:
                arr.append("EMA20↑ über EMA50 (bullish)")
            if prev["EMA20"] > prev["EMA50"] and cur["EMA20"] < cur["EMA50"]:
                arr.append("EMA20↓ unter EMA50 (bearish)")
            # SMA cross 50/200
            if prev["SMA50"] < prev["SMA200"] and cur["SMA50"] > cur["SMA200"]:
                arr.append("SMA50↑ über SMA200 (Golden Cross möglich)")
            if prev["SMA50"] > prev["SMA200"] and cur["SMA50"] < cur["SMA200"]:
                arr.append("SMA50↓ unter SMA200 (Death Cross möglich)")
            # MACD cross
            if prev["MACD"] < prev["MACD_SIGNAL"] and cur["MACD"] > cur["MACD_SIGNAL"]:
                arr.append("MACD kreuzt über Signal (bullish momentum)")
            if prev["MACD"] > prev["MACD_SIGNAL"] and cur["MACD"] < cur["MACD_SIGNAL"]:
                arr.append("MACD kreuzt unter Signal (bearish momentum)")
            # RSI thresholds
            if cur["RSI"] > 70:
                arr.append(f"RSI überkauft ({cur['RSI']:.1f})")
            if cur["RSI"] < 30:
                arr.append(f"RSI überverkauft ({cur['RSI']:.1f})")
        except Exception:
            pass
        res[tf] = arr
    return res

# ---------------- Economic calendar (monthly summary) ----------------
@st.cache_data(ttl=1800)
def fetch_econ_calendar_monthly() -> list:
    """
    Versucht TradingEconomics Calendar (guest:guest). Liefert Liste wichtiger Events im Monat.
    """
    try:
        base = "https://api.tradingeconomics.com/calendar"
        start = datetime.utcnow().replace(day=1).strftime("%Y-%m-%d")
        end = (datetime.utcnow().replace(day=1) + pd.DateOffset(months=1) - pd.DateOffset(days=1)).strftime("%Y-%m-%d")
        params = {"c":"guest:guest","from":start,"to":end}
        r = requests.get(base, params=params, timeout=10)
        if r.status_code != 200:
            return []
        data = r.json()
        # filter high impact
        high = [e for e in data if str(e.get("impact")).lower() in ("high","3","3.0")]
        out = []
        for ev in high:
            out.append({"date": ev.get("date"), "country": ev.get("country"), "event": ev.get("event"), "impact": ev.get("impact")})
        return out
    except Exception:
        return []

# ---------------- Macro features (daily) ----------------
@st.cache_data(ttl=1800)
def fetch_macro_features_daily() -> pd.DataFrame:
    """
    Liefert daily Series: BTC, DXY (Fallback), VIX.
    """
    try:
        btc = yf.download("BTC-USD", period="2y", interval="1d", progress=False)["Close"]
    except Exception:
        btc = pd.Series(dtype=float)
    # DXY fallbacks
    dxy = pd.Series(dtype=float)
    for t in ("DX-Y.NYB","DXY","USDX"):
        try:
            tmp = yf.download(t, period="2y", interval="1d", progress=False)["Close"]
            if tmp is not None and not tmp.empty:
                dxy = tmp; break
        except Exception:
            continue
    if dxy is None or dxy.empty:
        try:
            eurusd = yf.download("EURUSD=X", period="2y", interval="1d", progress=False)["Close"]
            if eurusd is not None and not eurusd.empty:
                dxy = 100 / eurusd
        except Exception:
            dxy = pd.Series(dtype=float)
    try:
        vix = yf.download("^VIX", period="2y", interval="1d", progress=False)["Close"]
    except Exception:
        vix = pd.Series(dtype=float)
    try:
        df = pd.concat({"BTC": btc, "DXY": dxy, "VIX": vix}, axis=1)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df = df.dropna()
        return df
    except Exception:
        return pd.DataFrame()

# ---------------- Part 2 (Forecast features, hybrid model, plotting & chart helpers) ----------------

# ---------------- Feature preparation for hybrid forecast ----------------
@st.cache_data(ttl=600)
def prepare_features_for_hybrid(symbol: str):
    """
    Bereitet Features vor, die technische (multi-TF) und makro (daily) kombinieren.
    Liefert DataFrame mit Datum-Index und Features:
      - recent returns: 1d, 7d, 30d
      - ema/sma slopes: ema20-50 diff, sma50-200 diff (daily)
      - macd diff (daily)
      - rsi (daily)
      - vol14
      - macro: DXY, VIX (daily)
    """
    # price daily
    try:
        price = yf.download(symbol, period="400d", interval="1d", progress=False)["Close"]
    except Exception:
        price = pd.Series(dtype=float)
    if price is None or price.empty:
        return pd.DataFrame()

    df = pd.DataFrame({"Close": price})
    df["RET1"] = df["Close"].pct_change(1)
    df["RET7"] = df["Close"].pct_change(7)
    df["RET30"] = df["Close"].pct_change(30)
    df["VOL14"] = df["RET1"].rolling(14).std()

    # compute daily indicators on df (reuse compute_indicators_for_df on a temp df)
    temp = price.to_frame(name="Close")
    # need Open/High/Low columns for ta functions - approximate by shifting Close
    temp["Open"] = temp["Close"].shift(1).fillna(method="bfill")
    temp["High"] = temp[["Open","Close"]].max(axis=1)
    temp["Low"] = temp[["Open","Close"]].min(axis=1)
    temp = compute_indicators_for_df(temp)
    # slopes and diffs
    if "EMA20" in temp.columns and "EMA50" in temp.columns:
        df["EMA20_50_DIFF"] = (temp["EMA20"] - temp["EMA50"]).reindex(df.index)
    else:
        df["EMA20_50_DIFF"] = np.nan
    if "SMA50" in temp.columns and "SMA200" in temp.columns:
        df["SMA50_200_DIFF"] = (temp["SMA50"] - temp["SMA200"]).reindex(df.index)
    else:
        df["SMA50_200_DIFF"] = np.nan
    # MACD diff and RSI
    df["MACD_DIFF"] = temp["MACD_DIFF"].reindex(df.index) if "MACD_DIFF" in temp.columns else np.nan
    df["RSI"] = temp["RSI"].reindex(df.index) if "RSI" in temp.columns else np.nan

    # macro features
    macro = fetch_macro_features_daily()
    if not macro.empty:
        macro_reindexed = macro.reindex(df.index).ffill().bfill()
        df["DXY"] = macro_reindexed["DXY"]
        df["VIX"] = macro_reindexed["VIX"]
    else:
        df["DXY"] = np.nan; df["VIX"] = np.nan

    df = df.dropna()
    return df

# ---------------- Hybrid model training & prediction ----------------
def train_hybrid_and_predict(symbol: str, horizon_days: int = 30):
    """
    Train a simple linear model using hybrid features to predict cumulative return over horizon_days.
    Returns (predicted_return, r2) or (None,None)
    """
    df = prepare_features_for_hybrid(symbol)
    if df is None or df.empty or len(df) < MIN_DATA_POINTS_FORECAST:
        return None, None

    # target: cumulative return after horizon_days
    df = df.copy()
    df["FUT_PRICE"] = df["Close"].shift(-horizon_days)
    df["FUT_RET"] = df["FUT_PRICE"] / df["Close"] - 1
    df = df.dropna()
    if df.empty or len(df) < 30:
        return None, None

    # features selection (robust)
    candidate_features = ["RET1","RET7","RET30","VOL14","EMA20_50_DIFF","SMA50_200_DIFF","MACD_DIFF","RSI","DXY","VIX"]
    features = [f for f in candidate_features if f in df.columns and df[f].notna().any()]
    if not features:
        return None, None

    X = df[features].values
    y = df["FUT_RET"].values

    # train-test split
    split = max(int(len(X)*0.75), len(X)-30)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]
    if len(X_tr) < 10 or len(X_te) < 5:
        return None, None

    # scaling & linear regression
    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_te_s = scaler.transform(X_te)
    model = LinearRegression().fit(X_tr_s, y_tr)
    try:
        r2 = float(r2_score(y_te, model.predict(X_te_s)))
    except Exception:
        r2 = 0.0

    # predict latest
    last_feat = scaler.transform(df[features].tail(1).values)
    pred = float(model.predict(last_feat)[0])
    return pred, r2

# ---------------- Convenience wrapper: predict horizons (hybrid) ----------------
def predict_horizons_hybrid(symbol: str):
    """
    Returns dict with predictions for day/week/month now & next:
    keys: day_now, day_next, week_now, week_next, month_now, month_next
    Each value is tuple (pred_return, r2) or (None,None)
    """
    out = {}
    # Day: 1 and 2
    out["day_now"], out["day_now_r2"] = train_hybrid_and_predict(symbol, horizon_days=1)
    out["day_next"], out["day_next_r2"] = train_hybrid_and_predict(symbol, horizon_days=2)
    # Week: 7 and 14
    out["week_now"], out["week_now_r2"] = train_hybrid_and_predict(symbol, horizon_days=7)
    out["week_next"], out["week_next_r2"] = train_hybrid_and_predict(symbol, horizon_days=14)
    # Month: 30 and 60
    out["month_now"], out["month_now_r2"] = train_hybrid_and_predict(symbol, horizon_days=30)
    out["month_next"], out["month_next_r2"] = train_hybrid_and_predict(symbol, horizon_days=60)
    return out

# ---------------- Plot helpers: candles + SMA/EMA (EMAs solid + colored) ----------------
def plot_candles_with_lines(df: pd.DataFrame, title: str, sma_periods=(20,50,200), ema_periods=(20,50,200),
                            show_sma=True, show_ema=True):
    """
    Returns a Plotly Figure with candlesticks and SMA/EMA overlays.
    EMAs: EMA20=blue, EMA50=orange, EMA200=red (solid lines).
    """
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"))
    # SMA
    if show_sma:
        for p in sma_periods:
            col = f"SMA{p}"
            if col in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col, line=dict(width=1)))
    # EMA
    if show_ema:
        colors = {20:"blue", 50:"orange", 200:"red"}
        for p in ema_periods:
            col = f"EMA{p}"
            if col in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col, line=dict(color=colors.get(p,"black"), width=2, dash="solid")))
    fig.update_layout(title=title, height=420, margin=dict(l=10,r=10,t=30,b=10))
    return fig

# ---------------- Build signals table ----------------
def build_signals_table(signals_dict: dict) -> pd.DataFrame:
    rows = []
    for tf, arr in signals_dict.items():
        if not arr:
            rows.append({"TF": tf, "Signal": "Keine akuten Signale"})
        else:
            for s in arr:
                rows.append({"TF": tf, "Signal": s})
    return pd.DataFrame(rows)

# ---------------- Render charts & signals for one symbol ----------------
def render_charts_and_signals(symbol: str):
    st.subheader(f"{symbol} — Charts & Signale (Multi-TF)")
    tf_dfs = get_tf_indicators(symbol)
    signals = detect_signals_multi_tf(tf_dfs)
    # Show a compact signals DataFrame
    st.markdown("### Multi-Timeframe Signale")
    st.dataframe(build_signals_table(signals))

    # Tabs for SMA and EMA
    sma_tab, ema_tab = st.tabs(["SMA-Trends", "EMA-Trends"])
    with sma_tab:
        for tf, df in tf_dfs.items():
            st.markdown(f"**{symbol} — {tf} (SMA)**")
            if df is None or df.empty:
                st.write("Keine Daten für diesen Timeframe.")
                continue
            fig = plot_candles_with_lines(df, f"{symbol} {tf} SMA-Chart", show_sma=True, show_ema=False)
            st.plotly_chart(fig, use_container_width=True)
    with ema_tab:
        for tf, df in tf_dfs.items():
            st.markdown(f"**{symbol} — {tf} (EMA)**")
            if df is None or df.empty:
                st.write("Keine Daten für diesen Timeframe.")
                continue
            fig = plot_candles_with_lines(df, f"{symbol} {tf} EMA-Chart", show_sma=False, show_ema=True)
            st.plotly_chart(fig, use_container_width=True)
            # mini metrics: Close, RSI, MACD_DIFF
            try:
                latest = df.dropna(subset=["Close"]).iloc[-1]
                c1, c2, c3 = st.columns(3)
                c1.metric("Close", f"{safe_get_latest_value(latest,'Close'):.2f}")
                c2.metric("RSI", f"{safe_get_latest_value(latest,'RSI'):.1f}")
                c3.metric("MACD_DIFF", f"{safe_get_latest_value(latest,'MACD_DIFF'):.4f}")
            except Exception:
                pass

# ---------------- End of Part 2 ----------------

# ---------------- Part 3 (Final UI: Overview, News, Calendar, Forecast display, main) ----------------

# ---------------- Fear & Greed ----------------
@st.cache_data(ttl=1800)
def fetch_fear_greed():
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=12)
        r.raise_for_status()
        js = r.json()
        val = int(js["data"][0]["value"])
        label = js["data"][0]["value_classification"]
        return val, label
    except Exception:
        return None, None

# ---------------- News aggregator ----------------
@st.cache_data(ttl=600)
def fetch_news_aggregated(limit=12):
    feeds = [
        ("CoinTelegraph", "https://cointelegraph.com/rss"),
        ("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
        ("Investing", "https://www.investing.com/rss/news_25.rss"),
        ("Bitcoin.com", "https://news.bitcoin.com/feed/"),
        ("GoogleCrypto", "https://news.google.com/rss/search?q=crypto OR bitcoin OR ethereum&hl=en-US&gl=US&ceid=US:en")
    ]
    headers = {"User-Agent": "Mozilla/5.0"}
    out = []
    for name, url in feeds:
        try:
            r = requests.get(url, timeout=12, headers=headers)
            if r.status_code != 200:
                continue
            soup = BeautifulSoup(r.content, "xml")
            items = soup.find_all("item")
            for it in items[: max(1, limit // len(feeds))]:
                title = it.title.text if it.title else ""
                link = it.link.text if it.link else ""
                pub = it.pubDate.text if it.pubDate else ""
                out.append({"source": name, "title": title.strip(), "link": link.strip(), "date": pub})
            if len(out) >= limit:
                break
        except Exception:
            continue
    # dedupe
    seen = set(); ded = []
    for it in out:
        key = it.get("title","")[:200]
        if key in seen: continue
        seen.add(key); ded.append(it)
    if not ded:
        return [{"source":"system","title":"Keine News verfügbar — prüfe Internetzugang oder RSS-Feeds","link":"","date":""}]
    return ded[:limit]

# ---------------- UI: overview ----------------
def render_overview():
    st.title("📊 Krypto Hybrid-Dashboard (BTC / ETH / SOL)")
    st.caption("Indikative Signale & hybrid Prognosen (technisch + makro). Keine Anlageberatung.")

    # Top metrics
    fgi_val, fgi_txt = fetch_fear_greed()
    c1, c2, c3 = st.columns(3)
    c1.metric("Fear & Greed", fgi_val if fgi_val is not None else "n/a", fgi_txt or "")
    try:
        btc_last = yf.download("BTC-USD", period="2d", interval="1d", progress=False)["Close"].iloc[-1]
        c2.metric("BTC Last", f"{btc_last:.2f} USD")
    except Exception:
        c2.metric("BTC Last", "n/a")
    try:
        vix_last = yf.download("^VIX", period="2d", interval="1d", progress=False)["Close"].iloc[-1]
        c3.metric("VIX Last", f"{vix_last:.2f}")
    except Exception:
        c3.metric("VIX Last", "n/a")

    # News & calendar
    st.subheader("📰 Top Krypto-News")
    news = fetch_news_aggregated(12)
    for n in news:
        if n.get("link"):
            st.markdown(f"- **[{n['source']}]** [{n['title']}]({n['link']})")
        else:
            st.markdown(f"- **[{n['source']}]** {n['title']}")

    st.subheader("📅 Wichtigste Wirtschaftsereignisse (Monat)")
    events = fetch_econ_calendar_monthly()
    if events:
        for ev in events[:8]:
            date = ev.get("date","")[:10]
            st.markdown(f"- {date} — **{ev.get('country','')}**: {ev.get('event','')} (Impact: {ev.get('impact','')})")
    else:
        st.info("Keine High-Impact Events gefunden oder API-Limit. (TradingEconomics guest fallback)")

# ---------------- UI: Macro & Forecasts ----------------
def render_macro_and_forecasts():
    st.header("🌍 Makro-Analyse & Kombinierte Preisprognosen (Hybrid)")
    st.write("Die Prognosen kombinieren technische (EMA/SMA/MACD/RSI) und makrovariablen (DXY, VIX). Sie sind indikativ.")

    rows = []
    for sym in WATCHLIST:
        preds = predict_horizons_hybrid(sym)
        def fmt(k):
            v = preds.get(k)
            r2 = preds.get(k + "_r2")
            if v is None:
                return "n/a"
            try:
                return f"{v*100:.2f}% (R²={r2:.2f})"
            except Exception:
                return f"{v*100:.2f}%"
        rows.append({
            "Symbol": sym,
            "Day (now)": fmt("day_now"),
            "Day (next)": fmt("day_next"),
            "Week (now)": fmt("week_now"),
            "Week (next)": fmt("week_next"),
            "Month (now)": fmt("month_now"),
            "Month (next)": fmt("month_next")
        })
    try:
        dfp = pd.DataFrame(rows)
        st.dataframe(dfp)
    except Exception:
        for r in rows:
            st.write(r)

    st.markdown("**Hinweis:** 'now' = für die laufende Periode; 'next' = die folgende Periode. Modelle sind einfache lineare Regressionsmodelle auf historischen Daten + Makrovariablen.")

# ---------------- UI: Charts & Signals (calls from Part2) ----------------
def render_all_charts_signals():
    st.header("📈 Charts & Multi-Timeframe Signale")
    for s in WATCHLIST:
        render_charts_and_signals(s)

# ---------------- Tools tab ----------------
def render_tools():
    st.header("🔧 Tools & Hinweise")
    st.markdown("""
    - Datenquellen: yfinance (Kurse), alternative.me (Fear&Greed), TradingEconomics (Kalender guest:guest), mehrere RSS-Feeds für News.  
    - Prognosen sind indikativ. Verwende mehrere Signale (EMA/SMA/MACD/RSI + Makro) bevor du Entscheidungen triffst.  
    - Logs: Bei Problemen öffne Streamlit Cloud → Manage app → Logs und kopiere den Traceback hierher.
    """)
    if st.button("Test: Zeige vorbereitete Hybrid-Features für BTC"):
        df = prepare_features_for_hybrid("BTC-USD")
        if df is None or df.empty:
            st.warning("Keine Feature-Daten verfügbar (zu wenig historische Daten?).")
        else:
            st.dataframe(df.tail(10))

# ---------------- Main app layout ----------------
def main_app():
    tabs = st.tabs(["Übersicht", "Charts & Signale", "Makro & Prognosen", "Tools"])
    with tabs[0]:
        render_overview()
    with tabs[1]:
        render_all_charts_signals()
    with tabs[2]:
        render_macro_and_forecasts()
    with tabs[3]:
        render_tools()
    st.markdown("---")
    st.caption("⚠️ Hinweis: Indikative Signale — keine Anlageberatung. Entwickelt für Analysezwecke.")

# ---------------- Run ----------------
if __name__ == "__main__":
    main_app()
