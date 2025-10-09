# crypto_dashboard_full_product.py
"""
Robustes Krypto-Dashboard (BTC / ETH / SOL)
- Technische Indikatoren: SMA, EMA, RSI, MACD, ATR
- RSI & MACD Subplots
- Multi-TF (4h, D, W) via yfinance (60m -> 4h resample)
- News + Fear&Greed
- Wirtschaftskalender (TradingEconomics fallback)
- Hybride Prognosen (technisch + makro) für Tag/Woche/Monat (robust gegen fehlende Daten)
Hinweis: Indikativ, keine Anlageberatung.
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
import traceback

# -------------------------
# Einstellungen
# -------------------------
st.set_page_config(layout="wide", page_title="Krypto Dashboard (robust)")
WATCHLIST = ["BTC-USD", "ETH-USD", "SOL-USD"]
MIN_DATA_FOR_MODEL = 30   # minimales Minimum (robust)
CACHE_SHORT = 300
CACHE_MED = 900
CACHE_LONG = 3600

# -------------------------
# Hilfsfunktionen (robust)
# -------------------------
def _ensure_series_1d(s):
    """Gibt eine 1D pandas Series zurück (oder leere Series)."""
    if s is None:
        return pd.Series(dtype=float)
    if isinstance(s, pd.Series):
        return s.astype(float)
    if isinstance(s, pd.DataFrame):
        # take first numeric column
        for c in s.columns:
            if pd.api.types.is_numeric_dtype(s[c]):
                return s[c].astype(float)
        return s.iloc[:, 0].astype(float)
    # array-like
    arr = np.asarray(s)
    if arr.ndim > 1:
        arr = arr.ravel()
    return pd.Series(arr).astype(float)

def safe_last_valid(df, col):
    try:
        if df is None or df.empty:
            return np.nan
        if col not in df.columns:
            return np.nan
        s = _ensure_series_1d(df[col]).dropna()
        if s.empty:
            return np.nan
        return float(s.iloc[-1])
    except Exception:
        return np.nan

# -------------------------
# Data fetching (cached)
# -------------------------
@st.cache_data(ttl=CACHE_SHORT)
def fetch_ohlc(symbol: str, months: int = 12, interval: str = "1d"):
    """Hole OHLC über yfinance; für 4h holen wir 60m und resamplen."""
    try:
        if interval == "4h":
            per = f"{max(30, min(months*30, 120))}d"
            df = yf.download(symbol, period=per, interval="60m", progress=False)
            if df is None or df.empty:
                return None
            df4h = df.resample("4H").agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"})
            df4h.dropna(how="any", inplace=True)
            if df4h.empty:
                return None
            df4h.index = pd.to_datetime(df4h.index)
            return df4h
        else:
            # daily or weekly
            if interval in ("1wk","1w"):
                df = yf.download(symbol, period=f"{months*2}d", interval="1wk", progress=False)
            else:
                df = yf.download(symbol, period=f"{months}mo", interval=interval, progress=False)
            if df is None or df.empty:
                return None
            df.index = pd.to_datetime(df.index)
            return df
    except Exception:
        return None

@st.cache_data(ttl=CACHE_MED)
def fetch_fear_greed():
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=8)
        r.raise_for_status()
        js = r.json()
        val = int(js["data"][0]["value"])
        cls = js["data"][0]["value_classification"]
        return val, cls
    except Exception:
        return None, None

@st.cache_data(ttl=CACHE_MED)
def fetch_news(limit=10):
    feeds = [
        ("CoinTelegraph","https://cointelegraph.com/rss"),
        ("CoinDesk","https://www.coindesk.com/arc/outboundfeeds/rss/"),
        ("Bitcoin.com","https://news.bitcoin.com/feed/"),
        ("Google","https://news.google.com/rss/search?q=crypto OR bitcoin OR ethereum&hl=en-US&gl=US&ceid=US:en")
    ]
    out = []
    headers = {"User-Agent":"Mozilla/5.0"}
    for name, url in feeds:
        try:
            r = requests.get(url, timeout=8, headers=headers)
            if r.status_code != 200:
                continue
            soup = BeautifulSoup(r.content, "xml")
            items = soup.find_all("item")
            for it in items[: max(1, limit // len(feeds))]:
                title = it.title.text if it.title else ""
                link = it.link.text if it.link else ""
                date = it.pubDate.text if it.pubDate else ""
                out.append({"source": name, "title": title.strip(), "link": link.strip(), "date": date})
            if len(out) >= limit:
                break
        except Exception:
            continue
    if not out:
        return [{"source":"system","title":"Keine News verfügbar","link":"","date":""}]
    # dedupe by title
    seen = set(); ded = []
    for it in out:
        key = it.get("title","")[:200]
        if key in seen: continue
        seen.add(key); ded.append(it)
    return ded[:limit]

@st.cache_data(ttl=CACHE_LONG)
def fetch_macro_daily():
    """Hole tägliche Zeitreihen: BTC, DXY (Fallback), VIX"""
    try:
        btc = yf.download("BTC-USD", period="3y", interval="1d", progress=False)["Close"]
    except Exception:
        btc = pd.Series(dtype=float)
    # DXY fallbacks
    dxy = pd.Series(dtype=float)
    for t in ("DX-Y.NYB","DXY","USDX"):
        try:
            tmp = yf.download(t, period="3y", interval="1d", progress=False)["Close"]
            if tmp is not None and not tmp.empty:
                dxy = tmp; break
        except Exception:
            continue
    # fallback to EURUSD inversion
    if dxy is None or dxy.empty:
        try:
            eur = yf.download("EURUSD=X", period="3y", interval="1d", progress=False)["Close"]
            if eur is not None and not eur.empty:
                dxy = 100 / eur
        except Exception:
            dxy = pd.Series(dtype=float)
    try:
        vix = yf.download("^VIX", period="3y", interval="1d", progress=False)["Close"]
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

@st.cache_data(ttl=CACHE_LONG)
def fetch_macro_monthly_returns():
    daily = fetch_macro_daily()
    if daily is None or daily.empty:
        return pd.DataFrame()
    try:
        monthly = daily.resample("M").last().pct_change().dropna()
        return monthly
    except Exception:
        return pd.DataFrame()

# -------------------------
# Indikatoren (robust)
# -------------------------
def compute_indicators(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Berechnet und fügt SMA, EMA, RSI, MACD, ATR hinzu.
    Liefert neues DataFrame (kopiert), behandelt fehlende Werte robust.
    """
    if df_in is None or df_in.empty:
        return df_in
    df = df_in.copy()
    try:
        close = _ensure_series_1d(df["Close"])
    except Exception:
        return df
    # SMA / EMA
    df["SMA20"] = close.rolling(20, min_periods=1).mean()
    df["SMA50"] = close.rolling(50, min_periods=1).mean()
    df["SMA200"] = close.rolling(200, min_periods=1).mean()
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
        high = _ensure_series_1d(df.get("High"))
        low = _ensure_series_1d(df.get("Low"))
        atr = ta.volatility.AverageTrueRange(high, low, close, window=14)
        df["ATR"] = atr.average_true_range()
    except Exception:
        df["ATR"] = close.pct_change().rolling(14).std() * close
    return df

# -------------------------
# Signale (robust)
# -------------------------
def detect_swing_signals(df: pd.DataFrame) -> list:
    alerts = []
    if df is None or df.empty or len(df) < 2:
        return alerts
    # use last two valid rows for columns of interest
    cols = ["EMA20","EMA50","EMA200","SMA50","SMA200","MACD","MACD_SIGNAL","RSI"]
    sub = df[cols].dropna(how="any")
    if len(sub) < 2:
        return alerts
    prev = sub.iloc[-2]; cur = sub.iloc[-1]
    try:
        # EMA20/EMA50
        if prev["EMA20"] < prev["EMA50"] and cur["EMA20"] > cur["EMA50"]:
            alerts.append("🚀 EMA20 kreuzt über EMA50 → mögliches bullishes Signal")
        if prev["EMA20"] > prev["EMA50"] and cur["EMA20"] < cur["EMA50"]:
            alerts.append("🔻 EMA20 kreuzt unter EMA50 → mögliches bearishes Signal")
        # SMA 50/200
        if prev["SMA50"] < prev["SMA200"] and cur["SMA50"] > cur["SMA200"]:
            alerts.append("📈 SMA50 kreuzt über SMA200 → mittelfristig Bullish")
        if prev["SMA50"] > prev["SMA200"] and cur["SMA50"] < cur["SMA200"]:
            alerts.append("📉 SMA50 kreuzt unter SMA200 → mittelfristig Bearish")
        # MACD cross
        if prev["MACD"] < prev["MACD_SIGNAL"] and cur["MACD"] > cur["MACD_SIGNAL"]:
            alerts.append("📈 MACD kreuzt über Signal → Momentum bullish")
        if prev["MACD"] > prev["MACD_SIGNAL"] and cur["MACD"] < cur["MACD_SIGNAL"]:
            alerts.append("📉 MACD kreuzt unter Signal → Momentum bearish")
        # RSI
        if cur["RSI"] > 70:
            alerts.append(f"⚠️ RSI überkauft ({cur['RSI']:.1f})")
        if cur["RSI"] < 30:
            alerts.append(f"⚠️ RSI überverkauft ({cur['RSI']:.1f})")
    except Exception:
        pass
    return alerts

# -------------------------
# Hybrid Forecast (robust, simpler)
# -------------------------
def prepare_features_daily(symbol: str, days: int = 400) -> pd.DataFrame:
    """Bereite tägliche Features vor (returns, vol, ema/sma diffs, macd_diff, rsi, DXY, VIX)."""
    try:
        price = yf.download(symbol, period=f"{days}d", interval="1d", progress=False)["Close"]
    except Exception:
        price = pd.Series(dtype=float)
    if price is None or price.empty:
        return pd.DataFrame()
    df = pd.DataFrame({"Close": price})
    df["RET1"] = df["Close"].pct_change(1)
    df["RET7"] = df["Close"].pct_change(7)
    df["VOL14"] = df["RET1"].rolling(14).std()
    # temporary OHLC-like df to compute indicators
    temp = df["Close"].to_frame()
    temp["Open"] = temp["Close"].shift(1).fillna(method="bfill")
    temp["High"] = temp[["Open","Close"]].max(axis=1)
    temp["Low"] = temp[["Open","Close"]].min(axis=1)
    temp = compute_indicators(temp)
    df["EMA20_50"] = (temp["EMA20"] - temp["EMA50"]).reindex(df.index)
    df["SMA50_200"] = (temp["SMA50"] - temp["SMA200"]).reindex(df.index)
    df["MACD_DIFF"] = temp["MACD_DIFF"].reindex(df.index)
    df["RSI"] = temp["RSI"].reindex(df.index)
    # macro
    macro_daily = fetch_macro_daily()
    if not macro_daily.empty:
        macro_reind = macro_daily.reindex(df.index).ffill().bfill()
        df["DXY"] = macro_reind["DXY"]
        df["VIX"] = macro_reind["VIX"]
    else:
        df["DXY"] = np.nan; df["VIX"] = np.nan
    df = df.dropna()
    return df

def train_and_predict_hybrid(symbol: str, horizon_days: int = 30):
    """
    Train a simple linear model on hybrid features to predict cumulative return over horizon_days.
    Returns pred_return (float) and r2 (float) or (None,None)
    """
    df = prepare_features_daily(symbol)
    if df.empty or len(df) < MIN_DATA_FOR_MODEL:
        return None, None
    df = df.copy()
    df["FUT"] = df["Close"].shift(-horizon_days)
    df["FUT_RET"] = df["FUT"] / df["Close"] - 1
    df = df.dropna()
    if df.empty or len(df) < MIN_DATA_FOR_MODEL//2:
        return None, None
    features = ["RET1","RET7","VOL14","EMA20_50","SMA50_200","MACD_DIFF","RSI","DXY","VIX"]
    # keep only available features
    feats = [f for f in features if f in df.columns and df[f].notna().any()]
    if not feats:
        return None, None
    X = df[feats].values
    y = df["FUT_RET"].values
    if len(X) < 20:
        return None, None
    # split
    split = int(len(X)*0.75)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]
    # scaling & training
    try:
        scaler = StandardScaler().fit(X_tr)
        X_tr_s = scaler.transform(X_tr)
        X_te_s = scaler.transform(X_te)
        model = LinearRegression().fit(X_tr_s, y_tr)
        r2 = float(r2_score(y_te, model.predict(X_te_s))) if len(y_te)>1 else 0.0
        last_feat = scaler.transform(df[feats].tail(1).values)
        pred = float(model.predict(last_feat)[0])
        return pred, r2
    except Exception:
        return None, None

def predict_horizons(symbol: str):
    """Convenience wrapper: returns dict with day/week/month now & next."""
    out = {}
    # day
    out["day_now"], out["day_now_r2"] = train_and_predict_hybrid(symbol, horizon_days=1)
    out["day_next"], out["day_next_r2"] = train_and_predict_hybrid(symbol, horizon_days=2)
    # week
    out["week_now"], out["week_now_r2"] = train_and_predict_hybrid(symbol, horizon_days=7)
    out["week_next"], out["week_next_r2"] = train_and_predict_hybrid(symbol, horizon_days=14)
    # month
    out["month_now"], out["month_now_r2"] = train_and_predict_hybrid(symbol, horizon_days=30)
    out["month_next"], out["month_next_r2"] = train_and_predict_hybrid(symbol, horizon_days=60)
    return out

# -------------------------
# Plotting: Candles + RSI + MACD subplots
# -------------------------
def plot_with_subplots(df: pd.DataFrame, title: str, show_sma=True, show_ema=True):
    if df is None or df.empty:
        st.write("Keine Daten für Chart.")
        return
    # main candlestick
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"))
    if show_sma:
        for p in (20,50,200):
            col = f"SMA{p}"
            if col in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col, line=dict(width=1)))
    if show_ema:
        cmap = {20:"blue",50:"orange",200:"red"}
        for p in (20,50,200):
            col = f"EMA{p}"
            if col in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col, line=dict(color=cmap.get(p,"black"), width=2)))
    fig.update_layout(title=title, height=400, margin=dict(l=10,r=10,t=30,b=10))
    st.plotly_chart(fig, use_container_width=True)
    # RSI
    if "RSI" in df.columns:
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI", line=dict(color="gold", width=2)))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
        fig_rsi.update_layout(title="RSI (14)", height=200)
        st.plotly_chart(fig_rsi, use_container_width=True)
    # MACD
    if "MACD" in df.columns:
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", line=dict(color="green", width=2)))
        if "MACD_SIGNAL" in df.columns:
            fig_macd.add_trace(go.Scatter(x=df.index, y=df["MACD_SIGNAL"], name="Signal", line=dict(color="red", width=1)))
        if "MACD_DIFF" in df.columns:
            fig_macd.add_trace(go.Bar(x=df.index, y=df["MACD_DIFF"], name="Hist", opacity=0.3))
        fig_macd.update_layout(title="MACD", height=220)
        st.plotly_chart(fig_macd, use_container_width=True)

# -------------------------
# Economic calendar (fallback)
# -------------------------
@st.cache_data(ttl=CACHE_MED)
def fetch_economic_calendar(limit=10):
    # try tradingeconomics
    try:
        r = requests.get("https://api.tradingeconomics.com/calendar", timeout=8)
        if r.status_code == 200:
            js = r.json()
            out = []
            for e in js[:limit]:
                out.append({
                    "date": e.get("date","")[:10],
                    "country": e.get("country",""),
                    "event": e.get("event",""),
                    "impact": e.get("impact","")
                })
            return out
    except Exception:
        pass
    # fallback: simple investing.com scrape (best effort)
    try:
        r = requests.get("https://www.investing.com/economic-calendar/", headers={"User-Agent":"Mozilla/5.0"}, timeout=8)
        soup = BeautifulSoup(r.text, "html.parser")
        rows = soup.select("tr[eventid]")[:limit]
        out = []
        for tr in rows:
            txt = tr.get_text(separator=" | ", strip=True)
            out.append({"date":"", "country":"", "event": txt, "impact":""})
        return out
    except Exception:
        return [{"date":"","country":"","event":"Keine High-Impact Events gefunden (API-Limit oder keine Daten).","impact":""}]

# -------------------------
# UI (Main)
# -------------------------
def main_app():
    st.title("📊 Krypto Profi-Dashboard — robust (BTC/ETH/SOL)")
    st.caption("Indikative Signale & Prognosen (technisch + makro). Keine Anlageberatung.")

    tabs = st.tabs(["Übersicht", "Charts & Indikatoren", "Makro & Prognosen", "Wirtschaftskalender", "Signale"])
    # Overview
    with tabs[0]:
        st.header("Übersicht")
        fg_val, fg_txt = fetch_fear_greed()
        st.metric("Fear & Greed", fg_val if fg_val is not None else "n/a", fg_txt or "")
        # top news
        st.subheader("Top News")
        news = fetch_news(8)
        for n in news:
            if n.get("link"):
                st.markdown(f"- **[{n['source']}]** [{n['title']}]({n['link']})")
            else:
                st.markdown(f"- **[{n['source']}]** {n['title']}")

    # Charts & Indicators
    with tabs[1]:
        st.header("Charts & Indikatoren (Multi-TF)")
        for sym in WATCHLIST:
            st.subheader(sym)
            # show daily main
            df_daily = fetch_ohlc(sym, months=14, interval="1d")
            if df_daily is None:
                st.warning(f"Keine Kursdaten für {sym}")
                continue
            df_daily = compute_indicators(df_daily)
            plot_with_subplots(df_daily, f"{sym} — Daily", show_sma=True, show_ema=True)
            st.dataframe(df_daily[["Close","EMA20","EMA50","EMA200","RSI","MACD"]].tail(5))

    # Macro & Forecasts
    with tabs[2]:
        st.header("Makro & Prognosen (Hybrid)")
        rows = []
        for sym in WATCHLIST:
            preds = predict_horizons(sym)
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
                "Day now": fmt("day_now"),
                "Day next": fmt("day_next"),
                "Week now": fmt("week_now"),
                "Week next": fmt("week_next"),
                "Month now": fmt("month_now"),
                "Month next": fmt("month_next")
            })
        st.dataframe(pd.DataFrame(rows))

        st.subheader("Monatliche Makrodaten (letzte 12)")
        mm = fetch_macro_monthly_returns()
        if mm is None or mm.empty:
            st.info("Makrodaten nicht verfügbar.")
        else:
            st.dataframe(mm.tail(12).round(4))

    # Calendar
    with tabs[3]:
        st.header("Wirtschaftskalender (Monat)")
        cal = fetch_economic_calendar(12)
        st.write("Wichtigste Events:")
        for ev in cal:
            st.markdown(f"- {ev.get('date','')} — **{ev.get('country','')}** {ev.get('event','')} (Impact: {ev.get('impact','')})")

    # Signals
    with tabs[4]:
        st.header("Swing- & Trend-Signale (4h, D, W)")
        for sym in WATCHLIST:
            st.subheader(sym)
            tfs = {"4h":"4h","D":"1d","W":"1wk"}
            got = False
            for tf in tfs:
                df_tf = fetch_ohlc(sym, months=90 if tf=="4h" else 400, interval=("4h" if tf=="4h" else ("1wk" if tf=="W" else "1d")))
                if df_tf is None:
                    continue
                df_tf = compute_indicators(df_tf)
                alerts = detect_swing_signals(df_tf)
                if alerts:
                    got = True
                    st.markdown(f"**{tf}**")
                    for a in alerts:
                        st.warning(a)
            if not got:
                st.info("Keine akuten Trendwechsel gefunden.")

    st.markdown("---")
    st.caption("⚠️ Indikative Signale — keine Anlageberatung.")

# Entry
if __name__ == "__main__":
    try:
        main_app()
    except Exception as e:
        st.error("Unerwarteter Fehler: " + str(e))
        st.text(traceback.format_exc())
