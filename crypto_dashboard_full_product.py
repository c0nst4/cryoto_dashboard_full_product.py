# crypto_dashboard_full_product.py
"""
Komplettes Krypto-Dashboard (robust) - Teil 1/3
Features:
 - Multi-Timeframe Indikatoren: 4h, D, W (SMA, EMA, RSI, MACD, ATR)
 - Signale: EMA / SMA / MACD / RSI (4h, D, W)
 - Makro-Analyse + einfache Preisprognosen (Tag / Woche / Monat aktuell + next)
 - News + Wirtschaftskalender (TradingEconomics guest fallback)
 - Robust: sichere Series-Handling, Caching, Fallbacks
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

# ---------------- Page config ----------------
st.set_page_config(layout="wide", page_title="Krypto Profi-Dashboard (Multi-TF)")

# ---------------- Settings ----------------
WATCHLIST = ["BTC-USD", "ETH-USD", "SOL-USD"]
# timeframes to compute: 4h, daily, weekly
TIMEFRAMES = {"4h": {"interval": "4h", "period_days": 90},
              "D":  {"interval": "1d", "period_days": 365},
              "W":  {"interval": "1wk", "period_days": 730}}

# ---------------- Utilities / Robust helpers ----------------
def _first_numeric_column_from_df(df):
    if df is None or df.empty:
        return pd.Series(dtype=float)
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            return s
    try:
        return df.iloc[:, 0].astype(float)
    except Exception:
        return pd.Series(dtype=float, index=df.index)

def safe_series_from_df(df, col_name):
    """Return a 1D pd.Series for column col_name from df (robust: handles MultiIndex and arrays)."""
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
            # handle (n,1) shapes
            arr = arr[:, 0]
        return pd.Series(arr, index=df.index).astype(float)
    except Exception:
        return pd.Series(dtype=float, index=df.index)

def safe_get_latest_value(latest_row, col):
    """Safely extract numeric value from latest row (Series) or dict; return np.nan on problems."""
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

# ---------------- Data fetching ----------------
@st.cache_data(ttl=300)
def fetch_ohlc_yf(symbol: str, interval: str, period_days: int):
    """
    Fetch OHLC via yfinance for a given interval. period_days approximate the lookback window.
    Returns DataFrame or None.
    """
    try:
        # yfinance period string conservative mapping
        # for intraday intervals yfinance needs 'period' like '60d' etc.
        if interval in ["4h", "1h", "60m"]:
            # if 4h is not supported by yfinance on some environments, try "60m" and resample
            # request e.g. '90d' for intraday
            per = f"{max(30, min(period_days, 90))}d"
            df = yf.download(symbol, period=per, interval="60m", progress=False)
            if df is None or df.empty:
                return None
            # resample to 4h (if possible)
            try:
                df_4h = df.resample("4H").agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"})
                df_4h.dropna(how="any", inplace=True)
                return df_4h
            except Exception:
                return df
        else:
            # daily or weekly
            if interval == "1d":
                per = f"{period_days}d"
                df = yf.download(symbol, period=per, interval="1d", progress=False)
            elif interval == "1wk" or interval == "1w":
                # use 730 days for 2 years weekly
                per = "730d"
                df = yf.download(symbol, period=per, interval="1wk", progress=False)
            else:
                per = f"{period_days}d"
                df = yf.download(symbol, period=per, interval=interval, progress=False)
            if df is None or df.empty:
                return None
            df.index = pd.to_datetime(df.index)
            return df
    except Exception:
        return None

# ---------------- Indicator computations multi-timeframe ----------------
def compute_indicators_for_df(df):
    """Compute SMA/EMA/RSI/MACD/ATR for a given OHLC df. Returns df with new cols."""
    if df is None or df.empty:
        return df
    # ensure close, high, low are 1d series
    close = safe_series_from_df(df, "Close")
    high = safe_series_from_df(df, "High")
    low = safe_series_from_df(df, "Low")
    if close.empty:
        return df

    # SMAs
    try:
        df["SMA20"] = close.rolling(20, min_periods=1).mean()
        df["SMA50"] = close.rolling(50, min_periods=1).mean()
        df["SMA200"] = close.rolling(200, min_periods=1).mean()
    except Exception:
        df["SMA20"] = df["SMA50"] = df["SMA200"] = np.nan

    # EMAs
    try:
        df["EMA20"] = close.ewm(span=20, adjust=False).mean()
        df["EMA50"] = close.ewm(span=50, adjust=False).mean()
        df["EMA200"] = close.ewm(span=200, adjust=False).mean()
    except Exception:
        df["EMA20"] = df["EMA50"] = df["EMA200"] = np.nan

    # RSI
    try:
        df["RSI"] = ta.momentum.RSIIndicator(close, window=14).rsi()
    except Exception:
        # fallback
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
def get_tf_indicators(symbol: str):
    """
    For each timeframe in TIMEFRAMES fetch OHLC and compute indicators.
    Returns dict: {tf: df_with_indicators}
    """
    res = {}
    for tf, meta in TIMEFRAMES.items():
        intv = meta["interval"]
        days = meta["period_days"]
        df = fetch_ohlc_yf(symbol, intv, days)
        df = compute_indicators_for_df(df) if df is not None else None
        res[tf] = df
    return res

# ---------------- Signals across timeframes ----------------
def detect_signals_multi_tf(tf_dfs: dict) -> dict:
    """
    Returns signals per timeframe for EMA/SMA/MACD/RSI.
    Format: {tf: [list of strings]}
    """
    out = {}
    for tf, df in tf_dfs.items():
        alerts = []
        if df is None or df.empty:
            out[tf] = alerts
            continue
        # use last two non-null rows for required columns
        cols_needed = ["EMA20","EMA50","EMA200","SMA50","SMA200","MACD","MACD_SIGNAL","RSI"]
        sub = df[cols_needed].dropna(how="any")
        if sub.shape[0] < 2:
            out[tf] = alerts
            continue
        prev = sub.iloc[-2]; cur = sub.iloc[-1]
        try:
            # EMA cross
            if prev["EMA20"] < prev["EMA50"] and cur["EMA20"] > cur["EMA50"]:
                alerts.append("EMA20↑ über EMA50 (bullish crossover)")
            if prev["EMA20"] > prev["EMA50"] and cur["EMA20"] < cur["EMA50"]:
                alerts.append("EMA20↓ unter EMA50 (bearish crossover)")
            # SMA cross (50/200)
            if prev["SMA50"] < prev["SMA200"] and cur["SMA50"] > cur["SMA200"]:
                alerts.append("SMA50↑ über SMA200 (Golden Cross möglich)")
            if prev["SMA50"] > prev["SMA200"] and cur["SMA50"] < cur["SMA200"]:
                alerts.append("SMA50↓ unter SMA200 (Death Cross möglich)")
            # MACD cross
            if prev["MACD"] < prev["MACD_SIGNAL"] and cur["MACD"] > cur["MACD_SIGNAL"]:
                alerts.append("MACD kreuzt über Signal (bullish momentum)")
            if prev["MACD"] > prev["MACD_SIGNAL"] and cur["MACD"] < cur["MACD_SIGNAL"]:
                alerts.append("MACD kreuzt unter Signal (bearish momentum)")
            # RSI thresholds
            if cur["RSI"] > 70:
                alerts.append(f"RSI überkauft ({cur['RSI']:.1f})")
            if cur["RSI"] < 30:
                alerts.append(f"RSI überverkauft ({cur['RSI']:.1f})")
        except Exception:
            pass
        out[tf] = alerts
    return out

# ---------------- Economic calendar (monthly summary) ----------------
@st.cache_data(ttl=1800)
def fetch_econ_calendar_monthly():
    """
    Try TradingEconomics calendar endpoint (guest:guest) for the current month.
    Fallback: return empty list with message.
    """
    try:
        base = "https://api.tradingeconomics.com/calendar"
        # TradingEconomics might require API key; guest:guest works for limited calls
        params = {"c":"guest:guest", "from": (datetime.utcnow().replace(day=1)).strftime("%Y-%m-%d"),
                  "to": (datetime.utcnow().replace(day=1) + pd.DateOffset(months=1) - pd.DateOffset(days=1)).strftime("%Y-%m-%d")}
        r = requests.get(base, params=params, timeout=10)
        if r.status_code != 200:
            return []
        data = r.json()
        # filter high impact
        high = [ev for ev in data if ev.get("impact") in ("High","high","3")]
        # normalize
        out = []
        for ev in high:
            out.append({"date": ev.get("date"), "country": ev.get("country"), "event": ev.get("event"), "impact": ev.get("impact")})
        return out
    except Exception:
        return []

# ---------------- Macro features for forecasts ----------------
@st.cache_data(ttl=1800)
def fetch_macro_features():
    """
    Returns recent macro series used for forecasting: DXY, VIX, BTC close (daily)
    """
    try:
        vix = yf.download("^VIX", period="2y", interval="1d", progress=False)["Close"]
    except Exception:
        vix = pd.Series(dtype=float)
    # try multiple DXY tickers
    dxy = pd.Series(dtype=float)
    for t in ("DX-Y.NYB","DXY","USDX"):
        try:
            tmp = yf.download(t, period="2y", interval="1d", progress=False)["Close"]
            if tmp is not None and not tmp.empty:
                dxy = tmp; break
        except Exception:
            continue
    try:
        btc = yf.download("BTC-USD", period="2y", interval="1d", progress=False)["Close"]
    except Exception:
        btc = pd.Series(dtype=float)
    # combine
    try:
        df = pd.concat({"BTC": btc, "DXY": dxy, "VIX": vix}, axis=1)
        df.columns = ["BTC","DXY","VIX"] if not isinstance(df.columns, pd.MultiIndex) else [c[0] for c in df.columns]
        df = df.dropna()
        return df
    except Exception:
        return pd.DataFrame()

# ---------------- Forecast helpers (day/week/month horizons) ----------------
def prepare_features_for_forecast(symbol):
    """
    Prepare simple features for forecasting:
    - recent returns (daily)
    - recent volatility (rolling std)
    - macro features (DXY, VIX)
    Returns a DataFrame indexed by date with features.
    """
    try:
        df_price = yf.download(symbol, period="400d", interval="1d", progress=False)["Close"]
    except Exception:
        df_price = pd.Series(dtype=float)
    macro = fetch_macro_features()
    if df_price is None or df_price.empty:
        return pd.DataFrame()
    df = pd.DataFrame({"Close": df_price})
    df["RET1"] = df["Close"].pct_change()
    df["RET7"] = df["Close"].pct_change(7)
    df["VOL14"] = df["RET1"].rolling(14).std()
    # merge latest macro (forward-fill)
    if not macro.empty:
        macro_daily = macro.reindex(df.index).ffill().bfill()
        df = pd.concat([df, macro_daily[["DXY","VIX"]]], axis=1)
    df = df.dropna()
    return df

def train_simple_model_and_predict(symbol, horizon_days=30):
    """
    Train a linear regression on recent daily returns to predict the cumulative return over horizon_days.
    This is a simplistic approach: X are recent features, y is future cumulative return over horizon.
    Returns predicted cumulative return (float) and r2 score (float) or (None,None)
    """
    df = prepare_features_for_forecast(symbol)
    if df.empty or len(df) < 60:
        return None, None
    df = df.copy()
    # target: cumulative return over next horizon_days
    df["FUTURE_PRICE"] = df["Close"].shift(-horizon_days)
    df["FUTURE_RET"] = df["FUTURE_PRICE"] / df["Close"] - 1
    df = df.dropna()
    if df.empty or len(df) < 40:
        return None, None
    # features: recent returns + vol + macro
    features = ["RET1","RET7","VOL14"]
    if "DXY" in df.columns and "VIX" in df.columns:
        features += ["DXY","VIX"]
    X = df[features].values
    y = df["FUTURE_RET"].values
    # simple train-test split (last 20% test)
    split = int(len(X)*0.8)
    if split < 10:
        return None, None
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]
    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_te_s = scaler.transform(X_te)
    model = LinearRegression().fit(X_tr_s, y_tr)
    try:
        r2 = float(r2_score(y_te, model.predict(X_te_s)))
    except Exception:
        r2 = 0.0
    # predict from most recent features (last row)
    last_feat = scaler.transform(df[features].tail(1).values)
    pred = float(model.predict(last_feat)[0])
    return pred, r2

# convenience wrappers for different horizons
def predict_horizons(symbol):
    """
    Returns dictionary with:
     - day_now, day_next
     - week_now, week_next
     - month_now, month_next
    Each value is (pred_return, r2) or (None,None)
    """
    out = {}
    # day: horizon 1 day
    out["day_now"], out["day_now_r2"] = train_simple_model_and_predict(symbol, horizon_days=1)
    out["day_next"], out["day_next_r2"] = train_simple_model_and_predict(symbol, horizon_days=2)
    # week: 7 and 14
    out["week_now"], out["week_now_r2"] = train_simple_model_and_predict(symbol, horizon_days=7)
    out["week_next"], out["week_next_r2"] = train_simple_model_and_predict(symbol, horizon_days=14)
    # month: 30 and 60
    out["month_now"], out["month_now_r2"] = train_simple_model_and_predict(symbol, horizon_days=30)
    out["month_next"], out["month_next_r2"] = train_simple_model_and_predict(symbol, horizon_days=60)
    return out

# ---------------- Visualization helpers ----------------
def plot_candles_with_sma(ax_df, title, show_sma=True, sma_periods=(20,50,200), show_ema=False, ema_periods=(20,50,200)):
    """
    Return a plotly.Figure with candlesticks + requested SMA/EMA lines.
    EMAs will be solid colored lines.
    """
    fig = go.Figure(data=[go.Candlestick(x=ax_df.index, open=ax_df["Open"], high=ax_df["High"],
                                         low=ax_df["Low"], close=ax_df["Close"], name="Price")])
    # SMA lines
    if show_sma:
        for p in sma_periods:
            col = f"SMA{p}"
            if col in ax_df.columns:
                fig.add_trace(go.Scatter(x=ax_df.index, y=ax_df[col], name=col, line=dict(width=1)))
    # EMA lines (solid colored)
    if show_ema:
        color_map = {20: "blue", 50: "orange", 200: "red"}
        for p in ema_periods:
            col = f"EMA{p}"
            if col in ax_df.columns:
                fig.add_trace(go.Scatter(x=ax_df.index, y=ax_df[col], name=col,
                                         line=dict(color=color_map.get(p,"black"), width=2, dash="solid")))
    fig.update_layout(title=title, height=420, margin=dict(l=10,r=10,t=30,b=10))
    return fig

# ---------------- UI helpers for signals table ----------------
def build_signals_table(signals_dict):
    """
    signals_dict: {tf: [str,...]}
    returns DataFrame with columns TF and Signal (one row per signal)
    """
    rows = []
    for tf, arr in signals_dict.items():
        if not arr:
            rows.append({"TF": tf, "Signal": "Keine akuten Signale"})
        else:
            for s in arr:
                rows.append({"TF": tf, "Signal": s})
    return pd.DataFrame(rows)

# ---------------- UI: Part (charts & signals) ----------------
# (This part will be connected into final UI in Part 3)
# Provide a function to render charts & signals for a given symbol
def render_charts_and_signals(symbol):
    """
    Renders SMA-Tab and EMA-Tab charts + multi-timeframe signals + MACD/RSi mini-panels
    """
    st.subheader(f"{symbol} — Charts & Signals")
    # fetch multi-TF dfs with indicators
    tf_dfs = get_tf_indicators(symbol)
    # Signals per TF
    signals = detect_signals_multi_tf(tf_dfs)
    # display signals summary
    st.markdown("### Multi-Timeframe Signale")
    sig_df = build_signals_table(signals)
    st.dataframe(sig_df)

    # display charts per TF (SMA & EMA separated)
    # SMA tab
    sma_tab, ema_tab = st.tabs(["SMA-Trends", "EMA-Trends"])
    with sma_tab:
        for tf, df in tf_dfs.items():
            st.markdown(f"**{symbol} — {tf} (SMA)**")
            if df is None or df.empty:
                st.write("Keine Daten für diesen Timeframe.")
                continue
            fig = plot_candles_with_sma(df, f"{symbol} {tf} SMA-Chart", show_sma=True, show_ema=False)
            st.plotly_chart(fig, use_container_width=True)
    with ema_tab:
        for tf, df in tf_dfs.items():
            st.markdown(f"**{symbol} — {tf} (EMA)**")
            if df is None or df.empty:
                st.write("Keine Daten für diesen Timeframe.")
                continue
            fig = plot_candles_with_sma(df, f"{symbol} {tf} EMA-Chart", show_sma=False, show_ema=True)
            st.plotly_chart(fig, use_container_width=True)
            # show small indicator boxes
            try:
                latest = df.dropna(subset=["Close"]).iloc[-1]
                ccol1, ccol2, ccol3 = st.columns(3)
                ccol1.metric("Close", f"{safe_get_latest_value(latest,'Close'):.2f}")
                ccol2.metric("RSI", f"{safe_get_latest_value(latest,'RSI'):.1f}")
                ccol3.metric("MACD_DIFF", f"{safe_get_latest_value(latest,'MACD_DIFF'):.4f}")
            except Exception:
                pass

# ---------------- End of Part 2 ----------------

# ---------------- Part 3: UI Final (Overview, News, Calendar, Macro Forecasts) ----------------

# ---------------- Fear & Greed (single call) ----------------
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

# ---------------- News aggregator (robust) ----------------
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

# ---------------- Main UI ----------------
def render_overview():
    st.title("📊 Krypto Profi-Dashboard (vollständig)")
    st.caption("Multi-Timeframe Indikatoren, Signale, Makro-Prognosen — indikativ, keine Anlageberatung.")

    # top metrics row
    fgi_val, fgi_text = fetch_fear_greed()
    cols = st.columns(3)
    cols[0].metric("Fear & Greed", fgi_val if fgi_val is not None else "n/a", fgi_text or "")
    # show VIX and BTC last close
    try:
        btc_close = yf.download("BTC-USD", period="2d", interval="1d", progress=False)["Close"].iloc[-1]
        cols[1].metric("BTC Last", f"{btc_close:.2f} USD")
    except Exception:
        cols[1].metric("BTC Last", "n/a")
    try:
        vix_val = yf.download("^VIX", period="2d", interval="1d", progress=False)["Close"].iloc[-1]
        cols[2].metric("VIX Last", f"{vix_val:.2f}")
    except Exception:
        cols[2].metric("VIX Last", "n/a")

    # News & Calendar
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
        st.info("Keine High-Impact Events gefunden oder API-Limit. Prüfe TradingEconomics-Zugang, falls du mehr brauchst.")

# ---------------- Macro Forecast Display ----------------
def render_macro_and_forecasts():
    st.header("🌍 Makro-Analyse & Preisprognosen (indikativ)")
    st.write("Prognosen basieren auf einfachen Regressionsmodellen mit historischen Returns + Makrovariablen (DXY, VIX).")
    rows = []
    for symbol in WATCHLIST:
        try:
            preds = predict_horizons(symbol)
        except Exception:
            preds = {}
        # format outputs
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
            "Symbol": symbol,
            "Day (now)": fmt("day_now"),
            "Day (next)": fmt("day_next"),
            "Week (now)": fmt("week_now"),
            "Week (next)": fmt("week_next"),
            "Month (now)": fmt("month_now"),
            "Month (next)": fmt("month_next")
        })
    try:
        df_out = pd.DataFrame(rows)
        st.dataframe(df_out)
    except Exception:
        for r in rows:
            st.write(r)

    st.markdown("**Erläuterung:** „now“ heißt die Prognose für die aktuelle Periode (z.B. aktueller Tag / Woche / Monat), „next“ ist die darauf folgende Periode. Das Modell ist einfach: verwendet historische Returns + Volatilität + DXY/VIX als Features.")

# ---------------- Render per-symbol charts & signals ----------------
def render_all_charts_signals():
    st.header("📈 Charts & Multi-Timeframe Signale")
    for sym in WATCHLIST:
        render_charts_and_signals(sym)

# ---------------- App Layout ----------------
def main_app():
    tabs = st.tabs(["Übersicht", "Charts & Signale", "Makro & Prognosen", "Tools"])
    with tabs[0]:
        render_overview()
    with tabs[1]:
        render_all_charts_signals()
    with tabs[2]:
        render_macro_and_forecasts()
    with tabs[3]:
        st.header("🔧 Tools & Hinweise")
        st.markdown("""
        - Datenquelle: yfinance (Kurse), alternative.me (Fear&Greed), TradingEconomics (Kalender, guest:guest)  
        - Prognosen: einfache lineare Modelle — nur Indikationen.  
        - Wenn Feeds fehlen: prüfe Netzwerk / Streamlit-Cloud-Logs.  
        - Möchtest du Alerts via Telegram/Discord? Ich kann das hinzufügen.
        """)
        st.caption("© Dashboard – Indikative Daten. Keine Anlageberatung.")

# ---------------- Run ----------------
if __name__ == "__main__":
    main_app()
