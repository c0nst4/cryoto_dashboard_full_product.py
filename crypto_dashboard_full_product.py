# crypto_dashboard_full_product.py
"""
Krypto Profi-Dashboard (vollständig, robust)
- BTC / ETH / SOL
- Multi-Timeframe Indikatoren (4h, D, W)
- SMA, EMA, RSI, MACD, ATR
- RSI- & MACD-Subplots
- News + Fear&Greed
- Wirtschaftskalender (TradingEconomics fallback)
- Hybrid-Prognosen (technisch + makro)
Hinweis: Indikative Signale — keine Anlageberatung.
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
import typing
import traceback

# ---------------- Page config ----------------
st.set_page_config(layout="wide", page_title="Krypto Profi-Dashboard")
WATCHLIST = ["BTC-USD", "ETH-USD", "SOL-USD"]
TIMEFRAMES = {"4h": {"interval": "4h", "period_days": 90},
              "D": {"interval": "1d", "period_days": 365},
              "W": {"interval": "1wk", "period_days": 730}}
MIN_DATA_POINTS_FORECAST = 30
CACHE_SHORT = 300
CACHE_MED = 900
CACHE_LONG = 3600

# ---------------- Helpers ----------------
def _first_numeric_column_from_df(df: pd.DataFrame) -> pd.Series:
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
            arr = arr[:, 0]
        return pd.Series(arr, index=df.index).astype(float)
    except Exception:
        return pd.Series(dtype=float, index=df.index)

def safe_get_latest_value(latest_row, col: str) -> float:
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

# ---------------- OHLC Fetch ----------------
@st.cache_data(ttl=CACHE_SHORT)
def fetch_ohlc_yf(symbol: str, interval: str, period_days: int) -> typing.Optional[pd.DataFrame]:
    try:
        if interval == "4h":
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
            df = yf.download(symbol, period=f"{period_days}d", interval=interval, progress=False)
            if df is None or df.empty:
                return None
            df.index = pd.to_datetime(df.index)
            return df
    except Exception:
        return None

# ---------------- Indicators ----------------
def compute_indicators_for_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    close = safe_series_from_df(df, "Close")
    high = safe_series_from_df(df, "High")
    low = safe_series_from_df(df, "Low")
    if close.empty:
        return df
    # SMAs & EMAs
    for p in (20,50,200):
        df[f"SMA{p}"] = close.rolling(window=p, min_periods=1).mean()
        df[f"EMA{p}"] = close.ewm(span=p, adjust=False).mean()
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
        df["ATR"] = close.pct_change().rolling(14, min_periods=1).std() * close
    return df

@st.cache_data(ttl=CACHE_SHORT)
def get_tf_indicators(symbol: str) -> dict:
    out = {}
    for tf, meta in TIMEFRAMES.items():
        df = fetch_ohlc_yf(symbol, meta["interval"], meta["period_days"])
        df = compute_indicators_for_df(df) if df is not None else None
        out[tf] = df
    return out

# ---------------- Signals ----------------
def detect_swing_signals(df: pd.DataFrame) -> list:
    alerts = []
    if df is None or df.empty:
        return alerts
    # need the columns
    cols = ["EMA20","EMA50","EMA200","SMA50","SMA200","MACD","MACD_SIGNAL","RSI"]
    # ensure exist
    for c in cols:
        if c not in df.columns:
            return alerts
    sub = df[cols].dropna(how="any")
    if sub.shape[0] < 2:
        return alerts
    prev = sub.iloc[-2]; cur = sub.iloc[-1]
    try:
        # EMA cross
        if prev["EMA20"] < prev["EMA50"] and cur["EMA20"] > cur["EMA50"]:
            alerts.append("🚀 EMA20 kreuzt über EMA50 → bullishes Signal")
        if prev["EMA20"] > prev["EMA50"] and cur["EMA20"] < cur["EMA50"]:
            alerts.append("🔻 EMA20 kreuzt unter EMA50 → bearishes Signal")
        # SMA 50/200
        if prev["SMA50"] < prev["SMA200"] and cur["SMA50"] > cur["SMA200"]:
            alerts.append("📈 SMA50 kreuzt über SMA200 → mittelfristig Bullish")
        if prev["SMA50"] > prev["SMA200"] and cur["SMA50"] < cur["SMA200"]:
            alerts.append("📉 SMA50 kreuzt unter SMA200 → mittelfristig Bearish")
        # MACD
        if prev["MACD"] < prev["MACD_SIGNAL"] and cur["MACD"] > cur["MACD_SIGNAL"]:
            alerts.append("📊 MACD kreuzt über Signal → Momentum bullisch")
        if prev["MACD"] > prev["MACD_SIGNAL"] and cur["MACD"] < cur["MACD_SIGNAL"]:
            alerts.append("📊 MACD kreuzt unter Signal → Momentum bearisch")
        # RSI
        if cur["RSI"] > 70:
            alerts.append(f"⚠️ RSI überkauft ({cur['RSI']:.1f})")
        if cur["RSI"] < 30:
            alerts.append(f"⚠️ RSI überverkauft ({cur['RSI']:.1f})")
    except Exception:
        pass
    return alerts

# ---------------- Fear & Greed ----------------
@st.cache_data(ttl=CACHE_LONG)
def fetch_fear_greed():
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
        r.raise_for_status()
        js = r.json()
        return int(js["data"][0]["value"]), js["data"][0]["value_classification"]
    except Exception:
        return None, None

# ---------------- News aggregator ----------------
@st.cache_data(ttl=CACHE_MED)
def fetch_news(limit=10):
    feeds = [
        ("CoinTelegraph", "https://cointelegraph.com/rss"),
        ("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
        ("Bitcoin.com", "https://news.bitcoin.com/feed/"),
        ("Investing", "https://www.investing.com/rss/news_25.rss")
    ]
    headers = {"User-Agent": "Mozilla/5.0"}
    out = []
    for name, url in feeds:
        try:
            r = requests.get(url, timeout=10, headers=headers)
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
    if not out:
        return [{"source":"system","title":"Keine News verfügbar — prüfe Internetzugang oder RSS-Feeds","link":"","date":""}]
    # dedupe titles
    seen = set(); ded = []
    for i in out:
        key = i.get("title","")[:200]
        if key in seen: continue
        seen.add(key); ded.append(i)
    return ded[:limit]

# ---------------- Economic calendar (monthly) ----------------
@st.cache_data(ttl=CACHE_LONG)
def fetch_econ_calendar_monthly():
    try:
        base = "https://api.tradingeconomics.com/calendar"
        start = datetime.utcnow().replace(day=1).strftime("%Y-%m-%d")
        end = (datetime.utcnow().replace(day=1) + pd.DateOffset(months=1) - pd.DateOffset(days=1)).strftime("%Y-%m-%d")
        params = {"c":"guest:guest","from":start,"to":end}
        r = requests.get(base, params=params, timeout=10)
        if r.status_code == 200:
            data = r.json()
            # filter high impact
            out = []
            for ev in data:
                imp = str(ev.get("impact","")).lower()
                if imp in ("high","3","3.0"):
                    out.append({"date": ev.get("date"), "country": ev.get("country"), "event": ev.get("event"), "impact": ev.get("impact")})
            return out
    except Exception:
        pass
    # fallback: empty list
    return []

# ---------------- Macro features ----------------
@st.cache_data(ttl=CACHE_LONG)
def fetch_macro_features_daily():
    try:
        btc = yf.download("BTC-USD", period="3y", interval="1d", progress=False)["Close"]
    except Exception:
        btc = pd.Series(dtype=float)
    dxy = pd.Series(dtype=float)
    for t in ("DX-Y.NYB","DXY","USDX"):
        try:
            tmp = yf.download(t, period="3y", interval="1d", progress=False)["Close"]
            if tmp is not None and not tmp.empty:
                dxy = tmp; break
        except Exception:
            continue
    if dxy is None or dxy.empty:
        try:
            eurusd = yf.download("EURUSD=X", period="3y", interval="1d", progress=False)["Close"]
            if eurusd is not None and not eurusd.empty:
                dxy = 100 / eurusd
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
    df_daily = fetch_macro_features_daily()
    if df_daily is None or df_daily.empty:
        return pd.DataFrame()
    dfm = df_daily.resample("M").last().pct_change().dropna()
    return dfm

# ---------------- Prepare features for hybrid forecast ----------------
@st.cache_data(ttl=CACHE_MED)
def prepare_features_for_hybrid(symbol: str) -> pd.DataFrame:
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
    # temp to compute indicators
    temp = price.to_frame(name="Close")
    temp["Open"] = temp["Close"].shift(1).fillna(method="bfill")
    temp["High"] = temp[["Open","Close"]].max(axis=1)
    temp["Low"] = temp[["Open","Close"]].min(axis=1)
    temp = compute_indicators_for_df(temp)
    # diffs/slopes
    df["EMA20_50_DIFF"] = (temp["EMA20"] - temp["EMA50"]).reindex(df.index)
    df["SMA50_200_DIFF"] = (temp["SMA50"] - temp["SMA200"]).reindex(df.index)
    df["MACD_DIFF"] = temp["MACD_DIFF"].reindex(df.index) if "MACD_DIFF" in temp.columns else np.nan
    df["RSI"] = temp["RSI"].reindex(df.index) if "RSI" in temp.columns else np.nan
    # macro
    macro = fetch_macro_features_daily()
    if not macro.empty:
        macro_reindexed = macro.reindex(df.index).ffill().bfill()
        df["DXY"] = macro_reindexed["DXY"]
        df["VIX"] = macro_reindexed["VIX"]
    else:
        df["DXY"]=np.nan; df["VIX"]=np.nan
    # fill small gaps to avoid full dropna
    df = df.fillna(method="ffill").fillna(method="bfill").fillna(0)
    return df

# ---------------- Hybrid train & predict ----------------
def train_hybrid_and_predict(symbol: str, horizon_days: int = 30):
    df = prepare_features_for_hybrid(symbol)
    if df is None or df.empty or len(df) < MIN_DATA_POINTS_FORECAST:
        return None, None, "Zu wenige Daten"
    df = df.copy()
    df["FUT"] = df["Close"].shift(-horizon_days)
    df["FUT_RET"] = df["FUT"]/df["Close"] - 1
    df = df.dropna()
    if df.empty or len(df) < 20:
        return None, None, "Nicht genug vollständige Zeilen"
    candidate = ["RET1","RET7","RET30","VOL14","EMA20_50_DIFF","SMA50_200_DIFF","MACD_DIFF","RSI","DXY","VIX"]
    features = [c for c in candidate if c in df.columns]
    X = df[features].values
    y = df["FUT_RET"].values
    # split
    split = max(int(len(X)*0.75), len(X)-30)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]
    if len(X_tr) < 10 or len(X_te) < 5:
        return None, None, "Train/Test zu klein"
    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_te_s = scaler.transform(X_te)
    model = LinearRegression().fit(X_tr_s, y_tr)
    try:
        r2 = float(r2_score(y_te, model.predict(X_te_s)))
    except Exception:
        r2 = 0.0
    last_feat = scaler.transform(df[features].tail(1).values)
    pred = float(model.predict(last_feat)[0])
    return pred, r2, f"{len(df)} Zeilen verwendet"

def predict_horizons_hybrid(symbol: str):
    out = {}
    out["day_now"], out["day_now_r2"], out["day_now_info"] = train_hybrid_and_predict(symbol, 1)
    out["day_next"], out["day_next_r2"], out["day_next_info"] = train_hybrid_and_predict(symbol, 2)
    out["week_now"], out["week_now_r2"], out["week_now_info"] = train_hybrid_and_predict(symbol, 7)
    out["week_next"], out["week_next_r2"], out["week_next_info"] = train_hybrid_and_predict(symbol, 14)
    out["month_now"], out["month_now_r2"], out["month_now_info"] = train_hybrid_and_predict(symbol, 30)
    out["month_next"], out["month_next_r2"], out["month_next_info"] = train_hybrid_and_predict(symbol, 60)
    return out

# ---------------- Plot helpers (price + RSI + MACD) ----------------
def plot_with_indicators(df: pd.DataFrame, ticker: str, mode: str="EMA"):
    if df is None or df.empty:
        st.warning("Keine Daten")
        return
    # price
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"))
    if mode=="SMA":
        for p in (20,50,200):
            col = f"SMA{p}"
            if col in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col, line=dict(width=1)))
    else:
        cmap = {20:"blue",50:"orange",200:"red"}
        for p in (20,50,200):
            col = f"EMA{p}"
            if col in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col, line=dict(color=cmap.get(p,"black"), width=2)))
    fig.update_layout(title=f"{ticker} {mode}", height=420, margin=dict(l=10,r=10,t=30,b=10))
    st.plotly_chart(fig, use_container_width=True)
    # RSI subplot
    if "RSI" in df.columns:
        fig_r = go.Figure()
        fig_r.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI", line=dict(color="gold", width=2)))
        fig_r.add_hrect(y0=30,y1=70,fillcolor="lightgray",opacity=0.2,line_width=0)
        fig_r.update_layout(title="RSI (14)", height=200, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig_r, use_container_width=True)
    # MACD subplot
    if "MACD" in df.columns and "MACD_SIGNAL" in df.columns:
        fig_m = go.Figure()
        fig_m.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", line=dict(color="green", width=2)))
        fig_m.add_trace(go.Scatter(x=df.index, y=df["MACD_SIGNAL"], name="Signal", line=dict(color="red", width=2)))
        if "MACD_DIFF" in df.columns:
            fig_m.add_trace(go.Bar(x=df.index, y=df["MACD_DIFF"], name="Diff", opacity=0.3))
        fig_m.update_layout(title="MACD", height=220, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig_m, use_container_width=True)

# ---------------- UI Rendering per symbol ----------------
def render_charts_and_signals(symbol: str):
    st.header(f"{symbol} — Charts & Signale")
    tf_dfs = get_tf_indicators(symbol)
    sigs = {tf: detect_swing_signals(df) if df is not None else [] for tf, df in tf_dfs.items()}
    st.subheader("Multi-Timeframe Signale")
    # show table of signals
    rows=[]
    for tf, arr in sigs.items():
        if not arr:
            rows.append({"TF": tf, "Signal": "Keine akuten Signale"})
        else:
            for a in arr:
                rows.append({"TF": tf, "Signal": a})
    try:
        st.dataframe(pd.DataFrame(rows))
    except Exception:
        for r in rows: st.write(r)
    # plot SMA and EMA tabs
    sma_tab, ema_tab = st.tabs(["SMA-Charts","EMA-Charts"])
    for mode, tab in zip(["SMA","EMA"], (sma_tab, ema_tab)):
        with tab:
            for tf, df in tf_dfs.items():
                st.markdown(f"**{tf}**")
                if df is None or df.empty:
                    st.write("Keine Daten für diesen Timeframe.")
                    continue
                plot_with_indicators(df, f"{symbol} ({tf})", mode=mode)
                try:
                    latest = df.dropna(subset=["Close"]).iloc[-1]
                    c1,c2,c3 = st.columns(3)
                    c1.metric("Close", f"{safe_get_latest_value(latest,'Close'):.2f}")
                    c2.metric("RSI", f"{safe_get_latest_value(latest,'RSI'):.1f}")
                    c3.metric("MACD_DIFF", f"{safe_get_latest_value(latest,'MACD_DIFF'):.4f}")
                except Exception:
                    pass

# ---------------- Main UI ----------------
def main_app():
    st.title("📊 Krypto Profi-Dashboard (Pro-Version)")

    tabs = st.tabs(["Übersicht","Charts & Signale","Makro & Prognosen","Wirtschaftskalender","Tools"])

    with tabs[0]:
        st.header("Schnellübersicht")
        fgi_val, fgi_text = fetch_fear_greed()
        st.metric("Fear & Greed", fgi_val if fgi_val is not None else "n/a", fgi_text or "")
        st.subheader("Top News")
        news = fetch_news(10)
        if not news or (len(news)==1 and news[0].get("title","").startswith("Keine")):
            st.info("Keine aktuellen News verfügbar.")
        else:
            for n in news[:8]:
                if n.get("link"):
                    st.markdown(f"- **[{n['source']}]** [{n['title']}]({n['link']})")
                else:
                    st.markdown(f"- **[{n['source']}]** {n['title']}")

    with tabs[1]:
        st.header("Charts & Signale")
        for sym in WATCHLIST:
            render_charts_and_signals(sym)

    with tabs[2]:
        st.header("Makro-Analyse & Prognosen (Hybrid)")
        rows=[]
        for sym in WATCHLIST:
            preds = predict_horizons_hybrid(sym)
            def fmt(k):
                v = preds.get(k)
                r2 = preds.get(k + "_r2")
                info = preds.get(k + "_info")
                if v is None:
                    return f"n/a ({info})" if info else "n/a"
                try:
                    return f"{v*100:.2f}% (R²={r2:.2f})"
                except Exception:
                    return f"{v*100:.2f}%"
            rows.append({"Symbol": sym,
                         "Day now": fmt("day_now"),
                         "Day next": fmt("day_next"),
                         "Week now": fmt("week_now"),
                         "Week next": fmt("week_next"),
                         "Month now": fmt("month_now"),
                         "Month next": fmt("month_next")})
        st.dataframe(pd.DataFrame(rows))

        st.subheader("Makrodaten (Monatliche Returns)")
        dfm = fetch_macro_monthly_returns()
        if dfm is None or dfm.empty:
            st.info("Keine Makro-Daten verfügbar.")
        else:
            st.dataframe(dfm.tail(12).round(4))

    with tabs[3]:
        st.header("Wirtschaftskalender (Monat)")
        events = fetch_econ_calendar_monthly()
        if not events:
            st.info("Keine High-Impact Events gefunden oder API-Limit.")
        else:
            for ev in events[:12]:
                date = ev.get("date","")[:10]
                st.markdown(f"- {date} — **{ev.get('country','')}**: {ev.get('event','')} (Impact: {ev.get('impact','')})")

    with tabs[4]:
        st.header("Tools & Hinweise")
        st.markdown("""
        - Datenquellen: yfinance (Kurse), alternative.me (Fear&Greed), TradingEconomics (Kalender guest:guest), mehrere RSS-Feeds.
        - Prognosen sind indikativ. Keine Anlageberatung.
        - Bei Problemen: Manage app -> Logs in Streamlit Cloud und hier posten.
        """)
        if st.button("Test: Hybrid-Features (BTC)"):
            df = prepare_features_for_hybrid("BTC-USD")
            if df is None or df.empty:
                st.warning("Keine Feature-Daten verfügbar.")
            else:
                st.dataframe(df.tail(10))

    st.markdown("---")
    st.caption("⚠️ Indikative Signale — keine Anlageberatung.")

if __name__ == "__main__":
    main_app()
