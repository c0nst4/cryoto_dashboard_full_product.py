# crypto_forecast_pro_multisource.py
"""
Krypto-Prognose-App (einseitig)
- Multi-News-Sources (Investing.com + CoinDesk + CoinTelegraph + Bitcoin.com + Google News RSS)
- Makro + Event-Analyse
- Technische Indikatoren: SMA/EMA(20/50/200), RSI(14), MACD, ATR
- ML: GradientBoostingRegressor (Fallbacks: LinearRegression -> heuristic)
- Vorhersagen: Day now/next, Week now/next, Month now/next
Hinweis: Indikative Signale — keine Anlageberatung.
"""

import os, traceback, time
from datetime import datetime, timedelta
import math

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import ta
import plotly.graph_objects as go

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

# Optional: pandas_datareader for FRED (if user wants later)
try:
    import pandas_datareader.data as pdr
    PDR_AVAILABLE = True
except Exception:
    PDR_AVAILABLE = False

# ---------------- Config ----------------
st.set_page_config(layout="wide", page_title="Krypto Prognosen (Multi-Source, ML)")
st.title("🔮 Krypto-Prognosen — Tag / Woche / Monat (ML + Makro + Events)")

ASSETS = ["BTC-USD", "ETH-USD", "SOL-USD"]
NOW = datetime.utcnow()
YEARS_EVENT_HISTORY = 5

# ML thresholds
MIN_ROWS_GB = 300   # preferable rows for GB
MIN_ROWS_LR = 80    # acceptable rows for linear model
MIN_ROWS_HEUR = 20  # for heuristic fallback

# horizons in days
HORIZONS = {"day":1, "week":7, "month":30}

GB_PARAMS = {"n_estimators":150, "max_depth":4, "learning_rate":0.05, "random_state":42}

# cache ttls
TTL_SHORT = 300
TTL_MED = 900
TTL_LONG = 3600

# ---------------- Helpers ----------------
def log_and_print(e):
    print("ERR:", e)
    traceback.print_exc()

def safe_fetch(url, headers=None, timeout=12):
    try:
        r = requests.get(url, headers=headers or {}, timeout=timeout)
        if r.status_code == 200:
            return r
    except Exception as e:
        # silent
        pass
    return None

# ---------------- News aggregator (multi-source) ----------------
@st.cache_data(ttl=TTL_MED)
def fetch_news_multi(limit=20):
    """
    Aggregiert mehrere Feeds, parst und dedupliziert.
    Quellen: CoinTelegraph, CoinDesk, Bitcoin.com, Investing (RSS if available), Google News search.
    """
    feeds = [
        ("CoinTelegraph","https://cointelegraph.com/rss"),
        ("CoinDesk","https://www.coindesk.com/arc/outboundfeeds/rss/"),
        ("Bitcoin.com","https://news.bitcoin.com/feed/"),
        # Investing.com doesn't provide a simple RSS for calendar news; we use their generic news RSS via search
        ("Investing","https://www.investing.com/rss/news_25.rss"),
        ("GoogleNews","https://news.google.com/rss/search?q=bitcoin+OR+crypto+OR+ethereum&hl=en-US&gl=US&ceid=US:en")
    ]
    headers = {"User-Agent":"Mozilla/5.0"}
    items = []
    for name, url in feeds:
        try:
            r = safe_fetch(url, headers=headers)
            if not r:
                continue
            # feed might be xml
            soup = BeautifulSoup(r.content, "xml")
            found = soup.find_all("item")
            for it in found[: max(1, limit // len(feeds))]:
                title = it.title.text if it.title else ""
                link = (it.link.text if it.link else "") or (it.guid.text if it.guid else "")
                pub = it.pubDate.text if it.pubDate else ""
                items.append({"source":name, "title": title.strip(), "link": link.strip(), "date": pub})
        except Exception:
            continue
    # dedupe by title
    seen = set(); ded=[]
    for it in items:
        t = it.get("title","")[:200]
        if not t: continue
        if t in seen: continue
        seen.add(t); ded.append(it)
    if not ded:
        return [{"source":"system","title":"Keine News verfügbar – prüfe Internetzugang oder RSS-Feeds","link":"","date":""}]
    # sort by date if possible (items may have pubDate)
    try:
        ded_sorted = sorted(ded, key=lambda x: x.get("date",""), reverse=True)
    except Exception:
        ded_sorted = ded
    return ded_sorted[:limit]

# ---------------- Investing.com calendar scraping (next days) ----------------
@st.cache_data(ttl=TTL_MED)
def fetch_investing_events(days_ahead=30):
    """
    Versucht Investing.com Economic Calendar zu parsen.
    Hinweis: Investing.com ist JS-lastig — die HTML liefert oft nur limited info.
    Wir parsen simple event rows if available.
    """
    out=[]
    base = "https://www.investing.com/economic-calendar/"
    headers = {"User-Agent":"Mozilla/5.0"}
    r = safe_fetch(base, headers=headers)
    if not r:
        return []
    try:
        soup = BeautifulSoup(r.text, "html.parser")
        # They often include rows with id 'eventRowId...'
        rows = soup.find_all("tr", id=lambda x: x and x.startswith("eventRowId"))
        for r0 in rows:
            try:
                ev_name = r0.get("event_attr_name") or r0.get("data-event-name") or ""
                country = r0.get("data-event-country") or ""
                # date
                date_attr = r0.get("data-event-datetime") or r0.get("data-event-datetime2") or ""
                date_str = ""
                if date_attr:
                    try:
                        date_str = str(pd.to_datetime(date_attr).date())
                    except Exception:
                        date_str = ""
                # impact
                impact_td = r0.find("td", class_="sentiment")
                impact = ""
                if impact_td:
                    impact = impact_td.get("title","") or impact_td.text or ""
                out.append({"date":date_str, "country":country.strip(), "event":ev_name.strip(), "impact":impact.strip()})
            except Exception:
                continue
    except Exception:
        return []
    # dedupe
    seen=set(); ded=[]
    for e in out:
        key=(e.get("date",""), e.get("event",""))
        if key in seen: continue
        seen.add(key); ded.append(e)
    return ded

# ---------------- TradingEconomics API events (historical & fallback) ----------------
@st.cache_data(ttl=TTL_LONG)
def fetch_tradingeconomics_calendar(start_date, end_date, limit=500):
    """
    Uses TradingEconomics public API with guest:guest (may be rate limited). Returns list of events.
    """
    try:
        base="https://api.tradingeconomics.com/calendar"
        params={"c":"guest:guest","from":start_date,"to":end_date}
        r = requests.get(base, params=params, timeout=12)
        if r.status_code == 200:
            return r.json()[:limit]
    except Exception:
        pass
    return []

# ---------------- Macro timeseries: DXY, VIX, BTC ----------------
@st.cache_data(ttl=TTL_LONG)
def fetch_macro_timeseries(days=365*5):
    """
    Returns daily DataFrame with DXY, VIX, BTC_close (if available).
    """
    try:
        # BTC close
        btc = yf.download("BTC-USD", period=f"{days}d", interval="1d", progress=False)
        btc_c = btc["Close"] if (btc is not None and "Close" in btc.columns) else pd.Series(dtype=float)
    except Exception:
        btc_c = pd.Series(dtype=float)

    # VIX
    try:
        v = yf.download("^VIX", period=f"{days}d", interval="1d", progress=False)
        vix = v["Close"] if (v is not None and "Close" in v.columns) else pd.Series(dtype=float)
    except Exception:
        vix = pd.Series(dtype=float)

    # DXY candidates
    dxy = pd.Series(dtype=float)
    for t in ("DX-Y.NYB","DXY","USDX"):
        try:
            d = yf.download(t, period=f"{days}d", interval="1d", progress=False)
            if d is not None and not d.empty and "Close" in d.columns:
                dxy = d["Close"]; break
        except Exception:
            continue
    if dxy is None or dxy.empty:
        try:
            eur = yf.download("EURUSD=X", period=f"{days}d", interval="1d", progress=False)
            if eur is not None and not eur.empty and "Close" in eur.columns:
                dxy = 100 / eur["Close"]
        except Exception:
            dxy = pd.Series(dtype=float)

    df = pd.concat({"BTC":btc_c, "DXY":dxy, "VIX":vix}, axis=1)
    # flatten
    if isinstance(df.columns, pd.MultiIndex):
        df.columns=[c[0] for c in df.columns]
    df = df.dropna()
    return df

# ---------------- Fear & Greed ----------------
@st.cache_data(ttl=TTL_MED)
def fetch_fear_greed():
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
        if r.status_code==200:
            js=r.json()
            return int(js["data"][0]["value"]), js["data"][0]["value_classification"]
    except Exception:
        pass
    return None, None

# ---------------- Technical indicator builder ----------------
def compute_technical_indicators(df):
    """
    Input: OHLCV df with DateTimeIndex
    Adds: SMA20/50/200, EMA20/50/200, RSI(14), MACD, MACD_SIGNAL, MACD_DIFF, ATR
    Returns df with columns added.
    """
    df = df.copy()
    close = df["Close"].astype(float)
    high = df["High"].astype(float) if "High" in df.columns else close
    low = df["Low"].astype(float) if "Low" in df.columns else close
    for p in (20,50,200):
        df[f"SMA{p}"] = close.rolling(window=p, min_periods=1).mean()
        df[f"EMA{p}"] = close.ewm(span=p, adjust=False).mean()
    # RSI
    try:
        df["RSI"] = ta.momentum.RSIIndicator(close, window=14).rsi()
    except Exception:
        df["RSI"] = np.nan
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

# ---------------- Build features for ML ----------------
@st.cache_data(ttl=TTL_MED)
def build_feature_table(asset_symbol):
    """
    Build daily features for ML training:
    - price returns: 1/7/30
    - volatility (vol14)
    - technical indicators (EMA/SMA/RSI/MACD/ATR)
    - macro: DXY, VIX, BTC (macro)
    - fear & greed current (scalar)
    - event features for days where investing.com lists events (impact/count)
    """
    try:
        # price history (6 years)
        df_price = yf.download(asset_symbol, period="2200d", interval="1d", progress=False)
        if df_price is None or df_price.empty:
            return pd.DataFrame()
        df_price.index = pd.to_datetime(df_price.index).normalize()
        df = pd.DataFrame(index=df_price.index)
        df["Close"] = df_price["Close"]
        df["Ret1"] = df["Close"].pct_change(1)
        df["Ret7"] = df["Close"].pct_change(7)
        df["Ret30"] = df["Close"].pct_change(30)
        df["Vol14"] = df["Ret1"].rolling(14).std()

        # technicals
        tech = compute_technical_indicators(df_price)
        for col in ["EMA20","EMA50","EMA200","SMA20","SMA50","SMA200","RSI","MACD","MACD_SIGNAL","MACD_DIFF","ATR"]:
            if col in tech.columns:
                df[col] = tech[col].reindex(df.index)

        # macro
        macro = fetch_macro_timeseries(days=365*6)
        if macro is not None and not macro.empty:
            macro_r = macro.reindex(df.index).ffill().bfill()
            df["DXY"] = macro_r["DXY"]
            df["VIX"] = macro_r["VIX"]
            df["BTC_macro"] = macro_r["BTC"]
        else:
            df["DXY"] = np.nan; df["VIX"] = np.nan; df["BTC_macro"]=np.nan

        # fear&greed
        fgi, lbl = fetch_fear_greed()
        df["FearGreed"] = fgi if fgi is not None else np.nan

        # events: investing upcoming (we will mark event days)
        events = fetch_investing_events(days_ahead=90)
        event_map = {}
        for e in events:
            try:
                d = e.get("date","")
                if not d: continue
                d0 = pd.to_datetime(d).date()
                impact_text = e.get("impact","").lower()
                imp = 2 if "high" in impact_text else (1 if "med" in impact_text else 0)
                event_map.setdefault(d0, []).append({"event":e.get("event",""), "impact":imp})
            except Exception:
                continue
        df["event_impact_today"]=0
        df["event_count_today"]=0
        df["event_hist_mean_1d"]=0.0
        # historical reaction stats from TradingEconomics for typical keywords (best-effort)
        for dt in df.index:
            ddate = dt.date()
            arr = event_map.get(ddate, [])
            if arr:
                df.at[dt,"event_impact_today"] = max([a["impact"] for a in arr])
                df.at[dt,"event_count_today"] = len(arr)
                # take first event keyword and compute historical mean
                try:
                    kw = arr[0]["event"].split()[0] if arr[0]["event"] else ""
                    stats = analyze_past_event_reactions(asset_symbol, kw, years=YEARS_EVENT_HISTORY)
                    if stats and stats.get("mean_1d") is not None:
                        df.at[dt,"event_hist_mean_1d"] = stats["mean_1d"]
                except Exception:
                    pass

        # final cleaning: forward/backfill small gaps
        df = df.fillna(method="ffill").fillna(method="bfill")
        df = df.dropna(how="all")
        return df
    except Exception as e:
        log_and_print(e)
        return pd.DataFrame()

# ---------------- Historical event analysis (TradingEconomics) ----------------
@st.cache_data(ttl=TTL_LONG)
def analyze_past_event_reactions(asset_symbol, event_kw, window_days=7, years=YEARS_EVENT_HISTORY):
    """
    Best-effort: use TradingEconomics calendar to find events containing keyword in last `years` years,
    then compute average 1d/3d/7d returns on the asset after the event.
    """
    try:
        base = "https://api.tradingeconomics.com/calendar"
        start = (datetime.utcnow() - pd.DateOffset(years=years)).strftime("%Y-%m-%d")
        end = datetime.utcnow().strftime("%Y-%m-%d")
        params = {"c":"guest:guest","from":start,"to":end}
        r = requests.get(base, params=params, timeout=12)
        hits=[]
        if r.status_code==200:
            arr=r.json()
            for ev in arr:
                if event_kw.lower() in str(ev.get("event","")).lower():
                    hits.append(ev)
        if not hits:
            return {"n":0,"mean_1d":None,"mean_3d":None,"mean_7d":None}
        prices = yf.download(asset_symbol, period=f"{years*365+60}d", interval="1d", progress=False)
        if prices is None or prices.empty:
            return {"n":0,"mean_1d":None,"mean_3d":None,"mean_7d":None}
        close = prices["Close"]
        rets1, rets3, rets7 = [], [], []
        for ev in hits:
            dstr = ev.get("date","")[:10]
            try:
                ev_date = pd.to_datetime(dstr).date()
                idx_dates = close.index.normalize()
                pos = None
                try:
                    pos = idx_dates.get_loc(pd.Timestamp(ev_date))
                except Exception:
                    pos = np.searchsorted(idx_dates, pd.Timestamp(ev_date))
                    if pos >= len(idx_dates): continue
                p0 = close.iloc[int(pos)]
                if int(pos)+1 < len(close):
                    rets1.append((close.iloc[int(pos)+1]/p0)-1)
                if int(pos)+3 < len(close):
                    rets3.append((close.iloc[int(pos)+3]/p0)-1)
                if int(pos)+7 < len(close):
                    rets7.append((close.iloc[int(pos)+7]/p0)-1)
            except Exception:
                continue
        out={"n":len(hits),"mean_1d": (np.mean(rets1) if rets1 else None), "mean_3d":(np.mean(rets3) if rets3 else None), "mean_7d":(np.mean(rets7) if rets7 else None)}
        return out
    except Exception as e:
        log_and_print(e)
        return {"n":0,"mean_1d":None,"mean_3d":None,"mean_7d":None}

# ---------------- ML training/prediction helpers ----------------
def train_predict_model(feature_df, horizon_days):
    """
    Trains model to predict forward return after horizon_days.
    Strategy:
     - if rows >= MIN_ROWS_GB: GradientBoostingRegressor
     - elif rows >= MIN_ROWS_LR: LinearRegression
     - elif rows >= MIN_ROWS_HEUR: heuristic (recent mean returns)
     - else: can't produce meaningful forecast (rare)
    Returns (pred_return, info_dict)
    """
    try:
        df = feature_df.copy()
        df["future_price"] = df["Close"].shift(-horizon_days)
        df["target"] = df["future_price"]/df["Close"] - 1
        df = df.dropna()
        n = len(df)
        if n < MIN_ROWS_HEUR:
            return None, {"status":"not_enough_rows", "n":n}
        # candidate features
        candidate = ["Ret1","Ret7","Ret30","Vol14","EMA20","EMA50","EMA200","SMA20","SMA50","SMA200",
                     "RSI","MACD_DIFF","ATR","DXY","VIX","FearGreed","event_impact_today","event_count_today","event_hist_mean_1d"]
        features = [c for c in candidate if c in df.columns]
        if not features:
            return None, {"status":"no_features"}
        X = df[features].values
        y = df["target"].values
        # if enough rows for GB
        if n >= MIN_ROWS_GB:
            try:
                split = max(int(n*0.7), n-100)
                Xtr, Xte = X[:split], X[split:]
                ytr, yte = y[:split], y[split:]
                scaler = StandardScaler().fit(Xtr)
                Xtr_s = scaler.transform(Xtr); Xte_s = scaler.transform(Xte)
                model = GradientBoostingRegressor(**GB_PARAMS)
                model.fit(Xtr_s, ytr)
                ypred = model.predict(Xte_s)
                r2 = float(r2_score(yte, ypred))
                mse = float(mean_squared_error(yte, ypred))
                last_X = scaler.transform(df[features].tail(1).values)
                pred = float(model.predict(last_X)[0])
                return pred, {"model":"GB","r2":r2,"mse":mse,"n":n}
            except Exception as e:
                log_and_print(e)
                # fallback to LR
        # Linear regression fallback if enough rows
        if n >= MIN_ROWS_LR:
            try:
                split = max(int(n*0.7), n-50)
                Xtr, Xte = X[:split], X[split:]
                ytr, yte = y[:split], y[split:]
                scaler = StandardScaler().fit(Xtr)
                Xtr_s = scaler.transform(Xtr); Xte_s = scaler.transform(Xte)
                lr = LinearRegression().fit(Xtr_s, ytr)
                ypred = lr.predict(Xte_s)
                r2 = float(r2_score(yte, ypred))
                mse = float(mean_squared_error(yte, ypred))
                last_X = scaler.transform(df[features].tail(1).values)
                pred = float(lr.predict(last_X)[0])
                return pred, {"model":"LR","r2":r2,"mse":mse,"n":n}
            except Exception as e:
                log_and_print(e)
        # heuristic fallback: use recent mean returns scaled
        recent = df["Ret7"].tail(60).dropna()
        if len(recent) >= 3:
            mean = float(recent.mean())
            # scale by volatility
            vol = float(df["Vol14"].tail(60).mean()) if "Vol14" in df.columns else 0.0
            pred = mean  # simple
            return pred, {"model":"heuristic_mean","n":n, "recent_mean":mean, "vol":vol}
        return None, {"status":"not_enough_after_fallback","n":n}
    except Exception as e:
        log_and_print(e)
        return None, {"status":"error","error":str(e)}

# ---------------- Predict all horizons wrapper ----------------
def predict_all_for_asset(asset):
    feats = build_feature_table(asset)
    results = {}
    if feats is None or feats.empty:
        for k in ["day_now","day_next","week_now","week_next","month_now","month_next"]:
            results[k] = (None, {"status":"no_features"})
        return results
    # day now: horizon 1, day next: horizon 2...
    p1, m1 = train_predict_model(feats, 1)
    p2, m2 = train_predict_model(feats, 2)
    p7, m7 = train_predict_model(feats, 7)
    p14, m14 = train_predict_model(feats, 14)
    p30, m30 = train_predict_model(feats, 30)
    p60, m60 = train_predict_model(feats, 60)
    results["day_now"] = (p1, m1)
    results["day_next"] = (p2, m2)
    results["week_now"] = (p7, m7)
    results["week_next"] = (p14, m14)
    results["month_now"] = (p30, m30)
    results["month_next"] = (p60, m60)
    return results

# ---------------- Small plotting helpers ----------------
def plot_indicator_table(asset):
    df = build_feature_table(asset)
    if df is None or df.empty:
        st.write("Keine Daten für Indikatoren.")
        return
    # show last row indicators
    last = df.tail(1).iloc[0]
    cols = {}
    for k in ["Close","RSI","MACD_DIFF","EMA20","EMA50","EMA200","SMA20","SMA50","SMA200","ATR","DXY","VIX","FearGreed"]:
        if k in df.columns:
            cols[k]= last.get(k, np.nan)
    st.json(cols)

# ---------------- UI: single page ----------------
def main_page():
    st.header("Kompakte Prognoseseite — ML-basiert mit Multi-News und Makro")
    st.markdown("Vorhersagen werden mit Gradient Boosting (wenn genug Daten) trainiert. Bei Datenknappheit greifen robuste Fallbacks (LinearRegression / heuristisch).")

    # macro snapshot
    st.subheader("Makro-Snapshot")
    macro = fetch_macro_timeseries(days=365*5)
    if macro is None or macro.empty:
        st.info("Makro-Daten nicht verfügbar.")
    else:
        try:
            last = macro.tail(1)
            st.write(f"DXY: {float(last['DXY']):.4f} | VIX: {float(last['VIX']):.2f} | BTC: {float(last['BTC']):.2f}")
        except Exception:
            st.write(macro.tail(1).to_dict())

    # news + events
    st.subheader("Top News (kombiniert)")
    news = fetch_news_multi(limit=12)
    for n in news[:8]:
        title = n.get("title","")
        src = n.get("source","")
        link = n.get("link","")
        if link:
            st.markdown(f"- **[{src}]** [{title}]({link})")
        else:
            st.markdown(f"- **[{src}]** {title}")

    st.subheader("Wirtschaftliche Ereignisse (Investing.com, next 60d)")
    events = fetch_investing_events(days_ahead=60)
    if not events:
        st.info("Keine Events (Investing.com konnte nicht geparst werden).")
    else:
        evdf = pd.DataFrame(events[:20])
        st.dataframe(evdf)

    # Predictions
    st.subheader("Preisprognosen (heute / morgen, diese Woche / nächste Woche, dieser Monat / nächster Monat)")
    for asset in ASSETS:
        st.markdown(f"## {asset}")
        with st.spinner(f"Berechne Prognosen für {asset}..."):
            preds = predict_all_for_asset(asset)
        # display table
        rows = []
        mapping = [("Day (now)","day_now"),("Day (next)","day_next"),("Week (now)","week_now"),("Week (next)","week_next"),("Month (now)","month_now"),("Month (next)","month_next")]
        for label,key in mapping:
            val, meta = preds.get(key, (None,{}))
            if val is None:
                info = meta.get("status") or meta.get("model") or meta.get("n") or ""
                rows.append({"Period":label, "Prediction":"n/a", "Info": str(info)})
            else:
                if isinstance(meta, dict) and meta.get("r2") is not None:
                    rows.append({"Period":label, "Prediction":f"{val*100:.2f}%", "Info":f"model={meta.get('model')} r2={meta.get('r2'):.3f}"})
                else:
                    rows.append({"Period":label, "Prediction":f"{val*100:.2f}%", "Info":f"model={meta.get('model','heuristic')} rows={meta.get('n',0)}"})
        st.table(pd.DataFrame(rows))

        # optional: indicators snapshot
        st.subheader("Letzte Indikatoren (Kurzüberblick)")
        plot_indicator_table(asset)

    st.markdown("---")
    st.caption("⚠️ Indikative Prognosen — keine Anlageberatung. Modelle sind datengetrieben und können Fehler aufweisen.")

# ---------------- Run ----------------
if __name__ == "__main__":
    main_page()
