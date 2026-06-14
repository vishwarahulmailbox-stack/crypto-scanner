"""
Chandelier Exit Scanner — GitHub Actions version
Pine Script v6 exact port | Heikin Ashi 4H | ATR(1)×3 + ZLSMA(50)
Crypto version: 24/7 UTC 4H bars | Output: docs/crypto_signals_<timestamp>.html
Light theme | 12-hour AM/PM display
"""

import yfinance as yf
import pandas as pd
import numpy as np
import pytz
import os
import sys
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

IST = pytz.timezone("Asia/Kolkata")
UTC = pytz.utc

CE_LENGTH    = 1
CE_MULT      = 3.0
ZLSMA_LENGTH = 50
MAX_WORKERS  = 8
SIGNAL_DAYS  = 2

def load_symbols(csv_path="crypto_list.csv"):
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found!")
        sys.exit(1)
    df = pd.read_csv(csv_path)
    symbol_col = None
    for col in df.columns:
        if col.strip().lower() in ["symbol","ticker","scrip","nse symbol","nse_symbol"]:
            symbol_col = col
            break
    if symbol_col is None:
        symbol_col = df.columns[0]
    symbols = df[symbol_col].dropna().astype(str).str.strip().tolist()
    symbols = [s if s.endswith("-USD") else s + ".NS" for s in symbols]
    return list(dict.fromkeys(symbols))

def build_4h_bars_crypto(df_1h):
    """24/7 UTC-based 4H bars. Buckets: 00,04,08,12,16,20 UTC"""
    if df_1h.index.tzinfo is None:
        df_1h.index = df_1h.index.tz_localize("UTC")
    else:
        df_1h.index = df_1h.index.tz_convert(UTC)
    df_1h = df_1h.copy()
    df_1h["_bucket"] = df_1h.index.floor("4h")
    records = []
    for bucket, grp in df_1h.groupby("_bucket"):
        if len(grp) < 1:
            continue
        records.append({
            "datetime": bucket,
            "Open":  float(grp["Open"].iloc[0]),
            "High":  float(grp["High"].max()),
            "Low":   float(grp["Low"].min()),
            "Close": float(grp["Close"].iloc[-1]),
        })
    if not records:
        return None
    df_4h = pd.DataFrame(records).set_index("datetime")
    df_4h.index = df_4h.index.tz_convert(IST)
    return df_4h

def calc_ha(df):
    n = len(df)
    ho, hh, hl, hc = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    o, h, l, c = df["Open"].values, df["High"].values, df["Low"].values, df["Close"].values
    hc[0] = (o[0]+h[0]+l[0]+c[0])/4
    ho[0] = (o[0]+c[0])/2
    hh[0] = max(h[0], ho[0], hc[0])
    hl[0] = min(l[0], ho[0], hc[0])
    for i in range(1, n):
        hc[i] = (o[i]+h[i]+l[i]+c[i])/4
        ho[i] = (ho[i-1]+hc[i-1])/2
        hh[i] = max(h[i], ho[i], hc[i])
        hl[i] = min(l[i], ho[i], hc[i])
    return pd.DataFrame({"Open":ho,"High":hh,"Low":hl,"Close":hc}, index=df.index)

def wilder_atr(high, low, close, length):
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr  = np.full(len(tr), np.nan)
    vals = tr.values
    idx  = int(np.argmax(~np.isnan(vals)))
    if idx + length <= len(vals):
        atr[idx+length-1] = np.mean(vals[idx:idx+length])
        for i in range(idx+length, len(vals)):
            atr[i] = (atr[i-1]*(length-1) + vals[i]) / length
    return pd.Series(atr, index=tr.index)

def calc_ce(ha, length=1, mult=3.0, use_close=True):
    close = ha["Close"]
    high  = ha["High"]
    low   = ha["Low"]
    atr   = mult * wilder_atr(high, low, close, length)
    highest = close.rolling(length).max() if use_close else high.rolling(length).max()
    lowest  = close.rolling(length).min() if use_close else low.rolling(length).min()
    ls = (highest - atr).values.copy()
    ss = (lowest  + atr).values.copy()
    c  = close.values
    for i in range(1, len(c)):
        lp = ls[i-1] if not np.isnan(ls[i-1]) else ls[i]
        sp = ss[i-1] if not np.isnan(ss[i-1]) else ss[i]
        if not np.isnan(ls[i]):
            ls[i] = max(ls[i], lp) if c[i-1] > lp else ls[i]
            ss[i] = min(ss[i], sp) if c[i-1] < sp else ss[i]
    direction = np.ones(len(c), dtype=int)
    for i in range(1, len(c)):
        lp = ls[i-1] if not np.isnan(ls[i-1]) else ls[i]
        sp = ss[i-1] if not np.isnan(ss[i-1]) else ss[i]
        if   c[i] > sp: direction[i] = 1
        elif c[i] < lp: direction[i] = -1
        else:           direction[i] = direction[i-1]
    dir_s   = pd.Series(direction, index=ha.index)
    buy_sig = (dir_s == 1) & (dir_s.shift(1) == -1)
    return buy_sig

def calc_zlsma(series, length=50):
    def linreg(s, n):
        result = np.full(len(s), np.nan)
        arr = s.values if hasattr(s, 'values') else np.array(s)
        for i in range(n-1, len(arr)):
            y = arr[i-n+1:i+1]
            if np.any(np.isnan(y)): continue
            x  = np.arange(n, dtype=float)
            xm, ym = x.mean(), y.mean()
            denom = np.sum((x-xm)**2)
            if denom == 0: continue
            slope     = np.sum((x-xm)*(y-ym)) / denom
            result[i] = (ym - slope*xm) + slope*(n-1)
        return pd.Series(result, index=s.index if hasattr(s,'index') else range(len(result)))
    lsma  = linreg(series, length)
    lsma2 = linreg(lsma,   length)
    return 2*lsma - lsma2

def is_fresh(signal_ts, now_ist, max_days=2):
    sig_date = signal_ts.date()
    now_date = now_ist.date()
    delta = (now_date - sig_date).days
    return 0 <= delta <= max_days

def format_price(price):
    if price is None:
        return "—"
    if price >= 1:
        return f"${price:,.4f}"
    else:
        return f"${price:.8f}"

def scan(symbol):
    try:
        df_1h = yf.download(symbol, period="60d", interval="1h",
                            progress=False, auto_adjust=True)
        if df_1h is None or len(df_1h) < 30:
            return None
        if isinstance(df_1h.columns, pd.MultiIndex):
            df_1h.columns = df_1h.columns.get_level_values(0)
        df_1h = df_1h[["Open","High","Low","Close"]].dropna()
        df_4h = build_4h_bars_crypto(df_1h)
        if df_4h is None or len(df_4h) < ZLSMA_LENGTH + 5:
            return None
        # Always exclude forming bar (crypto 24/7)
        df_4h = df_4h.iloc[:-1]
        now_ist = datetime.now(IST)
        ha      = calc_ha(df_4h)
        buy_sig = calc_ce(ha, length=CE_LENGTH, mult=CE_MULT)
        last_sig   = buy_sig.iloc[-1]
        second_sig = buy_sig.iloc[-2] if len(buy_sig) > 1 else False
        last_fresh   = last_sig   and is_fresh(buy_sig.index[-1], now_ist, SIGNAL_DAYS)
        second_fresh = second_sig and is_fresh(buy_sig.index[-2], now_ist, SIGNAL_DAYS)
        if not (last_fresh or second_fresh):
            return None
        if last_fresh:
            label    = "Last Bar"
            sig_time = buy_sig.index[-1]
        else:
            label    = "2nd Last Bar"
            sig_time = buy_sig.index[-2]
        zlsma_series = calc_zlsma(df_4h["Close"], length=ZLSMA_LENGTH)
        zlsma_val    = round(float(zlsma_series.iloc[-1]), 6) if not np.isnan(zlsma_series.iloc[-1]) else None
        price        = round(float(df_4h["Close"].iloc[-1]), 6)
        diff         = round(price - zlsma_val, 6)      if zlsma_val else None
        diff_pct     = round(diff / zlsma_val * 100, 2) if zlsma_val else None
        # 12-hour AM/PM display
        sig_time_str = sig_time.strftime("%d %b %I:%M %p")
        return {
            "symbol":   symbol,
            "price":    price,
            "bar":      label,
            "sig_ts":   sig_time,
            "time":     sig_time_str,
            "zlsma":    zlsma_val,
            "diff":     diff,
            "diff_pct": diff_pct,
        }
    except Exception:
        pass
    return None

def make_html(results, scan_time, total):
    rows = ""
    for i, r in enumerate(results, 1):
        badge_bg    = "#16a34a" if r["bar"] == "Last Bar" else "#ea580c"
        badge_color = "#ffffff"
        sym_tv = r['symbol'].replace("-USD", "USD").replace("-", "")
        tv_url = f"https://www.tradingview.com/chart/?symbol=BINANCE%3A{sym_tv}"
        price_str = format_price(r["price"])
        if r["diff"] is not None:
            diff_color = "#16a34a" if r["diff"] > 0 else "#dc2626"
            diff_arrow = "▲" if r["diff"] > 0 else "▼"
            zlsma_str  = format_price(r["zlsma"])
            diff_str   = f'<span style="color:{diff_color};font-weight:600">{diff_arrow} {format_price(abs(r["diff"]))} ({abs(r["diff_pct"])}%)</span>'
        else:
            zlsma_str = diff_str = "—"
        rows += f"""<tr>
            <td class="num">{i}</td>
            <td class="sym"><a href="{tv_url}" target="_blank" class="tv-link">{r['symbol']}<span class="tv-icon">↗</span></a></td>
            <td class="price">{price_str}</td>
            <td><span class="badge" style="background:{badge_bg};color:{badge_color}">{r['bar']}</span></td>
            <td class="time">{r['time']}</td>
            <td class="zlsma">{zlsma_str}</td>
            <td class="diff">{diff_str}</td></tr>"""

    count = len(results)
    empty = '<div class="empty"><div style="font-size:40px;margin-bottom:12px">🔍</div><p>No fresh Crypto buy signals found.</p></div>'
    table = f'<table><thead><tr><th>#</th><th>Symbol</th><th>Price (USD)</th><th>Signal</th><th>Bar Time (IST)</th><th>ZLSMA (50)</th><th>Price vs ZLSMA</th></tr></thead><tbody>{rows}</tbody></table>' if results else empty

    cnt_color = "#16a34a" if count > 0 else "#dc2626"

    return f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="300">
<title>Crypto Scanner — {scan_time}</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@400;500;600;700&display=swap');
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#f8fafc;color:#0f172a;font-family:'Inter',sans-serif;padding:32px 16px;min-height:100vh}}
.hdr{{max-width:1160px;margin:0 auto 28px;display:flex;justify-content:space-between;align-items:center;background:#ffffff;border:1px solid #e2e8f0;border-radius:16px;padding:24px 28px;flex-wrap:wrap;gap:16px;box-shadow:0 1px 4px rgba(0,0,0,.06)}}
.hdr-left h1{{font-size:24px;font-weight:700;color:#0f172a;letter-spacing:-0.5px}}
.hdr-left p{{font-size:12px;color:#64748b;margin-top:5px;font-family:'JetBrains Mono',monospace}}
.meta{{text-align:right}}
.cnt{{font-size:44px;font-weight:700;color:{cnt_color};font-family:'Inter',sans-serif;line-height:1}}
.cnt-label{{font-size:12px;color:#64748b;margin-top:2px;font-family:'JetBrains Mono',monospace}}
.scanned{{font-size:11px;color:#94a3b8;margin-top:2px;font-family:'JetBrains Mono',monospace}}
.scan-time{{font-size:11px;color:#94a3b8;margin-top:6px;font-family:'JetBrains Mono',monospace}}
.wrap{{max-width:1160px;margin:0 auto}}
table{{width:100%;border-collapse:separate;border-spacing:0;border-radius:14px;overflow:hidden;border:1px solid #e2e8f0;background:#ffffff;box-shadow:0 1px 4px rgba(0,0,0,.05)}}
thead tr{{background:#f1f5f9}}
th{{padding:13px 18px;font-size:10px;letter-spacing:.8px;text-transform:uppercase;color:#64748b;text-align:left;font-family:'JetBrains Mono',monospace;white-space:nowrap;border-bottom:1px solid #e2e8f0;font-weight:600}}
tbody tr{{border-bottom:1px solid #f1f5f9;transition:background .12s}}
tbody tr:last-child{{border-bottom:none}}
tbody tr:hover{{background:#f8fafc}}
td{{padding:15px 18px;font-size:14px;vertical-align:middle}}
td.num{{color:#cbd5e1;font-family:'JetBrains Mono',monospace;font-size:12px;width:40px;font-weight:600}}
td.sym{{font-weight:700;font-size:15px;color:#0f172a}}
td.price{{font-family:'JetBrains Mono',monospace;color:#1d4ed8;font-size:14px;font-weight:600}}
td.time{{font-family:'JetBrains Mono',monospace;font-size:13px;color:#64748b}}
td.zlsma{{font-family:'JetBrains Mono',monospace;color:#7c3aed;font-size:13px;font-weight:500}}
td.diff{{font-family:'JetBrains Mono',monospace;font-size:13px}}
.tv-link{{color:#0f172a;text-decoration:none;display:inline-flex;align-items:center;gap:6px}}
.tv-link:hover{{color:#1d4ed8}}
.tv-icon{{font-size:11px;color:#cbd5e1;transition:color .15s}}
.tv-link:hover .tv-icon{{color:#1d4ed8}}
.badge{{display:inline-block;padding:4px 12px;border-radius:6px;font-size:10px;font-weight:700;font-family:'JetBrains Mono',monospace;letter-spacing:.4px}}
.empty{{margin:60px auto;text-align:center;color:#94a3b8;font-family:'JetBrains Mono',monospace}}
.foot{{max-width:1160px;margin:20px auto 0;font-size:11px;color:#94a3b8;font-family:'JetBrains Mono',monospace;text-align:center;padding-top:16px}}
</style></head><body>
<div class="hdr">
  <div class="hdr-left">
    <h1>₿ Crypto CE Scanner</h1>
    <p>4H UTC bars &nbsp;·&nbsp; Heikin Ashi &nbsp;·&nbsp; ATR(1)×3 &nbsp;·&nbsp; ZLSMA(50) &nbsp;·&nbsp; 24/7 markets</p>
  </div>
  <div class="meta">
    <div class="cnt">{count}</div>
    <div class="cnt-label">signals found</div>
    <div class="scanned">{total} coins scanned</div>
    <div class="scan-time">{scan_time}</div>
  </div>
</div>
<div class="wrap">{table}</div>
<div class="foot">
  Click symbol → TradingView 4H chart (Binance) &nbsp;·&nbsp; CE buy = direction flip short→long on HA 4H<br>
  ZLSMA = 2×linreg(close,50) − linreg(linreg(close,50),50) &nbsp;·&nbsp; Bars: UTC 00/04/08/12/16/20 &nbsp;·&nbsp; Fresh: last {SIGNAL_DAYS} days
</div>
</body></html>"""

def main():
    now = datetime.now(IST)
    # 12-hour AM/PM for display everywhere
    scan_time = now.strftime("%d %b %Y, %I:%M %p IST")
    print(f"Crypto Scanner starting — {scan_time}")
    print(f"Market: 24/7 (crypto — no bar open/close check)")

    symbols = load_symbols("crypto_list.csv")
    total   = len(symbols)
    print(f"Scanning {total} coins...")

    results = []
    done    = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(scan, sym): sym for sym in symbols}
        for f in as_completed(futures):
            done += 1
            print(f"[{done}/{total}]", end="\r")
            r = f.result()
            if r:
                results.append(r)
                print(f"  SIGNAL: {r['symbol']} — {r['bar']} — {r['time']}")

    results.sort(key=lambda x: (
        0 if x["bar"] == "Last Bar" else 1,
        -x["sig_ts"].timestamp(),
        x["symbol"]
    ))

    print(f"\nSignals found: {len(results)} / {total}")

    html    = make_html(results, scan_time, total)
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")
    os.makedirs(out_dir, exist_ok=True)

    # Filename stays 24hr for correct alphabetical/chronological sorting
    fname    = f"crypto_signals_{now.strftime('%Y-%m-%d_%I-%M-%p')}.html"
    out_path = os.path.join(out_dir, fname)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  Report : {out_path}")

    # Auto-generate index.html
    all_files = sorted(
        [fn for fn in os.listdir(out_dir) if fn.startswith("crypto_signals_") and fn.endswith(".html")],
        reverse=True
    )

    def make_index(files, generated_at):
        rows = ""
        for i, fn in enumerate(files):
            try:
                ts_part = fn.replace("crypto_signals_", "").replace(".html", "")
                dt      = datetime.strptime(ts_part, "%Y-%m-%d_%I-%M-%p")
                dt_ist  = IST.localize(dt)
                # 12-hour AM/PM in index too
                label   = dt_ist.strftime("%d %b %Y, %I:%M %p IST")
            except Exception:
                label = fn
            badge = '<span style="background:#16a34a;color:#fff;font-size:10px;font-weight:700;padding:3px 8px;border-radius:4px;font-family:monospace;margin-left:10px">LATEST</span>' if i == 0 else ""
            rows += f"""<tr>
                <td class="num">{i+1}</td>
                <td class="lbl"><a href="{fn}" class="link">{label}{badge}</a></td>
                <td class="fn">{fn}</td>
            </tr>"""

        return f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Crypto Scanner — Reports</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@400;500;600;700&display=swap');
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#f8fafc;color:#0f172a;font-family:'Inter',sans-serif;padding:40px 16px;min-height:100vh}}
.hdr{{max-width:900px;margin:0 auto 28px;background:#ffffff;border:1px solid #e2e8f0;border-radius:16px;padding:24px 28px;box-shadow:0 1px 4px rgba(0,0,0,.06)}}
.hdr h1{{font-size:22px;font-weight:700;color:#0f172a}}
.hdr p{{font-size:12px;color:#64748b;margin-top:6px;font-family:'JetBrains Mono',monospace}}
table{{width:100%;max-width:900px;margin:0 auto;border-collapse:separate;border-spacing:0;border-radius:14px;overflow:hidden;border:1px solid #e2e8f0;background:#ffffff;box-shadow:0 1px 4px rgba(0,0,0,.05)}}
thead tr{{background:#f1f5f9}}
th{{padding:12px 18px;font-size:10px;letter-spacing:.8px;text-transform:uppercase;color:#64748b;text-align:left;font-family:'JetBrains Mono',monospace;border-bottom:1px solid #e2e8f0;font-weight:600}}
tbody tr{{border-bottom:1px solid #f1f5f9;transition:background .12s}}
tbody tr:last-child{{border-bottom:none}}
tbody tr:hover{{background:#f8fafc}}
td{{padding:14px 18px;font-size:14px;vertical-align:middle}}
td.num{{color:#cbd5e1;font-family:'JetBrains Mono',monospace;font-size:12px;width:40px;font-weight:600}}
td.fn{{font-family:'JetBrains Mono',monospace;font-size:11px;color:#94a3b8}}
.link{{color:#1d4ed8;text-decoration:none;font-weight:600}}
.link:hover{{text-decoration:underline}}
.foot{{max-width:900px;margin:20px auto 0;font-size:11px;color:#94a3b8;font-family:'JetBrains Mono',monospace;text-align:center;padding-top:16px}}
</style></head><body>
<div class="hdr">
  <h1>₿ Crypto Scanner — All Reports</h1>
  <p>Total {len(files)} report(s) &nbsp;·&nbsp; Generated: {generated_at}</p>
</div>
<table>
  <thead><tr><th>#</th><th>Scan Time</th><th>File</th></tr></thead>
  <tbody>{rows}</tbody>
</table>
<div class="foot">Latest report on top &nbsp;·&nbsp; Click to open report</div>
</body></html>"""

    index_html = make_index(all_files, scan_time)
    index_path = os.path.join(out_dir, "index.html")
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(index_html)
    print(f"  Index  : {index_path}")

if __name__ == "__main__":
    main()