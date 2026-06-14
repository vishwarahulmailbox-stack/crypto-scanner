# ₿ Crypto CE Scanner

Chandelier Exit buy signal scanner for crypto — Heikin Ashi 4H bars, ATR(1)×3, ZLSMA(50).  
Auto-publishes HTML reports to GitHub Pages via GitHub Actions.

---

## Setup

1. **Clone repo & add files**
   - `crypto_scanner.py` — main scanner
   - `crypto_list.csv` — your coin list (one symbol per line, e.g. `BTC-USD`)

2. **Install dependencies**
   ```
   pip install yfinance pandas numpy pytz
   ```

3. **Run locally**
   ```
   python crypto_scanner.py
   ```
   Report saved to `docs/crypto_signals_YYYY-MM-DD_HH-MM.html`

---

Enable **GitHub Pages** → source: `main` branch → `/docs` folder.

---

## crypto_list.csv format

```
symbol
BTC-USD
ETH-USD
SOL-USD
BNB-USD
```

---

## Signal Logic

| Parameter | Value |
|-----------|-------|
| Timeframe | 4H (UTC buckets: 00/04/08/12/16/20) |
| Candle type | Heikin Ashi |
| CE Length | 1 |
| CE Multiplier | ATR × 3.0 |
| ZLSMA Length | 50 |
| Fresh signal window | Last 2 days |

Buy signal = Chandelier Exit direction flips **short → long** on HA 4H bar.

---

## Output

- `docs/crypto_signals_YYYY-MM-DD_HH-MM.html` — timestamped report
- `docs/index.html` — auto-generated index of all reports (latest on top)
