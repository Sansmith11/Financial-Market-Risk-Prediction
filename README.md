# Financial Market Risk Pipeline — Render Deployment

## Stack
- **Backend**: Flask + Gunicorn
- **Models**: `advanced-distributions==0.2.1` (7 distributions)
- **Data**: `yfinance` (live NSE / global tickers)
- **Frontend**: Dark dashboard embedded in `app.py` (no separate template files)

## Files
```
app.py            ← Flask app + pipeline + dashboard HTML
requirements.txt  ← All dependencies
render.yaml       ← Render service config
```

## Deploy on Render

1. Push this folder to a GitHub repo.
2. Go to [render.com](https://render.com) → **New → Web Service**.
3. Connect your GitHub repo.
4. Render auto-detects `render.yaml` — click **Deploy**.

### Manual settings (if not using render.yaml)
| Field          | Value                                                              |
|----------------|--------------------------------------------------------------------|
| Runtime        | Python 3                                                           |
| Build Command  | `pip install -r requirements.txt`                                  |
| Start Command  | `gunicorn app:app --workers 2 --timeout 120 --bind 0.0.0.0:$PORT` |
| Python Version | `3.11.9` (set as env var `PYTHON_VERSION`)                         |

## Usage

Open your Render URL → enter a ticker (e.g. `RELIANCE.NS`, `TCS.NS`, `AAPL`) → click **Analyze**.

### API endpoint
```
GET /api/analyze?ticker=RELIANCE.NS
```
Returns a JSON report with all 7 model outputs + BUY/HOLD/SELL decision.

```
GET /health
```
Returns `{"status": "ok"}` — useful for Render health checks.

## Notes
- First analysis takes ~20–40s (yfinance download + 7 model fits).
- Gunicorn `--timeout 120` covers the worst case.
- `--workers 2` keeps memory usage low on Render's free tier.
