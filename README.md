# 📊 Financial Market Risk Intelligence Platform

<div align="center">

![Banner](docs/banner.png)

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.x-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)](https://scipy.org)
[![Render](https://img.shields.io/badge/Deploy-Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://render.com)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)]()
[![Markets](https://img.shields.io/badge/Markets-NSE%20%7C%20NASDAQ%20%7C%20NYSE-blue?style=flat-square)]()

> **A next-generation quantitative finance platform combining seven advanced probability distributions to detect market regimes, tail events, and stress conditions — producing actionable BUY / HOLD / SELL signals.**

</div>

---

## 🚀 Overview

**Financial Market Risk Intelligence Platform** is a quantitative finance application that goes beyond traditional technical analysis. It fetches live market data from Yahoo Finance, performs advanced statistical feature engineering, and analyzes market behavior using **seven specialized probability distributions** to produce intelligent investment signals.

### ✅ What Makes This Different

| Traditional Tools | This Platform |
|---|---|
| RSI, MACD, Moving Averages | Distribution-Based Risk Analysis |
| Linear & Symmetric Models | Fractal & Tail-Aware Models |
| Fails in Market Crashes | Designed for Extreme Events |
| No Uncertainty Quantification | Full Uncertainty & Regime Modeling |
| Single indicator signals | 7-Model Weighted Fusion Engine |

---

## 🎯 Problem Statement

Financial markets are highly **nonlinear systems**. Traditional indicators such as RSI, MACD, and Moving Averages often fail during:

- 📉 Market Crashes
- 🦢 Black Swan Events
- 🌪 High Volatility Regimes

This platform addresses those challenges using:

- **Fractional Dynamics** — Long memory detection
- **Fractal Analysis** — Market complexity modeling
- **Tail Distributions** — Extreme event detection
- **Quantile Risk Models** — VaR & CVaR computation
- **Davies Stress Modeling** — Regime identification

---

## ✨ Features

### 📈 Live Market Data

Supports live OHLCV data from Yahoo Finance:

| Exchange | Example Tickers |
|---|---|
| NSE (India) | `RELIANCE.NS` `TCS.NS` `INFY.NS` `HDFCBANK.NS` |
| NASDAQ | `AAPL` `MSFT` `TSLA` `AMZN` |
| NYSE | Any valid NYSE ticker symbol |

### 🧠 Seven Advanced Distribution Models

| # | Model | Purpose |
|---|---|---|
| 1 | **Fractional Distribution** | Long Memory Detection |
| 2 | **Fractal Distribution** | Market Complexity Modeling |
| 3 | **Sinh-Arcsinh Distribution** | Skewness & Heavy Tail Modeling |
| 4 | **Slash Distribution** | Extreme Event & Crash Detection |
| 5 | **JohnsonSU (Neural Spline)** | Uncertainty Estimation |
| 6 | **Quantile Distribution** | VaR / CVaR Risk Management |
| 7 | **Davies Distribution** ⭐ | Market Stress Regime Detection |

### 📊 Interactive Dashboard

- Real-Time Analysis
- Risk Gauges & Stress Indicators
- Price Charts & Distribution Histograms
- Signal Breakdown & Model Explainability

### 🎯 Intelligent Decision Engine

Produces **BUY / HOLD / SELL** based on aggregated Bull Score, Bear Score, Stress Score, Tail Risk, Uncertainty, Memory Effects, and Market Structure.

---

## 📸 Dashboard Preview

| Dashboard | Price Chart |
|---|---|
| ![Dashboard](docs/dashboard.png) | ![Price Chart](docs/price_chart.png) |

| Stress Gauge | Signal Breakdown |
|---|---|
| ![Stress Gauge](docs/stress_gauge.png) | ![Signals](docs/signals.png) |

---

## 🏗 System Architecture

### 📂 Project Structure

```
financial_market_risk_platform/
│
├── app.py                    # Flask application entry point
├── requirements.txt          # Python dependencies
├── render.yaml               # Render deployment config
├── README.md
│
├── docs/
│   ├── banner.png
│   ├── dashboard.png
│   ├── architecture.png
│   └── workflow.png
│
├── static/                   # CSS / JS / assets
├── templates/                # Jinja2 HTML templates
└── models/                   # Distribution model modules
```

---

## ⚙️ Data Pipeline

```
Step 1: Market Data Collection
    └── yf.download() → Open, High, Low, Close, Volume

Step 2: Data Cleaning
    └── Remove NaN, duplicates, outliers, invalid records

Step 3: Feature Engineering
    ├── Log Returns:  r_t = ln(P_t / P_t-1)
    ├── Volatility:   σ = std(log_returns)
    ├── Momentum:     (Price − Mean) / Std
    └── RSI:          100 − 100 / (1 + RS)

Step 4: Distribution Fitting
    └── Seven models fitted to return distribution

Step 5: Decision Fusion
    └── Weighted aggregation → BUY / HOLD / SELL
```

---

## 🧠 Distribution Models — Deep Dive

### 1️⃣ Fractional Distribution — Long Memory Detection

Detects persistence and mean reversion in the return series.

| Hurst Proxy | Market Regime |
|---|---|
| > 0.55 | Trending Market |
| ~0.50 | Random Walk |
| < 0.45 | Mean-Reverting |

**Outputs:** Hurst Proxy · Memory Score · Regime Classification

---

### 2️⃣ Fractal Distribution — Market Complexity

Measures self-similarity and structural complexity across time scales.

**Regimes:** `Structured` → `Complex` → `Chaotic`

**Outputs:** Complexity Score · Structure Classification

---

### 3️⃣ Sinh-Arcsinh Distribution — Asymmetry & Heavy Tails

Captures distributional shape beyond Gaussian assumptions.

**Outputs:** Skewness · Kurtosis · Tail Shape

---

### 4️⃣ Slash Distribution — Extreme Event Detection

Specifically tuned for detecting crash probabilities in fat-tail scenarios.

**Outputs:** Extreme Event Probability · Crash Risk · Entropy

---

### 5️⃣ JohnsonSU Neural Spline — Uncertainty Quantification

Approximates complex return distributions for probabilistic forecasting.

**Outputs:** Quantiles · Forecast Spread · Uncertainty Score

---

### 6️⃣ Quantile Distribution — VaR & CVaR Risk

Core risk management model for loss quantification.

| Metric | Definition |
|---|---|
| **VaR 95** | Maximum expected loss under normal conditions |
| **VaR 99** | Maximum expected loss at 99% confidence |
| **CVaR 95** | Expected loss *beyond* the VaR threshold |

**Outputs:** VaR 95 · VaR 99 · CVaR · Tail Risk Score

---

### 7️⃣ Davies Distribution ⭐ — Market Stress Intelligence

**Flagship model.** Identifies the current market stress regime with high sensitivity.

| Stress Regime | Condition |
|---|---|
| 🟢 Normal | Low volatility, stable structure |
| 🟡 Caution | Elevated risk, watch closely |
| 🟠 Stress | Significant market disruption |
| 🔴 Crisis | Extreme tail event underway |

**Outputs:** Stress Score · Volatility Ratio · Stress Regime

---

## 🎯 Decision Engine

### Weighted Model Fusion

```
Davies Distribution     ████████████████████████░  25%
Slash Distribution      ████████████████████░░░░░  20%
JohnsonSU               ████████████████████░░░░░  20%
Quantile Distribution   ██████████████████░░░░░░░  18%
Fractional Distribution ██████████░░░░░░░░░░░░░░░  10%
Fractal Distribution    ███████░░░░░░░░░░░░░░░░░░   7%
```

### Signal Logic

```python
Net Score = Bull Score − Bear Score

if Net > 0.08:
    decision = "BUY"
elif Net < -0.08:
    decision = "SELL"
else:
    decision = "HOLD"
```

---

## 🔌 API Documentation

### Analyze Stock

```http
GET /api/analyze?ticker=RELIANCE.NS
GET /api/analyze?ticker=AAPL
```

**Response:**
```json
{
  "ticker": "AAPL",
  "decision": {
    "decision": "BUY",
    "confidence": 0.81
  }
}
```

### Health Check

```http
GET /health
```

**Response:**
```json
{
  "status": "ok"
}
```

---

## 🚀 Local Installation

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/financial-market-risk-platform.git
cd financial-market-risk-platform
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / Mac
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run Application

```bash
python app.py
```

Open in browser: `http://localhost:5000`

---

## ☁️ Deploy on Render

```yaml
# render.yaml
build_command: pip install -r requirements.txt
start_command: gunicorn app:app --workers 2 --timeout 120
python_version: "3.11"
```

---

## 📦 Dependencies

```txt
Flask
Gunicorn
NumPy
Pandas
SciPy
yFinance
curl_cffi
advanced-distributions
```

---

## 🔮 Future Roadmap

### Phase 2 — Real-Time Streaming
- [ ] Kafka Streaming Integration
- [ ] Real-Time Market Feed
- [ ] Portfolio Analytics
- [ ] Risk Alerts & Notifications

### Phase 3 — Deep Learning
- [ ] LSTM Price Prediction
- [ ] Transformer-Based Models
- [ ] Reinforcement Learning Signals
- [ ] Explainable AI Layer

### Phase 4 — Multi-Asset
- [ ] Options Analytics
- [ ] Futures Analytics
- [ ] Portfolio Optimization
- [ ] Cross-Asset Risk Modeling

### Phase 5 — Institutional
- [ ] Institutional Risk Engine
- [ ] Stress Testing Framework
- [ ] Basel III Metrics
- [ ] VaR Backtesting

---

## 🛡 Disclaimer

> ⚠️ This software is intended for **research**, **education**, and **quantitative analysis** purposes only.
> It is **NOT financial advice**. Users are solely responsible for their own investment decisions.

---

## 👨‍💻 Author

**Sandeep Kumar**
M.Sc Computer Science — NIT Tiruchirappalli

*Quantitative Finance · Data Science · Machine Learning · Risk Analytics*

---

## ⭐ Support

If you find this project useful:

- ⭐ **Star** the repository
- 🍴 **Fork** and contribute
- 🐛 **Report issues** via GitHub Issues
- 📢 **Share** with the quant finance community

---

<div align="center">

Made with ❤️ by Sandeep Kumar

</div>
