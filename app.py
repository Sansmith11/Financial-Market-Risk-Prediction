"""
Financial Market Risk Pipeline — Flask Web App
Deploy on Render (Python 3.11)
"""

import os, json, warnings
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats as spstats
from flask import Flask, request, jsonify, render_template_string

warnings.filterwarnings("ignore")

from advanced_distributions.fractional_distribution    import FractionalDistribution
from advanced_distributions.fractal_distribution       import FractalDistribution
from advanced_distributions.sinh_arcsinh               import SinhArcsinhDistribution
from advanced_distributions.slash_distribution         import SlashDistribution
from advanced_distributions.neural_spline_distribution import NeuralSplineDistribution
from advanced_distributions.quantile_distribution      import QuantileDistribution
from advanced_distributions.davies_distribution        import DaviesDistribution

app = Flask(__name__)

# ── Common ticker corrections ────────────────────────────────────────────────
TICKER_CORRECTIONS = {
    "APPL": "AAPL", "GOOGL": "GOOGL", "GOOG": "GOOGL",
    "AMZON": "AMZN", "AMAZN": "AMZN", "MICROSFT": "MSFT",
    "MICROSFOT": "MSFT", "NFLX": "NFLX", "NETFLX": "NFLX",
    "TSLA": "TSLA", "TESTA": "TSLA", "META": "META",
    "RELINCE": "RELIANCE.NS", "RELIACE": "RELIANCE.NS",
    "TCS": "TCS.NS", "INFY": "INFY.NS", "WIPRO": "WIPRO.NS",
    "HDFCBANK": "HDFCBANK.NS", "ICICIBANK": "ICICIBANK.NS",
    "BAJFINANCE": "BAJFINANCE.NS", "SBIN": "SBIN.NS",
}

# ────────────────────────────────────────────────────────────────────────────
# PIPELINE FUNCTIONS
# ────────────────────────────────────────────────────────────────────────────

def resolve_ticker(ticker: str) -> tuple[str, str | None]:
    """Returns (resolved_ticker, suggestion_message_or_None)"""
    upper = ticker.upper().strip()
    if upper in TICKER_CORRECTIONS:
        corrected = TICKER_CORRECTIONS[upper]
        return corrected, f"'{upper}' auto-corrected to '{corrected}'"
    return upper, None


def fetch_data(ticker: str, period: str = "3y") -> pd.DataFrame:
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if df.empty:
        # Give a helpful hint based on the ticker
        hint = ""
        t = ticker.upper()
        if t == "APPL":
            hint = " Did you mean 'AAPL' (Apple Inc.)?"
        elif not t.endswith(".NS") and any(c.isalpha() for c in t):
            hint = f" For Indian stocks try '{t}.NS' (e.g. RELIANCE.NS, TCS.NS)."
        raise ValueError(f"No data found for ticker '{ticker}'.{hint}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(col[0]).lower() for col in df.columns]
    else:
        df.columns = [str(col).lower() for col in df.columns]
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().dropna()
    df = df[~df.index.duplicated(keep="first")]
    mask = (
        (df["high"] >= df["low"]) &
        (df["high"] >= df["close"]) &
        (df["low"]  <= df["close"]) &
        (df["volume"] > 0)
    )
    df = df[mask]
    log_ret = np.log(df["close"] / df["close"].shift(1)).dropna()
    z = (log_ret - log_ret.mean()) / log_ret.std()
    df = df.loc[z[np.abs(z) <= 5].index]
    return df.sort_index()


def engineer_features(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    df = df.copy()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["volatility"] = df["log_return"].rolling(window).std() * np.sqrt(252)
    rm = df["close"].rolling(window).mean()
    rs = df["close"].rolling(window).std()
    df["momentum"]   = (df["close"] - rm) / (rs + 1e-10)
    delta = df["close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"]        = 100 - 100 / (1 + gain / (loss + 1e-10))
    df["vol_zscore"] = (
        (df["volume"] - df["volume"].rolling(window).mean()) /
        (df["volume"].rolling(window).std() + 1e-10)
    )
    df.dropna(inplace=True)
    return df


def run_fractional(returns):
    pos  = np.abs(returns) + 1e-8
    dist = FractionalDistribution.fit(pos)
    hurst_proxy  = float(np.clip(0.5 + (dist.alpha - 1.5) * 0.1, 0.3, 0.8))
    memory_score = float(np.clip((hurst_proxy - 0.5) * 2, 0, 1))
    regime = ("trending" if hurst_proxy > 0.55 else
              "mean-reverting" if hurst_proxy < 0.45 else "random walk")
    return {
        "alpha": round(dist.alpha, 4), "beta": round(dist.beta, 4),
        "gamma": round(dist.gamma, 4), "mean": round(dist.mean(), 6),
        "std": round(dist.std(), 6), "skewness": round(dist.skewness(), 4),
        "kurtosis": round(dist.kurtosis(), 4),
        "hurst_proxy": round(hurst_proxy, 4),
        "memory_score": round(memory_score, 4), "regime": regime,
    }


def run_fractal(returns):
    pos    = np.abs(returns) + 1e-8
    params = FractalDistribution.fit(pos)
    D, lam = float(params[0]), float(params[1])
    dist   = FractalDistribution(D=D, lambda_=lam)
    complexity = float(np.clip(D - 1.0, 0, 1))
    structure  = ("chaotic" if D > 1.7 else "complex" if D > 1.4 else "structured")
    return {
        "D": round(D, 4), "lambda": round(lam, 4),
        "mean": round(dist.mean(), 6), "std": round(dist.std(), 6),
        "complexity_score": round(complexity, 4), "structure": structure,
    }


def run_sinh_arcsinh(returns):
    dist = SinhArcsinhDistribution.fit(returns)
    skew_dir = ("left-skewed" if dist.epsilon < -0.1 else
                "right-skewed" if dist.epsilon > 0.1 else "symmetric")
    return {
        "epsilon": round(dist.epsilon, 4), "delta": round(dist.delta, 4),
        "mu": round(dist.mu, 6), "sigma": round(dist.sigma, 6),
        "skewness": round(dist.skewness(), 4), "kurtosis": round(dist.kurtosis(), 4),
        "skewness_direction": skew_dir,
        "skew_score": round(float(np.tanh(dist.epsilon)), 4),
    }


def run_slash(returns):
    dist      = SlashDistribution.fit(returns)
    samples   = dist.rvs(size=5000, random_state=42)
    threshold = 3.0 * dist.sigma
    ext_prob  = float(np.mean(np.abs(samples - dist.mu) > threshold))
    crash_risk = ("high" if ext_prob > 0.05 else "medium" if ext_prob > 0.02 else "low")
    return {
        "mu": round(dist.mu, 6), "sigma": round(dist.sigma, 6),
        "extreme_event_prob": round(ext_prob, 4), "crash_risk": crash_risk,
        "median": round(dist.median(), 6), "entropy": round(dist.entropy(), 4),
    }


def run_neural_spline(returns):
    dist = NeuralSplineDistribution.fit(returns, knots=9)
    qlevels = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    qvals   = dist.ppf(np.array(qlevels))
    q_dict  = {f"q{int(q*100):02d}": round(float(v), 6)
               for q, v in zip(qlevels, qvals)}
    spread      = abs(q_dict["q95"] - q_dict["q05"])
    uncertainty = float(np.clip(spread * 50, 0, 1))
    return {
        **q_dict,
        "mean": round(dist.mean(), 6), "std": round(dist.std(), 6),
        "skewness": round(dist.skewness(), 4), "kurtosis": round(dist.kurtosis(), 4),
        "entropy": round(dist.entropy(), 4), "uncertainty": round(uncertainty, 4),
    }


def run_quantile(returns):
    dist    = QuantileDistribution.fit(returns)
    var_95  = float(dist.ppf(0.05))
    var_99  = float(dist.ppf(0.01))
    sims    = dist.rvs(size=10000, random_state=42)
    cvar_95 = float(np.mean(sims[sims <= var_95]))
    tail_risk   = float(np.clip(-cvar_95 * 10, 0, 1))
    var_breach  = float(np.mean(returns < var_95))
    return {
        "mu": round(dist.mu, 6), "sigma": round(dist.sigma, 6),
        "alpha_shape": round(dist.alpha, 4), "beta_shape": round(dist.beta, 4),
        "VaR_95": round(var_95, 6), "VaR_99": round(var_99, 6),
        "CVaR_95": round(cvar_95, 6),
        "tail_risk_score": round(tail_risk, 4),
        "var_breach_rate": round(var_breach, 4),
        "skewness": round(dist.skewness(), 4), "kurtosis": round(dist.kurtosis(), 4),
    }


def run_davies(returns, recent_window=20):
    pos    = np.abs(returns) + 1e-8
    params = DaviesDistribution.fit(pos)
    alpha, beta, theta, k = [float(p) for p in params]
    dist   = DaviesDistribution(alpha=alpha, beta=beta, theta=theta, k=k)
    recent_vol  = float(np.mean(np.abs(returns[-recent_window:])))
    overall_vol = float(np.mean(pos))
    vol_ratio   = recent_vol / (overall_vol + 1e-10)
    try:
        stress_cdf = float(dist.cdf(np.array([recent_vol]))[0])
    except Exception:
        stress_cdf = 0.5
    stress_score = float(np.clip(stress_cdf, 0, 1))
    regime = ("crisis" if stress_score > 0.85 else "stress" if stress_score > 0.65
              else "caution" if stress_score > 0.40 else "normal")
    return {
        "alpha": round(alpha, 4), "beta": round(beta, 4),
        "theta": round(theta, 4), "k": round(k, 4),
        "mean": round(dist.mean(), 6), "std": round(dist.std(), 6),
        "stress_score": round(stress_score, 4), "vol_ratio": round(vol_ratio, 4),
        "regime": regime,
    }


def make_decision(frac, fractal, sinh, slash, spline, quant, davies):
    bull = {
        "mem_trend":      1.0 if frac["regime"] == "trending" else 0.0,
        "low_complexity": 1.0 - fractal["complexity_score"],
        "right_skew":     max(0.0, sinh["skew_score"]),
        "low_tail":       1.0 - min(1.0, slash["extreme_event_prob"] * 10),
        "low_var":        1.0 - quant["tail_risk_score"],
        "low_stress":     1.0 - davies["stress_score"],
        "neural_up":      max(0.0, spline["q75"]) * 20,
    }
    bear = {
        "mean_rev":       1.0 if frac["regime"] == "mean-reverting" else 0.0,
        "high_complexity": fractal["complexity_score"],
        "left_skew":      max(0.0, -sinh["skew_score"]),
        "high_tail":      min(1.0, slash["extreme_event_prob"] * 10),
        "high_var":       quant["tail_risk_score"],
        "high_stress":    davies["stress_score"],
        "neural_dn":      max(0.0, -spline["q25"]) * 20,
    }
    W = {
        "mem_trend": 0.10, "mean_rev": 0.10,
        "low_complexity": 0.10, "high_complexity": 0.10,
        "right_skew": 0.08, "left_skew": 0.08,
        "low_tail": 0.15,  "high_tail": 0.15,
        "low_var":  0.15,  "high_var":  0.15,
        "low_stress": 0.20, "high_stress": 0.20,
        "neural_up": 0.22, "neural_dn": 0.22,
    }
    bs  = sum(bull[k] * W[k] for k in bull)
    brs = sum(bear[k] * W[k] for k in bear)
    net  = bs - brs
    conf = abs(net) / (bs + brs + 1e-6)
    decision = "BUY" if net > 0.12 else "SELL" if net < -0.12 else "HOLD"
    return {
        "decision": decision, "bull_score": round(bs, 4),
        "bear_score": round(brs, 4), "net_score": round(net, 4),
        "confidence": round(conf, 4),
    }


def run_pipeline(ticker: str) -> dict:
    ticker, suggestion = resolve_ticker(ticker)
    raw     = fetch_data(ticker)
    df      = clean_data(raw)
    df      = engineer_features(df)
    returns = df["log_return"].values
    prices  = df["close"].values

    frac    = run_fractional(returns)
    fractal = run_fractal(returns)
    sinh    = run_sinh_arcsinh(returns)
    slash   = run_slash(returns)
    spline  = run_neural_spline(returns)
    quant   = run_quantile(returns)
    davies  = run_davies(returns)
    decision = make_decision(frac, fractal, sinh, slash, spline, quant, davies)

    return {
        "ticker":        ticker,
        "suggestion":    suggestion,
        "n_sessions":    len(df),
        "mean_return":   round(returns.mean(), 6),
        "std_return":    round(returns.std(),  6),
        "fractional":    frac,
        "fractal":       fractal,
        "sinh_arcsinh":  sinh,
        "slash":         slash,
        "neural_spline": spline,
        "quantile":      quant,
        "davies":        davies,
        "decision":      decision,
        "_prices":       prices[-120:].tolist(),
        "_returns":      returns[-300:].tolist(),
    }


# ────────────────────────────────────────────────────────────────────────────
# HTML DASHBOARD
# ────────────────────────────────────────────────────────────────────────────

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Financial Risk Pipeline</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
  :root{--bg:#0d1117;--card:#161b22;--border:#30363d;--accent:#58a6ff;
        --green:#3fb950;--red:#f85149;--yellow:#d29922;--text:#e6edf3;--muted:#8b949e}
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;min-height:100vh}
  header{background:var(--card);border-bottom:1px solid var(--border);
         padding:1rem 2rem;display:flex;align-items:center;gap:1rem}
  header h1{font-size:1.3rem;font-weight:600;color:var(--accent)}
  header span{color:var(--muted);font-size:.9rem}
  .container{max-width:1200px;margin:0 auto;padding:1.5rem}
  .search-bar{display:flex;gap:.75rem;margin-bottom:.75rem}
  .search-bar input{flex:1;background:var(--card);border:1px solid var(--border);
                    color:var(--text);padding:.6rem 1rem;border-radius:6px;font-size:1rem}
  .search-bar input:focus{outline:none;border-color:var(--accent)}
  .search-bar button{background:var(--accent);color:#0d1117;font-weight:600;
                     border:none;padding:.6rem 1.4rem;border-radius:6px;cursor:pointer;font-size:1rem}
  .search-bar button:hover{opacity:.85}
  .examples{font-size:.8rem;color:var(--muted);margin-bottom:1rem}
  .examples span{color:var(--accent);cursor:pointer;margin-right:.5rem;
                 padding:.15rem .4rem;border:1px solid var(--border);border-radius:4px}
  .examples span:hover{border-color:var(--accent)}
  #status{color:var(--muted);font-size:.9rem;margin-bottom:1rem;min-height:1.2rem}
  #suggestion{color:var(--yellow);font-size:.85rem;margin-bottom:.75rem;
              background:#3a2e1a;border:1px solid #d29922;border-radius:6px;
              padding:.5rem .85rem;display:none}
  .grid-3{display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin-bottom:1rem}
  .grid-2{display:grid;grid-template-columns:repeat(2,1fr);gap:1rem;margin-bottom:1rem}
  @media(max-width:700px){.grid-3,.grid-2{grid-template-columns:1fr}}
  .card{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:1rem}
  .card h3{font-size:.75rem;text-transform:uppercase;color:var(--muted);margin-bottom:.5rem;letter-spacing:.05em}
  .metric{font-size:1.6rem;font-weight:700}
  .badge{display:inline-block;padding:.2rem .6rem;border-radius:4px;font-size:.8rem;font-weight:600}
  .BUY{background:#1a3a1f;color:var(--green)}
  .SELL{background:#3a1a1a;color:var(--red)}
  .HOLD{background:#3a2e1a;color:var(--yellow)}
  .trending,.structured,.low,.normal,.right-skewed{background:#1a2a3a;color:var(--accent)}
  .mean-reverting,.chaotic,.high,.crisis,.left-skewed{background:#3a1a1a;color:var(--red)}
  .random,.complex,.medium,.stress,.caution,.symmetric{background:#3a2e1a;color:var(--yellow)}
  .section-title{font-size:1rem;font-weight:600;color:var(--accent);
                 margin:1rem 0 .5rem;padding-bottom:.4rem;border-bottom:1px solid var(--border)}
  .kv-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:.4rem}
  .kv{display:flex;justify-content:space-between;font-size:.82rem;
      background:#0d1117;padding:.3rem .6rem;border-radius:4px}
  .kv .k{color:var(--muted)}.kv .v{font-weight:600;color:var(--text)}
  .chart-box{background:var(--card);border:1px solid var(--border);border-radius:8px;
             padding:1rem;margin-bottom:1rem}
  .hidden{display:none}
  #errorMsg{color:var(--red);background:#3a1a1a;border:1px solid var(--red);
            border-radius:6px;padding:.6rem 1rem;font-size:.9rem;margin-bottom:.75rem;display:none}
  #errorMsg .tip{color:var(--yellow);margin-top:.4rem;font-size:.82rem}
</style>
</head>
<body>
<header>
  <h1>📈 Financial Market Risk Pipeline</h1>
  <span>Powered by advanced-distributions · 7 statistical models</span>
</header>
<div class="container">
  <div class="search-bar">
    <input id="tickerInput" placeholder="Enter ticker e.g. RELIANCE.NS, TCS.NS, AAPL, MSFT" value="RELIANCE.NS"/>
    <button onclick="analyze()">Analyze</button>
  </div>
  <div class="examples">
    Try: 
    <span onclick="setTicker('RELIANCE.NS')">RELIANCE.NS</span>
    <span onclick="setTicker('TCS.NS')">TCS.NS</span>
    <span onclick="setTicker('INFY.NS')">INFY.NS</span>
    <span onclick="setTicker('AAPL')">AAPL</span>
    <span onclick="setTicker('MSFT')">MSFT</span>
    <span onclick="setTicker('TSLA')">TSLA</span>
  </div>
  <div id="errorMsg"><span id="errText"></span><div class="tip" id="errTip"></div></div>
  <div id="suggestion"></div>
  <div id="status"></div>

  <div id="results" class="hidden">
    <div class="grid-3">
      <div class="card"><h3>Signal</h3><div class="metric" id="dec_signal">—</div></div>
      <div class="card"><h3>Confidence</h3><div class="metric" id="dec_conf">—</div></div>
      <div class="card"><h3>Net Score</h3><div class="metric" id="dec_net">—</div></div>
    </div>
    <div class="grid-3">
      <div class="card"><h3>Bull Score</h3><div class="metric" id="dec_bull" style="color:var(--green)">—</div></div>
      <div class="card"><h3>Bear Score</h3><div class="metric" id="dec_bear" style="color:var(--red)">—</div></div>
      <div class="card"><h3>Sessions</h3><div class="metric" id="n_sessions">—</div></div>
    </div>

    <div class="chart-box"><div id="priceChart" style="height:260px"></div></div>
    <div class="grid-2">
      <div class="chart-box"><div id="returnsDist" style="height:220px"></div></div>
      <div class="chart-box"><div id="riskGauge"  style="height:220px"></div></div>
    </div>

    <div class="section-title">Model Results</div>
    <div class="grid-2">
      <div class="card"><h3>Fractional Distribution — Market Regime</h3><div class="kv-grid" id="kv_frac"></div></div>
      <div class="card"><h3>Fractal Distribution — Complexity</h3><div class="kv-grid" id="kv_fractal"></div></div>
    </div>
    <div class="grid-2">
      <div class="card"><h3>Sinh-Arcsinh Distribution — Skewness</h3><div class="kv-grid" id="kv_sinh"></div></div>
      <div class="card"><h3>Slash Distribution — Crash Risk</h3><div class="kv-grid" id="kv_slash"></div></div>
    </div>
    <div class="grid-2">
      <div class="card"><h3>Neural Spline — Quantile Forecast</h3><div class="kv-grid" id="kv_spline"></div></div>
      <div class="card"><h3>Quantile Distribution — VaR / CVaR</h3><div class="kv-grid" id="kv_quant"></div></div>
    </div>
    <div class="card" style="margin-bottom:1rem">
      <h3>Davies Distribution — Stress Regime</h3>
      <div class="kv-grid" id="kv_davies"></div>
    </div>
  </div>
</div>

<script>
const $ = id => document.getElementById(id);
const DARK={paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',
            font:{color:'#e6edf3',size:11},
            xaxis:{gridcolor:'#30363d',zerolinecolor:'#30363d'},
            yaxis:{gridcolor:'#30363d',zerolinecolor:'#30363d'},
            margin:{t:30,r:10,b:40,l:50}};

function setTicker(t){$('tickerInput').value=t;analyze();}

function kvHtml(obj){
  const badgeKeys=['regime','structure','crash_risk','skewness_direction'];
  return Object.entries(obj).map(([k,v])=>{
    let disp=v;
    if(badgeKeys.includes(k)){
      const cls=(v||'').toLowerCase().replace(/\s+/g,'-');
      disp=`<span class="badge ${cls}">${v}</span>`;
    }
    return `<div class="kv"><span class="k">${k}</span><span class="v">${disp}</span></div>`;
  }).join('');
}

function showError(msg, tip=''){
  $('errorMsg').style.display='block';
  $('errText').textContent=msg;
  $('errTip').textContent=tip;
  $('errTip').style.display=tip?'block':'none';
  $('results').classList.add('hidden');
  $('suggestion').style.display='none';
}

function hideError(){$('errorMsg').style.display='none';}

async function analyze(){
  const ticker=$('tickerInput').value.trim().toUpperCase();
  if(!ticker)return;
  hideError();
  $('suggestion').style.display='none';
  $('results').classList.add('hidden');
  $('status').textContent=`⏳ Running pipeline for ${ticker} — this may take ~30s…`;

  try{
    const res=await fetch(`/api/analyze?ticker=${encodeURIComponent(ticker)}`);
    const data=await res.json();

    if(!res.ok){
      $('status').textContent='';
      let tip='';
      const err=data.error||'Pipeline failed';
      if(err.includes('No data')){
        if(!ticker.endsWith('.NS') && ticker.length<=6)
          tip='💡 For Indian stocks, add .NS — e.g. '+ticker+'.NS';
        else
          tip='💡 Check the ticker symbol at finance.yahoo.com';
      }
      showError('❌ '+err, tip);
      return;
    }

    if(data.suggestion){
      $('suggestion').textContent='ℹ️ '+data.suggestion;
      $('suggestion').style.display='block';
      $('tickerInput').value=data.ticker;
    }

    $('status').textContent=`✅ Done — ${data.n_sessions} sessions analysed for ${data.ticker}`;

    const dec=data.decision;
    const sig=$('dec_signal');
    sig.textContent=dec.decision;
    sig.className=`metric badge ${dec.decision}`;
    $('dec_conf').textContent=(dec.confidence*100).toFixed(1)+'%';
    $('dec_net').textContent=dec.net_score;
    $('dec_bull').textContent=dec.bull_score;
    $('dec_bear').textContent=dec.bear_score;
    $('n_sessions').textContent=data.n_sessions;

    Plotly.newPlot('priceChart',[{
      x:[...Array(data._prices.length).keys()].map(i=>i-data._prices.length+1),
      y:data._prices,type:'scatter',mode:'lines',
      line:{color:'#58a6ff',width:1.5},name:'Close'
    }],{...DARK,title:{text:`${data.ticker} — Last 120 Sessions`,font:{color:'#e6edf3',size:13}}});

    Plotly.newPlot('returnsDist',[{
      x:data._returns,type:'histogram',nbinsx:60,
      marker:{color:'#58a6ff',opacity:.75},name:'Log Returns'
    }],{...DARK,title:{text:'Return Distribution (last 300)',font:{color:'#e6edf3',size:12}}});

    const stress=data.davies.stress_score;
    Plotly.newPlot('riskGauge',[{
      type:'indicator',mode:'gauge+number',value:Math.round(stress*100),
      gauge:{axis:{range:[0,100]},
             bar:{color:stress>0.65?'#f85149':stress>0.4?'#d29922':'#3fb950'},
             steps:[{range:[0,40],color:'#1a3a1f'},{range:[40,65],color:'#3a2e1a'},
                    {range:[65,100],color:'#3a1a1a'}]},
      title:{text:`Stress Score — ${data.davies.regime}`,font:{color:'#e6edf3',size:12}},
      number:{suffix:'%',font:{color:'#e6edf3'}}
    }],{...DARK,margin:{t:40,r:20,b:20,l:20}});

    $('kv_frac').innerHTML    = kvHtml(data.fractional);
    $('kv_fractal').innerHTML = kvHtml(data.fractal);
    $('kv_sinh').innerHTML    = kvHtml(data.sinh_arcsinh);
    $('kv_slash').innerHTML   = kvHtml(data.slash);
    $('kv_spline').innerHTML  = kvHtml(data.neural_spline);
    $('kv_quant').innerHTML   = kvHtml(data.quantile);
    $('kv_davies').innerHTML  = kvHtml(data.davies);

    $('results').classList.remove('hidden');
  }catch(e){
    $('status').textContent='';
    showError('❌ Network error — '+e.message);
  }
}

document.getElementById('tickerInput').addEventListener('keydown',e=>{
  if(e.key==='Enter') analyze();
});
</script>
</body>
</html>"""


# ────────────────────────────────────────────────────────────────────────────
# ROUTES
# ────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)


@app.route("/api/analyze")
def api_analyze():
    ticker = request.args.get("ticker", "").strip().upper()
    if not ticker:
        return jsonify({"error": "ticker parameter is required"}), 400
    try:
        report = run_pipeline(ticker)
        return jsonify(report)
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": f"Pipeline error: {str(e)}"}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
