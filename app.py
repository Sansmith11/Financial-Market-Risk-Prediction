"""
Financial Market Risk Pipeline — Flask Web App
Render deployment (Python 3.11) — v13 Groww UI
"""
import os, math, json, warnings
import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, request, Response, render_template_string

warnings.filterwarnings("ignore")

try:
    from curl_cffi.requests.exceptions import HTTPError as CurlHTTPError
except ImportError:
    CurlHTTPError = OSError

from advanced_distributions.fractional_distribution import FractionalDistribution
from advanced_distributions.fractal_distribution   import FractalDistribution
from advanced_distributions.sinh_arcsinh           import SinhArcsinhDistribution
from advanced_distributions.slash_distribution     import SlashDistribution
from advanced_distributions.quantile_distribution  import QuantileDistribution
from advanced_distributions.davies_distribution    import DaviesDistribution

app = Flask(__name__)

def sf(x, n=6):
    try:
        v = float(x)
        return 0.0 if (math.isnan(v) or math.isinf(v)) else round(v, n)
    except Exception: return 0.0

def _clean(obj):
    if isinstance(obj, float): return 0.0 if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):  return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):  return [_clean(v) for v in obj]
    return obj

def safe_json(obj, status=200):
    return Response(json.dumps(_clean(obj), allow_nan=False), status=status, mimetype="application/json")

CORRECTIONS = {
    "APPL":"AAPL","AMZON":"AMZN","AMAZN":"AMZN","MICROSFT":"MSFT","MICROSFOT":"MSFT",
    "NETFLX":"NFLX","TESTA":"TSLA","RELINCE":"RELIANCE.NS","RELIACE":"RELIANCE.NS",
    "HDFCBANK":"HDFCBANK.NS","ICICIBANK":"ICICIBANK.NS","BAJFINANCE":"BAJFINANCE.NS",
    "SBIN":"SBIN.NS","WIPRO":"WIPRO.NS","INFY":"INFY.NS",
}
def resolve(ticker):
    u = ticker.upper().strip()
    if u in CORRECTIONS: c=CORRECTIONS[u]; return c, f"'{u}' auto-corrected to '{c}'"
    return u, None

def fetch(ticker, period="3y"):
    df = pd.DataFrame()
    try: df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
    except Exception: pass
    if df is None or df.empty:
        try: df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        except Exception: pass
    if df is None or df.empty:
        t = ticker.upper()
        hint = f" Try '{t}.NS' for Indian stocks." if (not t.endswith(".NS") and t.isalpha() and len(t)<=6) else ""
        raise ValueError(f"No data found for '{ticker}'.{hint}")
    df.columns = ([str(c[0]).lower() for c in df.columns] if isinstance(df.columns, pd.MultiIndex)
                  else [str(c).lower() for c in df.columns])
    for col in ["open","high","low","close","volume"]:
        if col not in df.columns: raise ValueError(f"Missing column '{col}'")
    return df

def clean(df):
    df = df.copy().dropna(); df = df[~df.index.duplicated(keep="first")]
    m = (df.high>=df.low)&(df.high>=df.close)&(df.low<=df.close)&(df.volume>0); df = df[m]
    lr = np.log(df.close/df.close.shift(1)).dropna(); z = (lr-lr.mean())/lr.std()
    return df.loc[z[np.abs(z)<=5].index].sort_index()

def features(df, w=20):
    df = df.copy(); df["lr"] = np.log(df.close/df.close.shift(1))
    df["vol"] = df.lr.rolling(w).std()*np.sqrt(252)
    rm,rs = df.close.rolling(w).mean(), df.close.rolling(w).std()
    df["mom"] = (df.close-rm)/(rs+1e-10)
    d = df.close.diff(); g = d.clip(lower=0).rolling(14).mean(); l = (-d.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100-100/(1+g/(l+1e-10)); df.dropna(inplace=True); return df

def run_fractional(r):
    pos=np.abs(r)+1e-8; d=FractionalDistribution.fit(pos)
    h=float(np.clip(0.5+(d.alpha-1.5)*0.1,0.3,0.8)); mem=float(np.clip((h-0.5)*2,0,1))
    reg="trending" if h>0.55 else "mean-reverting" if h<0.45 else "random walk"
    bs=float(np.clip((h-0.5)*4,0,1)); brs=float(np.clip((0.5-h)*4,0,1))
    return {"alpha":sf(d.alpha,4),"beta":sf(d.beta,4),"gamma":sf(d.gamma,4),
            "mean":sf(d.mean()),"std":sf(d.std()),"skewness":sf(d.skewness(),4),
            "kurtosis":sf(d.kurtosis(),4),"hurst_proxy":sf(h,4),"memory_score":sf(mem,4),
            "regime":reg,"_bull":bs,"_bear":brs}

def run_fractal(r):
    pos=np.abs(r)+1e-8; p=FractalDistribution.fit(pos); D,lam=sf(p[0],4),sf(p[1],4)
    d=FractalDistribution(D=D,lambda_=lam); comp=float(np.clip(D-1.0,0,1))
    s="chaotic" if D>1.7 else "complex" if D>1.4 else "structured"
    return {"D":D,"lambda":lam,"mean":sf(d.mean()),"std":sf(d.std()),
            "complexity_score":sf(comp,4),"structure":s,"_bull":1.0-comp,"_bear":comp}

def run_sinh(r):
    d=SinhArcsinhDistribution.fit(r)
    sk="left-skewed" if d.epsilon<-0.1 else "right-skewed" if d.epsilon>0.1 else "symmetric"
    ss=float(np.tanh(d.epsilon*2)); tp=float(np.clip((d.delta-1.0)*0.3,0,0.3))
    return {"epsilon":sf(d.epsilon,4),"delta":sf(d.delta,4),"mu":sf(d.mu),"sigma":sf(d.sigma),
            "skewness":sf(d.skewness(),4),"kurtosis":sf(d.kurtosis(),4),
            "skewness_direction":sk,"skew_score":sf(np.tanh(d.epsilon),4),
            "_bull":max(0,float(np.clip(ss,0,1))-tp),"_bear":max(0,float(np.clip(-ss,0,1))+tp*0.5)}

def run_slash(r):
    d=SlashDistribution.fit(r); s=d.rvs(size=5000,random_state=42)
    ep=sf(np.mean(np.abs(s-d.mu)>3.0*d.sigma),4); cr="high" if ep>0.05 else "medium" if ep>0.02 else "low"
    ent=sf(d.entropy(),4); ep2=float(np.clip(ep*15,0,1)); ent_p=float(np.clip((ent-2.0)*0.1,0,0.3)) if ent>2.0 else 0.0
    return {"mu":sf(d.mu),"sigma":sf(d.sigma),"extreme_event_prob":ep,"crash_risk":cr,
            "median":sf(d.median()),"entropy":ent,
            "_bull":max(0,1.0-ep2-ent_p),"_bear":min(1,ep2+ent_p)}

def run_spline(r):
    from scipy.stats import johnsonsu
    try:
        a,b,loc,scale=johnsonsu.fit(r); ql=[0.05,0.10,0.25,0.50,0.75,0.90,0.95]
        qv=johnsonsu.ppf(ql,a,b,loc,scale); q={f"q{int(p*100):02d}":sf(v) for p,v in zip(ql,qv)}
        mn=sf(johnsonsu.mean(a,b,loc,scale)); std=sf(johnsonsu.std(a,b,loc,scale))
        skw=sf(johnsonsu.stats(a,b,loc,scale,moments='s'),4); krt=sf(johnsonsu.stats(a,b,loc,scale,moments='k'),4)
        ent=sf(johnsonsu.entropy(a,b,loc,scale),4); unc=sf(np.clip(abs(q["q95"]-q["q05"])*50,0,1),4)
        up=float(q["q75"])-float(q["q50"]); dn=float(q["q50"])-float(q["q25"])
        ratio=up/(abs(dn)+1e-8); bs=float(np.clip(ratio*0.4,0,1)); brs=float(np.clip((1/max(ratio,0.01))*0.4,0,1))
    except Exception:
        q={f"q{int(p*100):02d}":sf(np.percentile(r,p*100)) for p in [0.05,0.10,0.25,0.50,0.75,0.90,0.95]}
        mn=sf(np.mean(r)); std=sf(np.std(r)); skw=krt=ent=unc=0.0; bs=brs=0.3
    return {**q,"mean":mn,"std":std,"skewness":skw,"kurtosis":krt,"entropy":ent,"uncertainty":unc,"_bull":bs,"_bear":brs}

def run_quantile(r):
    d=QuantileDistribution.fit(r); v95=sf(d.ppf(0.05)); v99=sf(d.ppf(0.01))
    s=d.rvs(size=10000,random_state=42); cv=sf(np.mean(s[s<=v95])); tr=sf(np.clip(-cv*10,0,1),4)
    vbr=sf(np.mean(r<v95),4); bs=float(np.clip(1.0-tr-vbr*2,0,1)); brs=float(np.clip(tr+vbr*2,0,1))
    return {"mu":sf(d.mu),"sigma":sf(d.sigma),"alpha_shape":sf(d.alpha,4),"beta_shape":sf(d.beta,4),
            "VaR_95":v95,"VaR_99":v99,"CVaR_95":cv,"tail_risk_score":tr,"var_breach_rate":vbr,
            "skewness":sf(d.skewness(),4),"kurtosis":sf(d.kurtosis(),4),"_bull":bs,"_bear":brs}

def run_davies(r,w=20):
    pos=np.abs(r)+1e-8; p=DaviesDistribution.fit(pos); a,b,th,k=[sf(x,4) for x in p]
    d=DaviesDistribution(alpha=a,beta=b,theta=th,k=k)
    rv=float(np.mean(np.abs(r[-w:]))); ov=float(np.mean(pos)); vr=sf(rv/(ov+1e-10),4)
    try: sc=float(np.clip(d.cdf(np.array([rv]))[0],0,1))
    except: sc=0.5
    reg="crisis" if sc>0.85 else "stress" if sc>0.65 else "caution" if sc>0.40 else "normal"
    stab=float(np.clip(1.0-abs(vr-1.0),0,1))
    return {"alpha":a,"beta":b,"theta":th,"k":k,"mean":sf(d.mean()),"std":sf(d.std()),
            "stress_score":sf(sc,4),"vol_ratio":vr,"regime":reg,
            "_bull":float(np.clip((1.0-sc)*0.8+stab*0.2,0,1)),
            "_bear":float(np.clip(sc*0.8+(1-stab)*0.2,0,1))}

def make_decision(frac,fractal,sinh,slash,spline,quant,davies):
    W={"davies":0.25,"slash":0.20,"spline":0.20,"quant":0.18,"frac":0.10,"fractal":0.07}
    models={"davies":davies,"slash":slash,"spline":spline,"quant":quant,"frac":frac,"fractal":fractal}
    skew_boost=sinh["_bull"]-sinh["_bear"]
    bs=float(np.clip(sum(models[k]["_bull"]*W[k] for k in W)+skew_boost*0.10,0,1))
    brs=float(np.clip(sum(models[k]["_bear"]*W[k] for k in W)-skew_boost*0.10,0,1))
    net=bs-brs; raw=abs(net)/(bs+brs+1e-6)
    conf=1.0/(1.0+math.exp(-8.0*(raw-0.25)))
    dec="BUY" if net>0.08 else "SELL" if net<-0.08 else "HOLD"
    breakdown={k:{"bull":sf(models[k]["_bull"],3),"bear":sf(models[k]["_bear"],3),"weight":W[k]} for k in W}
    breakdown["sinh"]={"bull":sf(sinh["_bull"],3),"bear":sf(sinh["_bear"],3),"weight":0.0}
    return {"decision":dec,"bull_score":sf(bs,4),"bear_score":sf(brs,4),
            "net_score":sf(net,4),"confidence":sf(conf,4),"breakdown":breakdown}

def pipeline(ticker):
    ticker,suggestion=resolve(ticker)
    df=features(clean(fetch(ticker))); r=df["lr"].values; px=df["close"].values
    frac=run_fractional(r); fractal=run_fractal(r); sinh=run_sinh(r)
    slash=run_slash(r); spline=run_spline(r); quant=run_quantile(r); davies=run_davies(r)
    dec=make_decision(frac,fractal,sinh,slash,spline,quant,davies)
    def pub(d): return {k:v for k,v in d.items() if not k.startswith("_")}
    return _clean({"ticker":ticker,"suggestion":suggestion,"n_sessions":len(df),
        "mean_return":sf(r.mean()),"std_return":sf(r.std()),
        "fractional":pub(frac),"fractal":pub(fractal),"sinh_arcsinh":pub(sinh),
        "slash":pub(slash),"neural_spline":pub(spline),"quantile":pub(quant),
        "davies":pub(davies),"decision":dec,
        "_prices":px[-120:].tolist(),"_returns":r[-300:].tolist()})

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/><meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>RiskPulse — Financial Analysis</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0;-webkit-tap-highlight-color:transparent}
:root{
  --g0:#00b386;--g1:#00d4a0;--gbg:#e8faf5;
  --r0:#eb5757;--r1:#ff7070;--rbg:#fdf0f0;
  --y0:#f39c12;--ybg:#fff8ec;
  --bg:#f8f9fa;--white:#ffffff;--dark:#0e1117;
  --t1:#0e1117;--t2:#4a4a4a;--t3:#8a8a8a;--t4:#c0c0c0;
  --bdr:#e8e8e8;--bdr2:#d0d0d0;
  --card-r:16px;--pill-r:100px
}
body{background:var(--bg);color:var(--t1);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
     min-height:100vh;font-size:14px}

/* TOP NAV */
.nav{background:var(--white);border-bottom:1px solid var(--bdr);
     padding:0 20px;height:56px;display:flex;align-items:center;justify-content:space-between;
     position:sticky;top:0;z-index:100}
.nav-brand{display:flex;align-items:center;gap:8px}
.nav-logo{width:32px;height:32px;background:#00b386;border-radius:8px;
          display:flex;align-items:center;justify-content:center;color:#fff;font-weight:700;font-size:14px}
.nav-title{font-size:15px;font-weight:700;color:var(--t1)}
.nav-sub{font-size:11px;color:var(--t3);margin-top:1px}
.search-row{display:flex;align-items:center;gap:8px}
.search-box{display:flex;align-items:center;background:#f0f1f5;border-radius:10px;
            padding:8px 14px;gap:8px;min-width:240px;border:1.5px solid transparent;transition:.2s}
.search-box:focus-within{background:var(--white);border-color:var(--g0)}
.search-box svg{flex-shrink:0;opacity:.4}
.search-box input{background:none;border:none;outline:none;color:var(--t1);font-size:14px;width:100%}
.search-box input::placeholder{color:var(--t3)}
.btn-analyze{background:var(--g0);color:#fff;font-weight:600;border:none;
             border-radius:10px;padding:9px 20px;cursor:pointer;font-size:14px;
             white-space:nowrap;transition:background .2s}
.btn-analyze:hover{background:#009e78}
.btn-analyze:active{transform:scale(.97)}

/* BODY */
.body{max-width:1000px;margin:0 auto;padding:20px 16px}

/* QUICK CHIPS */
.chips{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:20px;align-items:center}
.chips-label{font-size:12px;color:var(--t3);margin-right:4px}
.chip{font-size:12px;font-weight:600;padding:5px 12px;border-radius:var(--pill-r);
      color:var(--g0);cursor:pointer;background:var(--gbg);border:1.5px solid #b8efe0;
      transition:.15s;white-space:nowrap}
.chip:hover,.chip.on{background:var(--g0);color:#fff;border-color:var(--g0)}

/* ALERTS */
.alert{border-radius:12px;padding:12px 16px;font-size:13px;margin-bottom:16px;
       display:none;line-height:1.5;border:1px solid}
.alert-err{background:var(--rbg);border-color:#f5c6c6;color:#c0392b}
.alert-warn{background:var(--ybg);border-color:#f9d79c;color:#b7770d}
.alert-tip{font-size:12px;margin-top:4px;opacity:.8}
#st{font-size:12px;color:var(--t3);margin-bottom:16px;min-height:16px}

/* SIGNAL HERO CARD */
.hero{background:var(--white);border-radius:var(--card-r);border:1px solid var(--bdr);
      padding:20px;margin-bottom:16px}
.hero-top{display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:16px}
.hero-ticker{font-size:22px;font-weight:700}
.hero-meta{font-size:12px;color:var(--t3);margin-top:2px}
.signal-badge{padding:6px 16px;border-radius:var(--pill-r);font-size:15px;font-weight:700;letter-spacing:.02em}
.sig-BUY{background:var(--gbg);color:var(--g0);border:1.5px solid #9ee8d4}
.sig-SELL{background:var(--rbg);color:var(--r0);border:1.5px solid #f5b8b8}
.sig-HOLD{background:var(--ybg);color:var(--y0);border:1.5px solid #f9d79c}
.hero-stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}
@media(max-width:600px){.hero-stats{grid-template-columns:repeat(2,1fr)}}
.hstat{background:#f8f9fa;border-radius:10px;padding:12px}
.hstat-l{font-size:11px;color:var(--t3);margin-bottom:6px;text-transform:uppercase;letter-spacing:.04em}
.hstat-v{font-size:18px;font-weight:700}
.conf-bar{height:4px;background:#e8e8e8;border-radius:4px;margin-top:8px}
.conf-fill{height:100%;border-radius:4px;transition:width .8s ease}

/* CHART CARD */
.chart-card{background:var(--white);border-radius:var(--card-r);border:1px solid var(--bdr);
            padding:16px;margin-bottom:16px}
.chart-card-title{font-size:12px;font-weight:600;color:var(--t2);text-transform:uppercase;
                  letter-spacing:.06em;margin-bottom:12px}
.chart-row{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:16px}
@media(max-width:600px){.chart-row{grid-template-columns:1fr}}

/* SCORE BREAKDOWN */
.score-card{background:var(--white);border-radius:var(--card-r);border:1px solid var(--bdr);
            padding:16px;margin-bottom:16px}
.score-title{font-size:12px;font-weight:700;color:var(--t2);text-transform:uppercase;
             letter-spacing:.06em;margin-bottom:14px}
.score-row{display:flex;align-items:center;gap:10px;margin-bottom:10px}
.score-row:last-child{margin-bottom:0}
.score-name{font-size:12px;color:var(--t2);font-weight:600;width:80px;flex-shrink:0}
.score-bars{flex:1;display:flex;flex-direction:column;gap:3px}
.sbar{height:6px;background:#f0f0f0;border-radius:4px}
.sbar-fill{height:100%;border-radius:4px;transition:width .6s ease}
.score-vals{display:flex;flex-direction:column;align-items:flex-end;gap:2px;width:68px;flex-shrink:0}
.sval{font-size:11px;font-weight:600}
.score-wt{font-size:11px;color:var(--t3);width:32px;text-align:right;flex-shrink:0}

/* MODEL ACCORDION */
.models-section{margin-bottom:16px}
.models-title{font-size:12px;font-weight:700;color:var(--t2);text-transform:uppercase;
              letter-spacing:.06em;margin-bottom:10px;display:flex;align-items:center;gap:8px}
.models-title::after{content:'';flex:1;height:1px;background:var(--bdr)}
.mod{background:var(--white);border:1px solid var(--bdr);border-radius:var(--card-r);
     margin-bottom:8px;overflow:hidden}
.mod-hdr{display:flex;align-items:center;justify-content:space-between;
          padding:14px 16px;cursor:pointer;transition:background .15s}
.mod-hdr:hover{background:#f8f9fa}
.mod-hdr:active{background:#f0f1f5}
.mod-left{display:flex;align-items:center;gap:12px}
.mod-icon{width:36px;height:36px;border-radius:10px;display:flex;align-items:center;
          justify-content:center;font-size:16px;flex-shrink:0}
.mod-name{font-size:13px;font-weight:600;color:var(--t1)}
.mod-tag{font-size:11px;color:var(--t3);margin-top:1px}
.mod-right{display:flex;align-items:center;gap:8px}
.pill-b{background:var(--gbg);color:var(--g0);border:1px solid #b8efe0;
        font-size:11px;font-weight:700;padding:3px 9px;border-radius:var(--pill-r)}
.pill-s{background:var(--rbg);color:var(--r0);border:1px solid #f5b8b8;
        font-size:11px;font-weight:700;padding:3px 9px;border-radius:var(--pill-r)}
.chev{font-size:12px;color:var(--t4);transition:transform .25s;margin-left:4px}
.mod.open .chev{transform:rotate(180deg)}
.mod-body{display:none;border-top:1px solid var(--bdr);padding:14px 16px;background:#fafafa}
.mod.open .mod-body{display:block}
.kv-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:6px}
.kv{display:flex;justify-content:space-between;align-items:center;
    background:var(--white);border:1px solid var(--bdr);border-radius:8px;
    padding:8px 10px;font-size:12px}
.kv .k{color:var(--t3)}.kv .v{font-weight:600;color:var(--t1)}
.tag{display:inline-block;padding:2px 8px;border-radius:var(--pill-r);font-size:11px;font-weight:600}
.tag-g{background:var(--gbg);color:var(--g0);border:1px solid #b8efe0}
.tag-r{background:var(--rbg);color:var(--r0);border:1px solid #f5c6c6}
.tag-y{background:var(--ybg);color:var(--y0);border:1px solid #f9d79c}

/* DAVIES STRIP */
.davies-strip{background:var(--white);border:1px solid var(--bdr);border-radius:var(--card-r);
              padding:14px 16px;margin-bottom:16px}
.davies-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}
@media(max-width:500px){.davies-grid{grid-template-columns:repeat(2,1fr)}}
.dv{background:#f8f9fa;border-radius:8px;padding:10px}
.dv-l{font-size:11px;color:var(--t3);margin-bottom:4px}
.dv-v{font-size:14px;font-weight:700;color:var(--t1)}

.hidden{display:none}
</style>
</head>
<body>

<div class="nav">
  <div class="nav-brand">
    <div class="nav-logo">RP</div>
    <div>
      <div class="nav-title">RiskPulse</div>
      <div class="nav-sub">7 distribution models</div>
    </div>
  </div>
  <div class="search-row">
    <div class="search-box">
      <svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
        <circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/>
      </svg>
      <input id="ti" placeholder="Search ticker — RELIANCE.NS, AAPL…" value="RELIANCE.NS"/>
    </div>
    <button class="btn-analyze" onclick="analyze()">Analyze</button>
  </div>
</div>

<div class="body">
  <div class="chips">
    <span class="chips-label">Quick:</span>
    <span class="chip" onclick="go('RELIANCE.NS',this)">RELIANCE.NS</span>
    <span class="chip" onclick="go('TCS.NS',this)">TCS.NS</span>
    <span class="chip" onclick="go('INFY.NS',this)">INFY.NS</span>
    <span class="chip" onclick="go('HDFCBANK.NS',this)">HDFCBANK.NS</span>
    <span class="chip" onclick="go('AAPL',this)">AAPL</span>
    <span class="chip" onclick="go('MSFT',this)">MSFT</span>
    <span class="chip" onclick="go('TSLA',this)">TSLA</span>
    <span class="chip" onclick="go('AMZN',this)">AMZN</span>
  </div>

  <div id="eb" class="alert alert-err"><span id="et"></span><div id="ep" class="alert-tip"></div></div>
  <div id="wb" class="alert alert-warn"></div>
  <div id="st"></div>

  <div id="res" class="hidden">
    <!-- Hero signal card -->
    <div class="hero">
      <div class="hero-top">
        <div>
          <div class="hero-ticker" id="h-ticker">—</div>
          <div class="hero-meta" id="h-meta">—</div>
        </div>
        <div class="signal-badge" id="h-signal">—</div>
      </div>
      <div class="hero-stats">
        <div class="hstat">
          <div class="hstat-l">Confidence</div>
          <div class="hstat-v" id="h-conf">—</div>
          <div class="conf-bar"><div class="conf-fill" id="h-cbar"></div></div>
        </div>
        <div class="hstat">
          <div class="hstat-l">Bull Score</div>
          <div class="hstat-v" style="color:var(--g0)" id="h-bull">—</div>
        </div>
        <div class="hstat">
          <div class="hstat-l">Bear Score</div>
          <div class="hstat-v" style="color:var(--r0)" id="h-bear">—</div>
        </div>
        <div class="hstat">
          <div class="hstat-l">Sessions</div>
          <div class="hstat-v" id="h-sess">—</div>
        </div>
      </div>
    </div>

    <!-- Price chart -->
    <div class="chart-card">
      <div class="chart-card-title" id="price-title">Price chart</div>
      <div id="cp" style="height:200px"></div>
    </div>

    <!-- Return dist + Gauge -->
    <div class="chart-row">
      <div class="chart-card" style="margin-bottom:0">
        <div class="chart-card-title">Return distribution</div>
        <div id="ch" style="height:155px"></div>
      </div>
      <div class="chart-card" style="margin-bottom:0">
        <div class="chart-card-title">Stress gauge · Davies</div>
        <div id="cg" style="height:155px"></div>
      </div>
    </div>

    <!-- Score breakdown -->
    <div class="score-card">
      <div class="score-title">Signal breakdown by model</div>
      <div id="sb"></div>
    </div>

    <!-- Model accordion -->
    <div class="models-section">
      <div class="models-title">Model details — tap to expand</div>
      <div id="mods"></div>
    </div>

    <!-- Davies strip -->
    <div class="davies-strip">
      <div class="chart-card-title" style="margin-bottom:10px">Davies distribution — stress regime</div>
      <div id="md"></div>
    </div>
  </div>
</div>

<script>
const $=id=>document.getElementById(id);
const PL={paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',
          font:{color:'#8a8a8a',size:10},
          xaxis:{gridcolor:'#f0f0f0',zerolinecolor:'#f0f0f0'},
          yaxis:{gridcolor:'#f0f0f0',zerolinecolor:'#f0f0f0'},
          margin:{t:8,r:8,b:28,l:48}};
const PC={responsive:true,displayModeBar:false};

function go(t,el){
  document.querySelectorAll('.chip').forEach(x=>x.classList.remove('on'));
  if(el)el.classList.add('on');
  $('ti').value=t; analyze();
}

function tagClass(v){
  const s=(v||'').toString().toLowerCase();
  if(['trending','structured','low','normal','right-skewed'].includes(s))return 'tag-g';
  if(['mean-reverting','chaotic','high','crisis','left-skewed'].includes(s))return 'tag-r';
  return 'tag-y';
}

function kvHtml(obj,tagKeys=[]){
  return '<div class="kv-grid">'+Object.entries(obj).map(([k,v])=>{
    let d=v;
    if(tagKeys.includes(k)){const c=tagClass(v);d=`<span class="tag ${c}">${v}</span>`;}
    return `<div class="kv"><span class="k">${k}</span><span class="v">${d}</span></div>`;
  }).join('')+'</div>';
}

const MODS=[
  {key:'frac',   icon:'〜',iconBg:'#e8f4ff',name:'Fractional Distribution',tag:'Long-memory · Hurst proxy',    data:'fractional',   bk:['regime']},
  {key:'fractal',icon:'❄', iconBg:'#f0e8ff',name:'Fractal Distribution',   tag:'Complexity · Market structure',data:'fractal',      bk:['structure']},
  {key:'sinh',   icon:'⟛', iconBg:'#fff0e8',name:'Sinh-Arcsinh Dist.',      tag:'Skewness · Tail shape',        data:'sinh_arcsinh', bk:['skewness_direction']},
  {key:'slash',  icon:'⚡', iconBg:'#fff8e8',name:'Slash Distribution',      tag:'Extreme events · Crash risk',  data:'slash',        bk:['crash_risk']},
  {key:'spline', icon:'📈', iconBg:'#e8fff4',name:'JohnsonSU / Neural Spline',tag:'Quantile forecast',          data:'neural_spline',bk:[]},
  {key:'quant',  icon:'📉', iconBg:'#ffe8e8',name:'Quantile Distribution',   tag:'VaR · CVaR · Tail risk',       data:'quantile',     bk:[]},
];

function buildBreakdown(dec){
  const bd=dec.breakdown;
  const order=['davies','slash','spline','quant','frac','fractal'];
  const names={davies:'Davies',slash:'Slash',spline:'JohnsonSU',quant:'Quantile',frac:'Fractional',fractal:'Fractal'};
  $('sb').innerHTML=order.map(k=>{
    const b=bd[k]||{}; const w=b.weight||0;
    const bv=+(b.bull||0).toFixed(2); const brv=+(b.bear||0).toFixed(2);
    return `<div class="score-row">
      <div class="score-name">${names[k]}</div>
      <div class="score-bars">
        <div class="sbar"><div class="sbar-fill" style="width:${Math.round(bv*100)}%;background:var(--g0)"></div></div>
        <div class="sbar"><div class="sbar-fill" style="width:${Math.round(brv*100)}%;background:var(--r0)"></div></div>
      </div>
      <div class="score-vals">
        <span class="sval" style="color:var(--g0)">▲ ${bv.toFixed(2)}</span>
        <span class="sval" style="color:var(--r0)">▼ ${brv.toFixed(2)}</span>
      </div>
      <div class="score-wt">${Math.round(w*100)}%</div>
    </div>`;
  }).join('');
}

function buildMods(d,dec){
  const bd=dec.breakdown;
  $('mods').innerHTML=MODS.map(m=>{
    const b=bd[m.key]||{}; const bv=+(b.bull||0).toFixed(2); const brv=+(b.bear||0).toFixed(2);
    return `<div class="mod" id="m_${m.key}">
      <div class="mod-hdr" onclick="tog('${m.key}')">
        <div class="mod-left">
          <div class="mod-icon" style="background:${m.iconBg}">${m.icon}</div>
          <div><div class="mod-name">${m.name}</div><div class="mod-tag">${m.tag}</div></div>
        </div>
        <div class="mod-right">
          <span class="pill-b">▲ ${bv}</span>
          <span class="pill-s">▼ ${brv}</span>
          <span class="chev">▼</span>
        </div>
      </div>
      <div class="mod-body">${kvHtml(d[m.data],m.bk)}</div>
    </div>`;
  }).join('');
}

function tog(k){$('m_'+k).classList.toggle('open');}

function showErr(msg,tip=''){
  $('eb').style.display='block';$('et').textContent=msg;
  $('ep').textContent=tip;$('ep').style.display=tip?'block':'none';
}

async function analyze(){
  const t=$('ti').value.trim().toUpperCase();
  if(!t)return;
  $('eb').style.display='none';$('wb').style.display='none';
  $('res').classList.add('hidden');
  $('st').textContent='Analyzing '+t+'…';
  try{
    const res=await fetch('/api/analyze?ticker='+encodeURIComponent(t));
    const ct=res.headers.get('content-type')||'';
    if(!ct.includes('application/json')){
      $('st').textContent='';
      showErr('Server is waking up — wait 30 seconds and retry.',
              'Render free tier sleeps after inactivity. First request takes ~30s.');return;
    }
    const d=await res.json();
    if(!res.ok){
      $('st').textContent='';
      const tip=d.error&&d.error.includes('No data')
        ?(!t.endsWith('.NS')&&/^[A-Z]+$/.test(t)?'For Indian stocks try '+t+'.NS (e.g. RELIANCE.NS)'
          :'Verify symbol at finance.yahoo.com'):'';
      showErr(d.error||'Error',tip);return;
    }
    if(d.suggestion){$('wb').textContent=d.suggestion;$('wb').style.display='block';$('ti').value=d.ticker;}
    $('st').textContent=d.ticker+' · '+d.n_sessions+' sessions · mean return '+(d.mean_return*100).toFixed(3)+'%';

    const dec=d.decision;
    $('h-ticker').textContent=d.ticker;
    $('h-meta').textContent=d.n_sessions+' trading sessions  ·  '+d.n_sessions+' days analysed';
    const sb=$('h-signal'); sb.textContent=dec.decision;
    sb.className='signal-badge sig-'+dec.decision;
    const cp=Math.round(dec.confidence*100);
    $('h-conf').textContent=cp+'%';
    const cc=cp>=70?'var(--g0)':cp>=45?'var(--y0)':'var(--r0)';
    $('h-cbar').style.cssText='width:'+cp+'%;background:'+cc;
    $('h-bull').textContent=dec.bull_score;
    $('h-bear').textContent=dec.bear_score;
    $('h-sess').textContent=d.n_sessions;

    $('price-title').textContent=d.ticker+' — price (last 120 sessions)';
    const n=d._prices.length;
    const isUp=d._prices[n-1]>=d._prices[0];
    const lineC=isUp?'#00b386':'#eb5757'; const fillC=isUp?'rgba(0,179,134,0.06)':'rgba(235,87,87,0.06)';
    Plotly.newPlot('cp',[{x:[...Array(n).keys()].map(i=>i-n+1),y:d._prices,
      type:'scatter',mode:'lines',line:{color:lineC,width:2},
      fill:'tozeroy',fillcolor:fillC}],{...PL,margin:{t:5,r:8,b:28,l:54}},PC);

    Plotly.newPlot('ch',[{x:d._returns,type:'histogram',nbinsx:50,
      marker:{color:'#00b386',opacity:.7}}],{...PL,margin:{t:5,r:8,b:28,l:42}},PC);

    const sc=d.davies.stress_score;
    const gc=sc>0.65?'#eb5757':sc>0.4?'#f39c12':'#00b386';
    Plotly.newPlot('cg',[{type:'indicator',mode:'gauge+number',value:Math.round(sc*100),
      gauge:{axis:{range:[0,100],tickcolor:'#c0c0c0'},bar:{color:gc,thickness:.22},
             bgcolor:'#f8f9fa',bordercolor:'#e8e8e8',
             steps:[{range:[0,40],color:'#f0fff8'},{range:[40,65],color:'#fffbf0'},
                    {range:[65,100],color:'#fff5f5'}]},
      number:{suffix:'%',font:{color:gc,size:22}},
      title:{text:d.davies.regime,font:{color:gc,size:12}}}],
      {...PL,margin:{t:18,r:16,b:10,l:16}},PC);

    buildBreakdown(dec);
    buildMods(d,dec);

    const dv=d.davies;
    $('md').innerHTML=`<div class="davies-grid">
      <div class="dv"><div class="dv-l">Stress score</div><div class="dv-v">${dv.stress_score}</div></div>
      <div class="dv"><div class="dv-l">Regime</div><div class="dv-v"><span class="tag ${tagClass(dv.regime)}">${dv.regime}</span></div></div>
      <div class="dv"><div class="dv-l">Vol ratio</div><div class="dv-v">${dv.vol_ratio}</div></div>
      <div class="dv"><div class="dv-l">Alpha</div><div class="dv-v">${dv.alpha}</div></div>
    </div>`;

    $('res').classList.remove('hidden');
  }catch(e){
    $('st').textContent='';showErr(e.message,'Check connection and retry.');
  }
}
$('ti').addEventListener('keydown',e=>{if(e.key==='Enter')analyze();});
</script>
</body>
</html>"""

@app.route("/")
def index(): return render_template_string(DASHBOARD_HTML)

@app.route("/api/analyze")
def api_analyze():
    ticker=request.args.get("ticker","").strip().upper()
    if not ticker: return safe_json({"error":"ticker parameter required"},400)
    try: return safe_json(pipeline(ticker))
    except ValueError as e: return safe_json({"error":str(e)},404)
    except CurlHTTPError as e: return safe_json({"error":f"Data fetch failed: {e}"},503)
    except Exception as e: return safe_json({"error":f"Pipeline error: {e}"},500)

@app.route("/health")
def health(): return safe_json({"status":"ok"})

if __name__=="__main__":
    port=int(os.environ.get("PORT",5000))
    app.run(host="0.0.0.0",port=port,debug=False)
