
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import pandas as pd, numpy as np, yfinance as yf, requests, os, json
from datetime import datetime, timezone
from typing import Optional

ROOT=Path(__file__).parent
DATA=ROOT/"data"; DATA.mkdir(exist_ok=True)
app=FastAPI(title="Market Monitor")
app.mount("/static",StaticFiles(directory=ROOT/"static"),name="static")

ASSETS={
 "Japan":{"Nikkei 225":"^N225","TOPIX ETF (1306)":"1306.T","USD/JPY":"JPY=X","Japan Volatility (proxy)":"^N225"},
 "US":{"S&P 500":"^GSPC","Nasdaq 100":"^NDX","VIX":"^VIX","US 10Y yield":"^TNX","Gold":"GC=F","Crude Oil":"CL=F"}
}
US_SECTORS={"Technology":"XLK","Financials":"XLF","Health Care":"XLV","Energy":"XLE","Industrials":"XLI","Consumer Discretionary":"XLY","Consumer Staples":"XLP","Utilities":"XLU","Materials":"XLB","Real Estate":"XLRE","Communication Services":"XLC"}
JP_SECTORS={"TOPIX ETF (broad market)":"1306.T"} # Sector-level Japanese flow data requires a licensed/source-specific feed.
def fetch_history(ticker,period="10y"):
    try:
        d=yf.download(ticker,period=period,interval="1d",auto_adjust=True,progress=False,threads=False)
        if d is None or d.empty: return pd.DataFrame()
        if isinstance(d.columns,pd.MultiIndex): d.columns=d.columns.get_level_values(0)
        return d.dropna(subset=["Close"])
    except Exception: return pd.DataFrame()
def series(ticker,period="1y"):
    d=fetch_history(ticker,period)
    if d.empty:return []
    return [{"date":str(i.date()),"value":round(float(r),4) if pd.notna(r) else None} for i,r in d["Close"].items()]
def latest(ticker):
    s=series(ticker,"1mo")
    return s[-1] if s else None
def fred(series_id):
    key=os.getenv("FRED_API_KEY")
    if not key:return None
    try:
        r=requests.get("https://api.stlouisfed.org/fred/series/observations",params={"series_id":series_id,"api_key":key,"file_type":"json","sort_order":"desc","limit":1},timeout=15)
        r.raise_for_status(); obs=r.json().get("observations",[])
        if obs and obs[0]["value"]!=".": return {"date":obs[0]["date"],"value":float(obs[0]["value"])}
    except Exception: pass
    return None
def score_market(region):
    tickers=({"breadth":"1306.T","benchmark":"^N225","vol":"^VIX","rates":"^TNX"} if region=="Japan" else {"breadth":"SPY","benchmark":"^GSPC","vol":"^VIX","rates":"^TNX"})
    parts={}
    for k,t in tickers.items():
        d=fetch_history(t,"1y")
        if d.empty: parts[k]=None; continue
        c=d["Close"].dropna()
        if len(c)<30: parts[k]=None; continue
        if k=="vol":
            v=float(c.iloc[-1]); parts[k]=max(0,min(100,100-(v-15)*3))
        elif k=="rates":
            v=float(c.iloc[-1]); parts[k]=max(0,min(100,70-(v-3.5)*12))
        else:
            ma50=c.rolling(50).mean().iloc[-1]; ma200=c.rolling(200).mean().iloc[-1] if len(c)>=200 else c.rolling(50).mean().iloc[-1]
            ret20=float(c.pct_change(20).iloc[-1])*100
            parts[k]=max(0,min(100,50+(25 if c.iloc[-1]>ma50 else -15)+(15 if c.iloc[-1]>ma200 else -15)+ret20*1.5))
    # Conservative proxy score; breadth is not true advance/decline breadth.
    values=[v for v in parts.values() if v is not None]
    total=round(float(np.mean(values))) if values else None
    return {"score":total,"components":parts,"method":"Proxy score from index/ETF trend, VIX and Treasury yield; not true market breadth."}
def backtest(ticker):
    d=fetch_history(ticker,"10y")
    if d.empty or len(d)<260:return {"available":False,"reason":"10年分の履歴を取得できませんでした"}
    c=d["Close"].dropna(); ma50=c.rolling(50).mean(); ma200=c.rolling(200).mean()
    # Simple, reproducible trend regime: long when close > 200d MA and 50d MA > 200d MA.
    pos=((c>ma200)&(ma50>ma200)).shift(1).fillna(False).astype(float)
    daily=c.pct_change().fillna(0)
    strat=pos*daily; bh=(1+daily).cumprod(); eq=(1+strat).cumprod()
    dd=(eq/eq.cummax()-1).min()
    years=max((c.index[-1]-c.index[0]).days/365.25,1)
    cagr=float(eq.iloc[-1]**(1/years)-1)
    bhcagr=float(bh.iloc[-1]**(1/years)-1)
    return {"available":True,"ticker":ticker,"start":str(c.index[0].date()),"end":str(c.index[-1].date()),"strategy_cagr":round(cagr*100,2),"buy_hold_cagr":round(bhcagr*100,2),"max_drawdown":round(float(dd)*100,2),"note":"200日移動平均トレンド戦略。配当・税・手数料・スリッページを完全には反映しない簡易検証。過去成績は将来を保証しない。"}
@app.get("/")
def home(): return FileResponse(ROOT/"static/index.html")
@app.get("/api/dashboard")
def dashboard(refresh: bool=False):
    cache=DATA/"dashboard.json"
    if not refresh and cache.exists():
        try: return json.loads(cache.read_text(encoding="utf-8"))
        except Exception: pass
    now=datetime.now(timezone.utc).isoformat()
    out={"updated_at":now,"markets":{},"us_sectors":{},"fred":{},"foreign_flows":None}
    for region,ts in ASSETS.items():
        out["markets"][region]={"score":score_market(region),"assets":{name:latest(t) for name,t in ts.items()}}
    for name,t in US_SECTORS.items():
        d=fetch_history(t,"3mo")
        if not d.empty and len(d)>20:
            c=d["Close"].dropna()
            out["us_sectors"][name]={"ticker":t,"change_1m_pct":round(float((c.iloc[-1]/c.iloc[-22]-1)*100),2)}
    for sid,label in [("DGS10","US 10Y"),("DGS2","US 2Y"),("BAMLH0A0HYM2","US High Yield Spread"),("VIXCLS","FRED VIX")]:
        out["fred"][label]=fred(sid)
    p=DATA/"foreign_flows.csv"
    if p.exists():
        try: out["foreign_flows"]=pd.read_csv(p).tail(12).to_dict(orient="records")
        except Exception: pass
    out["data_notes"]=["株価データはYahoo Finance経由。無料公開データのため遅延・欠損・利用条件変更の可能性があります。","日本の外国人投資家売買動向はJPX公表ファイルをCSVで取り込む設計です。未取り込み時は表示しません。","日本の業種別資金流入を正確に測るには業種別売買統計またはETFフローのデータソースが別途必要です。米国セクター欄はETF価格の1か月騰落率であり、純資金流入額ではありません。"]
    return out
@app.get("/api/backtest/{region}")
def run_backtest(region:str):
    tickers={"Japan":"1306.T","US":"SPY"}
    if region not in tickers: raise HTTPException(404,"Unknown region")
    return backtest(tickers[region])
@app.post("/api/foreign-flows")
async def upload_flows(file:UploadFile=File(...)):
    if not file.filename.lower().endswith(".csv"): raise HTTPException(400,"CSVファイルを指定してください")
    content=await file.read()
    try:
        from io import BytesIO
        df=pd.read_csv(BytesIO(content))
        if len(df.columns)<2: raise ValueError()
        df.to_csv(DATA/"foreign_flows.csv",index=False)
        return {"ok":True,"rows":len(df),"columns":list(df.columns)}
    except Exception: raise HTTPException(400,"CSVを読み込めません。列見出し付きCSVを確認してください。")
@app.get("/api/report")
def report():
    d=dashboard()
    lines=["# 市場モニター日次レポート",f"更新時刻(UTC): {d['updated_at']}"]
    for r,v in d["markets"].items():
        s=v["score"]; label="強気寄り" if s and s>=65 else "中立" if s and s>=45 else "警戒" if s is not None else "データ不足"
        lines += [f"## {r}: {label} ({s if s is not None else 'N/A'}/100)",v["score"]["method"] if False else v["score"]["method"] if s is not None else "十分なデータがありません"]
        lines.append("主要指標: "+", ".join(f"{k}={x['value'] if x else 'N/A'}" for k,x in v["assets"].items()))
    lines += ["## 注意","これは機械的な参考指標であり、投資助言ではありません。データ欠損・遅延を確認してください。"]
    base="\n".join(lines)
    api_key=os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            rr=requests.post("https://api.openai.com/v1/chat/completions",headers={"Authorization":f"Bearer {api_key}"},json={"model":os.getenv("OPENAI_MODEL","gpt-4.1-mini"),"messages":[{"role":"system","content":"あなたは市場データを慎重に要約するアナリストです。与えられたデータだけを使い、欠損を明示し、売買を断定せず、日本語でリスクと反対シナリオも説明してください。"},{"role":"user","content":base}],"temperature":0.2},timeout=45)
            rr.raise_for_status(); base=rr.json()["choices"][0]["message"]["content"]
        except Exception as e: base += "\n\nAIレポート接続失敗。ルールベースの内容を表示しています。"
    return {"report":base}
