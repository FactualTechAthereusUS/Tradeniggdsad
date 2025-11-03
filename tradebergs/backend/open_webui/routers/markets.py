from __future__ import annotations

import time
from typing import List, Literal, Optional
import json
from pathlib import Path

import httpx
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

router = APIRouter(prefix="/api/markets", tags=["markets"])


YF_BASE = "https://query2.finance.yahoo.com"
YF_SEARCH = f"{YF_BASE}/v1/finance/search"
YF_CHART = f"{YF_BASE}/v8/finance/chart"
BINANCE_API = "https://api.binance.com"
UA_HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0 Safari/537.36"}


@router.get("/search")
async def search_symbols(
    q: str = Query(""),
    cls: Literal["all", "crypto", "stocks", "forex", "futures"] = "all",
    limit: int = 30,
    page: int = 1,
):
    results: List[dict] = []
    # Load local universe first for paging stability
    def load_universe(kind: str) -> List[dict]:
        base = Path(__file__).resolve().parent.parent / "static" / "universe"
        fp = base / f"{kind}.json"
        if fp.exists():
            try:
                return json.loads(fp.read_text())
            except Exception:
                return []
        return []

    if cls in ("stocks", "forex", "futures", "all"):
        kinds = ([cls] if cls != "all" else ["stocks", "forex", "futures"])
        uni = []
        for k in kinds:
            uni.extend(load_universe(k))
        if q:
            ql = q.lower()
            uni = [u for u in uni if ql in u.get("symbol", "").lower() or ql in u.get("name", "").lower()]
        # paginate
        start = (page - 1) * limit
        for u in uni[start : start + limit]:
            results.append({"symbol": u.get("symbol"), "description": u.get("name", ""), "source": "yahoo"})
    async with httpx.AsyncClient(timeout=10, headers=UA_HEADERS) as client:
        # Yahoo search (supplemental)
        try:
            if cls in ("all", "stocks", "forex", "futures"):
                yparams = {"q": q, "quotesCount": limit, "newsCount": 0, "lang": "en-US", "region": "US"}
                y = await client.get(YF_SEARCH, params=yparams)
                if y.status_code == 200:
                    data = y.json()
                    for it in data.get("quotes", [])[:limit]:
                        type_disp = (it.get("typeDisp") or "").lower()
                        if cls == "stocks" and type_disp not in ("equity", "etf"):
                            continue
                        if cls == "forex" and type_disp not in ("currency",):
                            continue
                        if cls == "futures" and type_disp not in ("future", "commodity"):  # yahoo uses various labels
                            continue
                        results.append(
                            {
                                "symbol": it.get("symbol"),
                                "description": it.get("shortname") or it.get("longname") or "",
                                "source": "yahoo",
                            }
                        )
        except Exception:
            pass

        # Binance markets list (basic)
        try:
            if cls in ("all", "crypto"):
                b = await client.get(f"{BINANCE_API}/api/v3/exchangeInfo")
                if b.status_code == 200:
                    data = b.json()
                    for s in data.get("symbols", [])[:2000]:
                        sym = s.get("symbol")
                        if q and q.upper() not in sym:
                            continue
                        results.append({
                            "symbol": sym,
                            "description": f"{s.get('baseAsset')}/{s.get('quoteAsset')}",
                            "source": "binance",
                        })
        except Exception:
            pass

    # If no query and still empty, provide curated defaults by class
    if not results and not q:
        defaults = {
            "stocks": [
                "AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA","SPY","QQQ","^GSPC","^NDX","^DJI"
            ],
            "forex": [
                "EURUSD=X","GBPUSD=X","USDJPY=X","AUDUSD=X","USDCAD=X","DXY"
            ],
            "futures": [
                "ES=F","NQ=F","YM=F","RTY=F","CL=F","NG=F","GC=F","SI=F","ZN=F","ZB=F"
            ],
        }
        if cls in defaults:
            for sym in defaults[cls][:limit]:
                results.append({"symbol": sym, "description": "", "source": "yahoo"})

    # De-dupe by (symbol, source)
    dedup = {}
    out = []
    for r in results:
        k = (r["symbol"], r["source"])
        if k in dedup:
            continue
        dedup[k] = True
        out.append(r)
        if len(out) >= limit:
            break
    # If still empty and a simple ticker typed, propose it directly
    if not out and q and q.isalnum() and len(q) <= 10:
        out.append({"symbol": q.upper(), "description": "", "source": "yahoo"})
    return {"results": out}


def _yf_interval(iv: str) -> str:
    # Map 1m,5m,15m,30m,1h,4h,1d
    return iv


@router.get("/candles")
async def candles(
    provider: Literal["yahoo", "binance", "yfinance"],
    symbol: str,
    interval: str = "15m",
    limit: int = 500,
):
    """Return list of [openTime, open, high, low, close]."""
    async with httpx.AsyncClient(timeout=15) as client:
        if provider == "binance":
            params = {"symbol": symbol.upper(), "interval": interval, "limit": min(limit, 1000)}
            r = await client.get(f"{BINANCE_API}/api/v3/klines", params=params)
            if r.status_code != 200:
                raise HTTPException(400, r.text)
            arr = r.json()
            return {"candles": [[k[0], k[1], k[2], k[3], k[4]] for k in arr]}
        else:
            # Yahoo chart
            iv = _yf_interval(interval)
            # choose a range large enough for limit
            range_map = {
                "1m": "1d",
                "5m": "5d",
                "15m": "1mo",
                "30m": "3mo",
                "1h": "6mo",
                "4h": "1y",
                "1d": "max",
            }
            if provider == "yfinance":
                # optional: use yfinance if installed
                try:
                    import yfinance as yf  # type: ignore

                    # map interval to yfinance codes
                    yf_iv = interval
                    period_map = {
                        "1m": "1d",
                        "5m": "5d",
                        "15m": "1mo",
                        "30m": "3mo",
                        "1h": "6mo",
                        "4h": "1y",
                        "1d": "max",
                    }
                    yf_period = period_map.get(interval, "1mo")
                    df = yf.Ticker(symbol).history(period=yf_period, interval=yf_iv)
                    if df is None or df.empty:
                        raise Exception("empty yfinance data")
                    candles = []
                    for ts, row in df.iterrows():
                        ms = int(ts.to_pydatetime().timestamp() * 1000)
                        candles.append([ms, str(row["Open"]), str(row["High"]), str(row["Low"]), str(row["Close"])])
                    return {"candles": candles[-limit:]}
                except Exception as e:  # fall back to yahoo http
                    provider = "yahoo"

            rng = range_map.get(iv, "1mo")
            r = await client.get(f"{YF_CHART}/{symbol}", params={"interval": iv, "range": rng})
            if r.status_code != 200:
                raise HTTPException(400, r.text)
            data = r.json()
            try:
                res = data["chart"]["result"][0]
                ts = res["timestamp"]
                o = res["indicators"]["quote"][0]
                out = []
                for i in range(len(ts)):
                    out.append([int(ts[i]) * 1000, str(o["open"][i]), str(o["high"][i]), str(o["low"][i]), str(o["close"][i])])
                return {"candles": out[-limit:]}
            except Exception:
                raise HTTPException(400, "Invalid Yahoo chart payload")


@router.get("/snapshot")
async def snapshot(
    symbol: str,
    interval: str = "15m",
    provider: Literal["yfinance", "binance", "yahoo"] = "yfinance",
    width: int = 1200,
    height: int = 600,
    theme: Literal["dark", "light"] = "dark",
):
    """Render a lightweight chart server-side and return PNG bytes.
    Requires pyppeteer to be installed. Falls back to 501 if not available.
    """
    # 1) load candles (reuse logic above)
    async def load_candles() -> List[List[str]]:
        async with httpx.AsyncClient(timeout=20, headers=UA_HEADERS) as client:
            if provider == "binance":
                params = {"symbol": symbol.upper(), "interval": interval, "limit": min(1000, 1000)}
                r = await client.get(f"{BINANCE_API}/api/v3/klines", params=params)
                r.raise_for_status()
                arr = r.json()
                return [[k[0], k[1], k[2], k[3], k[4]] for k in arr]
            # yfinance/yahoo
            try:
                import yfinance as yf  # type: ignore

                yf_period = {
                    "1m": "1d",
                    "5m": "5d",
                    "15m": "1mo",
                    "30m": "3mo",
                    "1h": "6mo",
                    "4h": "1y",
                    "1d": "max",
                }.get(interval, "1mo")
                df = yf.Ticker(symbol).history(period=yf_period, interval=interval)
                if df is None or df.empty:
                    raise Exception("empty yfinance data")
                out: List[List[str]] = []
                for ts, row in df.iterrows():
                    ms = int(ts.to_pydatetime().timestamp() * 1000)
                    out.append([ms, str(row["Open"]), str(row["High"]), str(row["Low"]), str(row["Close"])] )
                return out[-1000:]
            except Exception:
                # Yahoo HTTP fallback
                rng = {
                    "1m": "1d",
                    "5m": "5d",
                    "15m": "1mo",
                    "30m": "3mo",
                    "1h": "6mo",
                    "4h": "1y",
                    "1d": "max",
                }.get(interval, "1mo")
                r = await client.get(f"{YF_CHART}/{symbol}", params={"interval": interval, "range": rng})
                r.raise_for_status()
                data = r.json()
                res = data["chart"]["result"][0]
                ts = res["timestamp"]
                q = res["indicators"]["quote"][0]
                out: List[List[str]] = []
                for i in range(len(ts)):
                    out.append([int(ts[i]) * 1000, str(q["open"][i]), str(q["high"][i]), str(q["low"][i]), str(q["close"][i])])
                return out[-1000:]

    data = await load_candles()

    # 2) render via pyppeteer
    try:
        from pyppeteer import launch  # type: ignore
    except Exception:
        raise HTTPException(501, "pyppeteer not installed. pip install pyppeteer")

    html = f"""
    <html>
    <head>
      <meta charset=\"utf-8\" />
      <style>html,body,#c{{margin:0;padding:0;width:100%;height:100%;background:{'#0b0b0b' if theme=='dark' else '#ffffff'};}}</style>
    </head>
    <body>
      <div id=\"c\" style=\"width:{width}px;height:{height}px;\"></div>
      <script src=\"/static/vendor/lightweight-charts.standalone.production.js\"></script>
      <script>
        (async function(){{
          const LW = window.LightweightCharts;
          const chart = LW.createChart(document.getElementById('c'), {{
            layout: {{ background: {{ color: '{'#0b0b0b' if theme=='dark' else '#ffffff'}' }}, textColor: '{'#d1d5db' if theme=='dark' else '#111827'}' }},
            grid: {{ vertLines: {{ color: '{'#1f2937' if theme=='dark' else '#e5e7eb'}' }}, horzLines: {{ color: '{'#1f2937' if theme=='dark' else '#e5e7eb'}' }} }}
          }});
          const series = chart.addCandlestickSeries({{ upColor:'#16a34a', downColor:'#ef4444', wickUpColor:'#16a34a', wickDownColor:'#ef4444', borderVisible:false }});
          const D = {json.dumps(data)}.map(k=>({{time:k[0]/1000,open:parseFloat(k[1]),high:parseFloat(k[2]),low:parseFloat(k[3]),close:parseFloat(k[4])}}));
          series.setData(D);
          setTimeout(()=>window.done && window.done(), 200);
        }})();
      </script>
    </body>
    </html>
    """

    browser = await launch(args=["--no-sandbox", "--disable-gpu"])  # type: ignore
    page = await browser.newPage()
    await page.setViewport({"width": width, "height": height})
    # Serve from same origin for script URL; set content and wait
    await page.setContent(html, waitUntil=["networkidle0"])  # type: ignore
    await page.exposeFunction("done", lambda: None)
    # give a moment for layout
    await page.waitFor(300)  # type: ignore
    png = await page.screenshot(type="png")
    await browser.close()
    return Response(content=png, media_type="image/png")
