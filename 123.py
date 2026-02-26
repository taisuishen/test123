import os
import time
import json
import hmac
import base64
import math
import asyncio
from urllib.parse import urlencode
from collections import deque
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
from typing import Optional, Dict, Any

import requests
import websockets

# =======================
# 基本配置
# =======================
INST_ID = "ETH-USDT-SWAP"
TD_MODE = "cross"          # 全仓
ORD_TYPE = "market"        # 市价单（taker）
LEVERAGE = 1.0             # 1x 名义
MAX_CONTRACTS = 10.0       # 最大张数上限（先小后大）

# ====== 触发参数 ======
VOL_LOOKBACK_SEC = 1800    # 30分钟
VOL_Q = 0.95               # 成交量阈值分位（p95）
MOVE_ATR_K = 0.35          # 触发要求：净推进 >= MOVE_ATR_K * ATR
COOLDOWN_SEC = 60
MIN_WARMUP_SEC = 300

# ====== ATR 周期（放大这里）======
ATR_BAR = "15m"            # ✅ 推荐：15m；想更中周期：改成 "1H"
ATR_PERIOD = 14
ATR_LIMIT = 200

# ====== 出场规则（TP/SL 只依赖 ATR_BAR 的 ATR）======
SL_ATR_K = 0.60
TP_ATR_K = 1.00

# =======================
# OKX Demo endpoints / Keys
# =======================
REST_BASE = os.getenv("OKX_REST_BASE", "https://www.okx.com")
WS_PUBLIC = os.getenv("OKX_WS_PUBLIC", "wss://wspap.okx.com:8443/ws/v5/public?brokerId=9999")

API_KEY = os.getenv("OKX_API_KEY", "edd59d8d-7214-4246-9c6c-6dee1b7c9d1d")
API_SECRET = os.getenv("OKX_API_SECRET", "29737A3E27AD6B9C0C145CC6BC5A4509")
API_PASSPHRASE = os.getenv("OKX_API_PASSPHRASE", "Sxw1998021299..")

SIM_HEADER = {"x-simulated-trading": "1"}  # Demo 必须

if not API_KEY or not API_SECRET or not API_PASSPHRASE:
    raise RuntimeError("Please set env OKX_API_KEY / OKX_API_SECRET / OKX_API_PASSPHRASE")

# =======================
# OKX 签名（GET query 入签名）
# =======================
def iso_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S.", time.gmtime()) + f"{int((time.time()%1)*1000):03d}Z"

def sign_okx(ts: str, method: str, request_path: str, body: str) -> str:
    prehash = f"{ts}{method}{request_path}{body}"
    mac = hmac.new(API_SECRET.encode("utf-8"), prehash.encode("utf-8"), digestmod="sha256").digest()
    return base64.b64encode(mac).decode()

def okx_headers(method: str, request_path: str, body: str = "") -> Dict[str, str]:
    ts = iso_ts()
    sig = sign_okx(ts, method, request_path, body)
    return {
        "Content-Type": "application/json",
        "OK-ACCESS-KEY": API_KEY,
        "OK-ACCESS-SIGN": sig,
        "OK-ACCESS-TIMESTAMP": ts,
        "OK-ACCESS-PASSPHRASE": API_PASSPHRASE,
        **SIM_HEADER,
    }

def rest_get(path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    qs = ""
    if params:
        qs = "?" + urlencode(params)
    request_path = path + qs
    url = REST_BASE + request_path

    r = requests.get(url, headers=okx_headers("GET", request_path, ""), timeout=15)
    if r.status_code != 200:
        print(f"[GET] HTTP {r.status_code} url={url}")
        print("Response:", r.text)
        r.raise_for_status()
    return r.json()

def rest_post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    request_path = path
    body = json.dumps(payload, separators=(",", ":"))
    url = REST_BASE + request_path

    r = requests.post(url, data=body, headers=okx_headers("POST", request_path, body), timeout=15)
    if r.status_code != 200:
        print(f"[POST] HTTP {r.status_code} url={url}")
        print("Response:", r.text)
        r.raise_for_status()
    return r.json()

# =======================
# 合约规格 / 精度工具（解决 51121）
# =======================
@dataclass
class ContractSpec:
    ctVal: float
    lotSz: float
    minSz: float
    ctValCcy: str
    settleCcy: str

def step_decimals(step: float) -> int:
    s = format(step, "f")
    if "." in s:
        return len(s.rstrip("0").split(".")[1])
    return 0

def floor_to_step(x: float, step: float) -> float:
    dx = Decimal(str(x))
    ds = Decimal(str(step))
    n = (dx / ds).to_integral_value(rounding=ROUND_DOWN)
    return float(n * ds)

def fmt_by_step(x: float, step: float) -> str:
    d = step_decimals(step)
    q = Decimal("1") if d == 0 else Decimal("0." + "0"*(d-1) + "1")
    return str(Decimal(str(x)).quantize(q, rounding=ROUND_DOWN))

def get_contract_spec() -> ContractSpec:
    j = rest_get("/api/v5/public/instruments", {"instType": "SWAP", "instId": INST_ID})
    if j.get("code") != "0" or not j.get("data"):
        raise RuntimeError(f"get instruments failed: {j}")
    x = j["data"][0]
    return ContractSpec(
        ctVal=float(x["ctVal"]),
        lotSz=float(x["lotSz"]),
        minSz=float(x["minSz"]),
        ctValCcy=x.get("ctValCcy", ""),
        settleCcy=x.get("settleCcy", ""),
    )

def get_equity_usdt() -> float:
    j = rest_get("/api/v5/account/balance", {"ccy": "USDT"})
    if j.get("code") != "0" or not j.get("data"):
        raise RuntimeError(f"get balance failed: {j}")
    data0 = j["data"][0]
    if data0.get("totalEq"):
        return float(data0["totalEq"])
    for d in data0.get("details", []):
        if d.get("ccy") == "USDT" and d.get("eq"):
            return float(d["eq"])
    raise RuntimeError(f"cannot parse equity: {j}")

def get_atr(bar: str, period: int = 14, limit: int = 200) -> float:
    j = rest_get("/api/v5/market/candles", {"instId": INST_ID, "bar": bar, "limit": str(limit)})
    if j.get("code") != "0" or not j.get("data"):
        raise RuntimeError(f"get candles failed: {j}")

    rows = list(reversed(j["data"]))  # 升序
    highs = [float(r[2]) for r in rows]
    lows  = [float(r[3]) for r in rows]
    closes= [float(r[4]) for r in rows]

    trs = []
    for i in range(1, len(rows)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i-1]),
            abs(lows[i] - closes[i-1]),
        )
        trs.append(tr)

    if len(trs) < period:
        raise RuntimeError("not enough candles for ATR")

    atr = sum(trs[:period]) / period
    for tr in trs[period:]:
        atr = (atr * (period - 1) + tr) / period
    return atr

def calc_contracts_1x(spec: ContractSpec, equity_usdt: float, price: float) -> float:
    if price <= 0:
        return 0.0
    notional = equity_usdt * LEVERAGE
    eth_qty = notional / price
    contracts = eth_qty / spec.ctVal

    qty = floor_to_step(contracts, spec.lotSz)
    if qty < spec.minSz:
        return 0.0

    cap = floor_to_step(MAX_CONTRACTS, spec.lotSz)
    qty = min(qty, cap)
    qty = floor_to_step(qty, spec.lotSz)
    if qty < spec.minSz:
        return 0.0
    return qty

# =======================
# 下单：随单附带 TP/SL（触发后市价执行）
# =======================
def place_market_with_tpsl(spec: ContractSpec, side: str, sz: float, tp_px: float, sl_px: float) -> Dict[str, Any]:
    sz_str = fmt_by_step(sz, spec.lotSz)

    payload = {
        "instId": INST_ID,
        "tdMode": TD_MODE,
        "side": side,         # buy / sell
        "ordType": ORD_TYPE,  # market
        "sz": sz_str,

        "attachAlgoOrds": [{
            "tpTriggerPx": str(tp_px),
            "tpOrdPx": "-1",            # -1 触发后市价
            "slTriggerPx": str(sl_px),
            "slOrdPx": "-1",            # -1 触发后市价
            "tpTriggerPxType": "last",
            "slTriggerPxType": "last",
        }]
    }
    return rest_post("/api/v5/trade/order", payload)

# =======================
# 运行状态
# =======================
@dataclass
class PositionLocal:
    side: str
    entry_px: float
    sz: float
    tp_px: float
    sl_px: float
    open_ts: float

class SweepBot:
    def __init__(self, spec: ContractSpec):
        self.spec = spec
        self.position: Optional[PositionLocal] = None
        self.last_trade_px: Optional[float] = None

        self.cooldown_until = 0.0
        self.vol_hist = deque(maxlen=VOL_LOOKBACK_SEC)
        self.start_ts = time.time()

        self.cur_sec = None
        self.sec_vol = 0.0
        self.sec_first_px = None
        self.sec_last_px = None

    def ready(self) -> bool:
        return (time.time() - self.start_ts) >= MIN_WARMUP_SEC and len(self.vol_hist) >= MIN_WARMUP_SEC

    def vol_threshold(self) -> float:
        if not self.ready():
            return float("inf")
        arr = sorted(self.vol_hist)
        idx = int(len(arr) * VOL_Q) - 1
        idx = max(0, min(idx, len(arr) - 1))
        return arr[idx]

    async def on_trade(self, t: Dict[str, Any]):
        px = float(t["px"])
        sz = float(t["sz"])
        ts_ms = int(t["ts"])
        self.last_trade_px = px

        sec = ts_ms // 1000
        if self.cur_sec is None:
            self.cur_sec = sec
            self.sec_first_px = px
            self.sec_last_px = px

        if sec != self.cur_sec:
            await self.on_second_bar()
            self.cur_sec = sec
            self.sec_vol = 0.0
            self.sec_first_px = px
            self.sec_last_px = px

        self.sec_vol += sz
        self.sec_last_px = px

    async def on_second_bar(self):
        if self.sec_first_px is None or self.sec_last_px is None:
            return

        V = self.sec_vol
        dP = self.sec_last_px - self.sec_first_px
        self.vol_hist.append(V)

        if self.position is not None:
            return
        if time.time() < self.cooldown_until:
            return

        v_th = self.vol_threshold()
        if V < v_th:
            return

        # ✅ ATR 用更大周期
        try:
            atr = get_atr(ATR_BAR, ATR_PERIOD, ATR_LIMIT)
        except Exception as e:
            print("[ATR ERROR]", e)
            self.cooldown_until = time.time() + COOLDOWN_SEC
            return

        if abs(dP) < MOVE_ATR_K * atr:
            return

        try:
            equity = get_equity_usdt()
        except Exception as e:
            print("[EQUITY ERROR]", e)
            self.cooldown_until = time.time() + COOLDOWN_SEC
            return

        ref_px = self.last_trade_px if self.last_trade_px else self.sec_last_px
        qty = calc_contracts_1x(self.spec, equity, ref_px)
        if qty <= 0:
            print("[SKIP] qty too small or capped to 0")
            self.cooldown_until = time.time() + COOLDOWN_SEC
            return

        side = "buy" if dP > 0 else "sell"
        pos_side = "long" if side == "buy" else "short"

        entry_px = ref_px
        sl_dist = SL_ATR_K * atr
        tp_dist = TP_ATR_K * atr

        sl_px = entry_px - sl_dist if pos_side == "long" else entry_px + sl_dist
        tp_px = entry_px + tp_dist if pos_side == "long" else entry_px - tp_dist

        print(
            f"[ENTRY SIGNAL] {pos_side} qty={qty} V={V:.4f} v_th={v_th:.4f} dP={dP:.3f} "
            f"ATR({ATR_BAR})={atr:.3f} equity={equity:.2f} entry_ref={entry_px:.2f} TP={tp_px:.2f} SL={sl_px:.2f}"
        )

        resp = place_market_with_tpsl(self.spec, side, qty, tp_px, sl_px)
        print("[ORDER RESP]", resp)

        if resp.get("code") == "0":
            self.position = PositionLocal(
                side=pos_side,
                entry_px=entry_px,
                sz=qty,
                tp_px=tp_px,
                sl_px=sl_px,
                open_ts=time.time(),
            )

        self.cooldown_until = time.time() + COOLDOWN_SEC

# =======================
# WS 循环
# =======================
async def ws_loop(bot: SweepBot):
    while True:
        try:
            async with websockets.connect(WS_PUBLIC, ping_interval=20, ping_timeout=20) as ws:
                sub = {"op": "subscribe", "args": [{"channel": "trades", "instId": INST_ID}]}
                await ws.send(json.dumps(sub))
                print("[WS] subscribed", sub)

                while True:
                    msg = await asyncio.wait_for(ws.recv(), timeout=30)
                    j = json.loads(msg)
                    ch = j.get("arg", {}).get("channel")

                    if "event" in j:
                        print("[WS EVENT]", j)

                    if ch == "trades":
                        for t in j.get("data", []):
                            await bot.on_trade(t)

        except asyncio.TimeoutError:
            print("[WS] 30s no message -> reconnect")
        except Exception as e:
            print("[WS ERROR]", repr(e))
            await asyncio.sleep(2)

async def main():
    spec = get_contract_spec()
    print("[SPEC]", spec, "ATR_BAR=", ATR_BAR)
    bot = SweepBot(spec)
    await ws_loop(bot)

if __name__ == "__main__":
    asyncio.run(main())
