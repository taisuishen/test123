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

import requests
import websockets

# =======================
# 基本配置
# =======================
INST_ID = "ETH-USDT-SWAP"
TD_MODE = "cross"          # 全仓
ORD_TYPE = "market"        # 市价单（taker）
NET_MODE = True            # 单向持仓（net） -> 下单不传 posSide

# ====== 仓位：1x 全仓 ======
LEVERAGE = 1.0             # 1倍名义
MAX_CONTRACTS = 10.0       # 最大下单张数上限（强烈建议先小一点，确认稳定后再调大）

# ====== 触发参数 ======
VOL_LOOKBACK_SEC = 1800    # 30分钟
VOL_Q = 0.95               # 成交量分位阈值（p95）
MOVE_ATR_K = 0.35          # 净推进 >= 0.35 * ATR_1m
COOLDOWN_SEC = 60          # 同方向/同策略冷却
MIN_WARMUP_SEC = 300       # 启动后至少累计300秒成交量历史再交易

# ====== 出场规则 ======
SL_ATR_K = 0.60
TP_ATR_K = 1.00
TIME_STOP_SEC = None

# =======================
# OKX Demo endpoints / Keys
# =======================
REST_BASE = os.getenv("OKX_REST_BASE", "https://www.okx.com")

# Demo 公共 WS（建议带 brokerId=9999）
WS_PUBLIC = os.getenv(
    "OKX_WS_PUBLIC",
    "wss://wspap.okx.com:8443/ws/v5/public?brokerId=9999"
)

API_KEY = os.getenv("OKX_API_KEY", "edd59d8d-7214-4246-9c6c-6dee1b7c9d1d")
API_SECRET = os.getenv("OKX_API_SECRET", "29737A3E27AD6B9C0C145CC6BC5A4509")
API_PASSPHRASE = os.getenv("OKX_API_PASSPHRASE", "Sxw1998021299..")

# Demo 必须带这个头
SIM_HEADER = {"x-simulated-trading": "1"}

if not API_KEY or not API_SECRET or not API_PASSPHRASE:
    raise RuntimeError("Please set env OKX_API_KEY / OKX_API_SECRET / OKX_API_PASSPHRASE")

# =======================
# OKX 签名工具（你验证过的版本）
# =======================
def iso_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S.", time.gmtime()) + f"{int((time.time()%1)*1000):03d}Z"

def sign_okx(ts: str, method: str, request_path: str, body: str) -> str:
    prehash = f"{ts}{method}{request_path}{body}"
    mac = hmac.new(API_SECRET.encode("utf-8"), prehash.encode("utf-8"), digestmod="sha256").digest()
    return base64.b64encode(mac).decode()

def okx_headers(method: str, request_path: str, body: str = "") -> dict:
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

def rest_get(path: str, params: dict | None = None) -> dict:
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

def rest_post(path: str, payload: dict) -> dict:
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
# 交易所信息 / 计算
# =======================
@dataclass
class ContractSpec:
    ctVal: float
    lotSz: float
    minSz: float
    ctValCcy: str
    settleCcy: str

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

def round_down_to_step(x: float, step: float) -> float:
    return math.floor(x / step) * step

def get_equity_usdt() -> float:
    j = rest_get("/api/v5/account/balance", {"ccy": "USDT"})
    if j.get("code") != "0" or not j.get("data"):
        raise RuntimeError(f"get balance failed: {j}")
    data0 = j["data"][0]

    # 尽量保守：优先 totalEq，不行再 details->eq
    if data0.get("totalEq"):
        return float(data0["totalEq"])
    for d in data0.get("details", []):
        if d.get("ccy") == "USDT" and d.get("eq"):
            return float(d["eq"])
    raise RuntimeError(f"cannot parse equity: {j}")

def get_atr_1m(period: int = 14, limit: int = 200) -> float:
    j = rest_get("/api/v5/market/candles", {"instId": INST_ID, "bar": "1m", "limit": str(limit)})
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

def calc_contracts_1x_cross(spec: ContractSpec, equity_usdt: float, price: float) -> float:
    """
    1x 全仓：名义价值 = equity_usdt * LEVERAGE
    ETH数量 = 名义 / 价格
    合约张数 = ETH数量 / ctVal
    """
    if price <= 0:
        return 0.0
    notional_usdt = equity_usdt * LEVERAGE
    eth_qty = notional_usdt / price
    contracts = eth_qty / spec.ctVal

    qty = round_down_to_step(contracts, spec.lotSz)
    if qty < spec.minSz:
        return 0.0

    # 上限保护
    qty = min(qty, MAX_CONTRACTS)
    qty = round_down_to_step(qty, spec.lotSz)
    if qty < spec.minSz:
        return 0.0
    return qty

# =======================
# 下单 / 平仓（市价）
# =======================
def place_market(side: str, sz: float) -> dict:
    payload = {
        "instId": INST_ID,
        "tdMode": TD_MODE,
        "side": side,        # buy / sell
        "ordType": ORD_TYPE, # market
        "sz": str(sz),
    }
    # 单向持仓 net：不传 posSide
    return rest_post("/api/v5/trade/order", payload)

def close_market(pos_side: str, sz: float) -> dict:
    # 单向持仓：平仓方向与开仓相反
    side = "sell" if pos_side == "long" else "buy"
    return place_market(side, sz)

# =======================
# 运行时状态
# =======================
@dataclass
class Position:
    side: str       # "long" / "short"
    entry_px: float
    sz: float
    tp_px: float
    sl_px: float
    open_ts: float

class SweepBot:
    def __init__(self, spec: ContractSpec):
        self.spec = spec
        self.position: Position | None = None
        self.last_trade_px: float | None = None

        self.cooldown_until = 0.0

        self.vol_hist = deque(maxlen=VOL_LOOKBACK_SEC)  # 每秒成交量
        self.start_ts = time.time()

        self.cur_sec = None
        self.sec_vol = 0.0
        self.sec_first_px = None
        self.sec_last_px = None
        self.sec_count = 0

    def ready(self) -> bool:
        return (time.time() - self.start_ts) >= MIN_WARMUP_SEC and len(self.vol_hist) >= MIN_WARMUP_SEC

    def vol_threshold(self) -> float:
        if not self.ready():
            return float("inf")
        arr = sorted(self.vol_hist)
        idx = int(len(arr) * VOL_Q) - 1
        idx = max(0, min(idx, len(arr) - 1))
        return arr[idx]

    async def on_trade(self, t: dict):
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
            # reset to new second
            self.cur_sec = sec
            self.sec_vol = 0.0
            self.sec_first_px = px
            self.sec_last_px = px
            self.sec_count = 0

        self.sec_vol += sz
        self.sec_last_px = px
        self.sec_count += 1

        await self.check_exit()

    async def on_second_bar(self):
        if self.sec_first_px is None or self.sec_last_px is None:
            return

        V = self.sec_vol
        dP = self.sec_last_px - self.sec_first_px
        self.vol_hist.append(V)

        # 只在无仓位时开新仓
        if self.position is not None:
            return
        if time.time() < self.cooldown_until:
            return

        v_th = self.vol_threshold()
        if V < v_th:
            return

        # 成交量满足后再计算 ATR（避免频繁 REST）
        try:
            atr = get_atr_1m(14, 200)
        except Exception as e:
            print("[ATR ERROR]", e)
            self.cooldown_until = time.time() + COOLDOWN_SEC
            return

        if abs(dP) < MOVE_ATR_K * atr:
            return

        # 计算 1x 全仓张数（用触发秒最后价估算）
        try:
            equity = get_equity_usdt()
        except Exception as e:
            print("[EQUITY ERROR]", e)
            self.cooldown_until = time.time() + COOLDOWN_SEC
            return

        ref_px = self.last_trade_px if self.last_trade_px else self.sec_last_px
        qty = calc_contracts_1x_cross(self.spec, equity, ref_px)
        if qty <= 0:
            print("[SKIP] qty too small or capped to 0")
            self.cooldown_until = time.time() + COOLDOWN_SEC
            return

        side = "buy" if dP > 0 else "sell"
        pos_side = "long" if side == "buy" else "short"

        entry_px = ref_px
        stop_dist = SL_ATR_K * atr
        tp_dist = TP_ATR_K * atr

        sl_px = entry_px - stop_dist if pos_side == "long" else entry_px + stop_dist
        tp_px = entry_px + tp_dist if pos_side == "long" else entry_px - tp_dist

        print(
            f"[ENTRY SIGNAL] {pos_side} qty={qty} V={V:.4f} v_th={v_th:.4f} "
            f"dP={dP:.3f} atr={atr:.3f} equity={equity:.2f} ref_px={ref_px:.2f}"
        )

        resp = place_market(side, qty)
        print("[ORDER RESP]", resp)

        if resp.get("code") == "0":
            self.position = Position(
                side=pos_side,
                entry_px=entry_px,
                sz=qty,
                tp_px=tp_px,
                sl_px=sl_px,
                open_ts=time.time(),
            )

        self.cooldown_until = time.time() + COOLDOWN_SEC

    async def check_exit(self):
        if self.position is None or self.last_trade_px is None:
            return

        p = self.position
        px = self.last_trade_px

        hit_tp = (px >= p.tp_px) if p.side == "long" else (px <= p.tp_px)
        hit_sl = (px <= p.sl_px) if p.side == "long" else (px >= p.sl_px)
        hit_time = False
        if TIME_STOP_SEC is not None:
            hit_time = (time.time() - p.open_ts) >= TIME_STOP_SEC

        if not (hit_tp or hit_sl or hit_time):
            return

        reason = "TP" if hit_tp else "SL" if hit_sl else "TIME"
        print(
            f"[EXIT:{reason}] {p.side} px={px:.2f} entry={p.entry_px:.2f} "
            f"tp={p.tp_px:.2f} sl={p.sl_px:.2f} sz={p.sz}"
        )

        resp = close_market(p.side, p.sz)
        print("[CLOSE RESP]", resp)

        self.position = None
        self.cooldown_until = time.time() + COOLDOWN_SEC

# =======================
# WS 循环
# =======================
async def ws_loop(bot: SweepBot):
    while True:
        try:
            async with websockets.connect(WS_PUBLIC, ping_interval=20, ping_timeout=20) as ws:
                sub = {
                    "op": "subscribe",
                    "args": [
                        {"channel": "trades", "instId": INST_ID},
                        {"channel": "books5", "instId": INST_ID},
                    ],
                }
                await ws.send(json.dumps(sub))
                print("[WS] subscribed", sub)

                # 先读订阅回包
                for _ in range(2):
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=5)
                        print("[WS RAW]", msg)
                    except asyncio.TimeoutError:
                        break

                while True:
                    msg = await asyncio.wait_for(ws.recv(), timeout=30)
                    j = json.loads(msg)

                    ch = j.get("arg", {}).get("channel")
                    if "event" in j:
                        print("[WS EVENT]", j)

                    if ch == "trades":
                        for t in j.get("data", []):
                            await bot.on_trade(t)

                    # books5 暂时不做过滤；你后续要加 spread 过滤可以在这里处理
        except asyncio.TimeoutError:
            print("[WS] 30s no message -> reconnect")
        except Exception as e:
            print("[WS ERROR]", repr(e))
            await asyncio.sleep(2)

async def main():
    spec = get_contract_spec()
    print("[SPEC]", spec)
    bot = SweepBot(spec)
    await ws_loop(bot)

if __name__ == "__main__":
    asyncio.run(main())
