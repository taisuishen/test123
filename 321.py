import os
import time
import json
import hmac
import base64
import requests
from urllib.parse import urlencode
# ========= 你的测试参数 =========
INST_ID = "ETH-USDT-SWAP"
TD_MODE = "cross"          # 全仓
OPEN_SIDE = "buy"          # buy=开多, sell=开空
SZ = "0.01"                # 下单张数（你这合约 lotSz=minSz=0.01，所以 0.01 可测）
SLEEP_BEFORE_CLOSE = 3     # 开仓后等几秒再平仓

# ========= OKX Demo 配置 =========
REST_BASE = os.getenv("OKX_REST_BASE", "https://www.okx.com")
SIM_HEADER = {"x-simulated-trading": "1"}  # Demo 必须带

API_KEY = os.getenv("OKX_API_KEY", "edd59d8d-7214-4246-9c6c-6dee1b7c9d1d")
API_SECRET = os.getenv("OKX_API_SECRET", "29737A3E27AD6B9C0C145CC6BC5A4509")
API_PASSPHRASE = os.getenv("OKX_API_PASSPHRASE", "Sxw1998021299..")

if not API_KEY or not API_SECRET or not API_PASSPHRASE:
    raise RuntimeError("Please set env OKX_API_KEY / OKX_API_SECRET / OKX_API_PASSPHRASE")

def iso_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S.", time.gmtime()) + f"{int((time.time()%1)*1000):03d}Z"

def sign_okx(ts: str, method: str, path: str, body: str) -> str:
    msg = f"{ts}{method}{path}{body}"
    mac = hmac.new(API_SECRET.encode(), msg.encode(), digestmod="sha256").digest()
    return base64.b64encode(mac).decode()

def okx_headers(method: str, path: str, body: str = "") -> dict:
    ts = iso_ts()
    sig = sign_okx(ts, method, path, body)
    return {
        "Content-Type": "application/json",
        "OK-ACCESS-KEY": API_KEY,
        "OK-ACCESS-SIGN": sig,
        "OK-ACCESS-TIMESTAMP": ts,
        "OK-ACCESS-PASSPHRASE": API_PASSPHRASE,
        **SIM_HEADER,
    }

def rest_post(path: str, payload: dict) -> dict:
    body = json.dumps(payload, separators=(",", ":"))
    url = REST_BASE + path
    r = requests.post(url, data=body, headers=okx_headers("POST", path, body), timeout=15)
    r.raise_for_status()
    return r.json()

def rest_get(path: str, params: dict | None = None) -> dict:
    qs = ""
    if params:
        # OKX：GET 的 query 参数属于 requestPath（不是 body）
        qs = "?" + urlencode(params)
    request_path = path + qs

    url = REST_BASE + request_path
    r = requests.get(url, headers=okx_headers("GET", request_path, ""), timeout=15)

    if r.status_code != 200:
        print(f"[GET] HTTP {r.status_code} url={url}")
        print("Response:", r.text)
        r.raise_for_status()
    return r.json()

def get_positions():
    # 单向持仓下也能返回 pos / avgPx 等
    return rest_get("/api/v5/account/positions", {"instId": INST_ID})

def place_market(side: str, sz: str) -> dict:
    payload = {
        "instId": INST_ID,
        "tdMode": TD_MODE,
        "side": side,        # buy / sell
        "ordType": "market",
        "sz": sz,
    }
    return rest_post("/api/v5/trade/order", payload)

def close_market_by_side(open_side: str, sz: str) -> dict:
    # 单向持仓：开多用 buy，平仓用 sell；开空用 sell，平仓用 buy
    close_side = "sell" if open_side == "buy" else "buy"
    return place_market(close_side, sz)

def main():
    print("=== 1) 开仓（市价）===")
    open_resp = place_market(OPEN_SIDE, SZ)
    print(json.dumps(open_resp, ensure_ascii=False, indent=2))

    if open_resp.get("code") != "0":
        print("开仓失败，结束。")
        return

    print("\n=== 2) 等待成交/仓位更新 ===")
    time.sleep(SLEEP_BEFORE_CLOSE)

    print("\n=== 3) 查询当前持仓（用于确认真的有仓）===")
    pos = get_positions()
    print(json.dumps(pos, ensure_ascii=False, indent=2))

    print("\n=== 4) 平仓（市价）===")
    close_resp = close_market_by_side(OPEN_SIDE, SZ)
    print(json.dumps(close_resp, ensure_ascii=False, indent=2))

    print("\n=== 5) 等待平仓生效后再查一次持仓 ===")
    time.sleep(2)
    pos2 = get_positions()
    print(json.dumps(pos2, ensure_ascii=False, indent=2))

    print("\nDone.")

if __name__ == "__main__":
    main()
