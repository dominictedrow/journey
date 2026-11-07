import requests
from datetime import datetime, timezone
from collectors.base import save_market_data

COINGECKO_GLOBAL   = "https://api.coingecko.com/api/v3/global"
COINGECKO_PRICE    = "https://api.coingecko.com/api/v3/simple/price"
FEAR_GREED_URL     = "https://api.alternative.me/fng/"
STABLECOIN_URL     = "https://stablecoins.llama.fi/stablecoins"
DEFI_TVL_URL       = "https://api.llama.fi/v2/chains"

HEADERS = {"User-Agent": "MarketResearchBot/1.0"}

# CoinGecko id -> (series_id, label)
COIN_PRICES = {
    "bitcoin":  ("BTC_USD", "Bitcoin (USD)"),
    "ethereum": ("ETH_USD", "Ethereum (USD)"),
    "solana":   ("SOL_USD", "Solana (USD)"),
}


def collect():
    rows = []
    now = datetime.now(timezone.utc)

    # --- CoinGecko global (BTC dominance) ---
    try:
        resp = requests.get(COINGECKO_GLOBAL, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        gdata = resp.json().get("data", {})
        btc_dom = gdata.get("market_cap_percentage", {}).get("btc")
        if btc_dom is not None:
            rows.append({
                "series_id":   "BTC_DOMINANCE",
                "metric_name": "BTC Dominance %",
                "value":       float(btc_dom),
                "text_value":  None,
                "source":      "CoinGecko",
                "category":    "crypto",
                "tier":        2,
                "recorded_at": now,
            })
    except Exception as e:
        print(f"crypto: CoinGecko global error: {e}")

    # --- CoinGecko prices (single batched request) ---
    try:
        ids = ",".join(COIN_PRICES.keys())
        resp = requests.get(COINGECKO_PRICE, params={
            "ids":           ids,
            "vs_currencies": "usd",
        }, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        price_data = resp.json()
        for coin_id, (series_id, label) in COIN_PRICES.items():
            price = price_data.get(coin_id, {}).get("usd")
            if price is not None:
                rows.append({
                    "series_id":   series_id,
                    "metric_name": label,
                    "value":       float(price),
                    "text_value":  None,
                    "source":      "CoinGecko",
                    "category":    "crypto",
                    "tier":        2,
                    "recorded_at": now,
                })
    except Exception as e:
        print(f"crypto: CoinGecko prices error: {e}")

    # --- Fear & Greed ---
    try:
        resp = requests.get(FEAR_GREED_URL, timeout=10)
        resp.raise_for_status()
        fng = resp.json()["data"][0]
        rows.append({
            "series_id":   "FEAR_GREED",
            "metric_name": "Crypto Fear & Greed",
            "value":       float(fng["value"]),
            "text_value":  fng.get("value_classification"),
            "source":      "Alternative.me",
            "category":    "crypto",
            "tier":        2,
            "recorded_at": now,
        })
    except Exception as e:
        print(f"crypto: Fear & Greed error: {e}")

    # --- Stablecoin market cap ---
    try:
        resp = requests.get(STABLECOIN_URL, timeout=15)
        resp.raise_for_status()
        stables = resp.json().get("peggedAssets", [])
        total_mcap = sum(
            float(s.get("circulating", {}).get("peggedUSD", 0) or 0)
            for s in stables
        )
        if total_mcap > 0:
            rows.append({
                "series_id":   "STABLECOIN_MCAP",
                "metric_name": "Stablecoin Market Cap",
                "value":       total_mcap,
                "text_value":  None,
                "source":      "DefiLlama",
                "category":    "crypto",
                "tier":        2,
                "recorded_at": now,
            })
    except Exception as e:
        print(f"crypto: stablecoin error: {e}")

    # --- DeFi TVL ---
    try:
        resp = requests.get(DEFI_TVL_URL, timeout=15)
        resp.raise_for_status()
        chains = resp.json()
        total_tvl = sum(float(c.get("tvl", 0) or 0) for c in chains)
        if total_tvl > 0:
            rows.append({
                "series_id":   "DEFI_TVL",
                "metric_name": "DeFi TVL",
                "value":       total_tvl,
                "text_value":  None,
                "source":      "DefiLlama",
                "category":    "crypto",
                "tier":        2,
                "recorded_at": now,
            })
    except Exception as e:
        print(f"crypto: DeFi TVL error: {e}")

    save_market_data(rows)
    print(f"crypto: saved {len(rows)} series")


if __name__ == "__main__":
    collect()
