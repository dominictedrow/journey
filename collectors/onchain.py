import re
import statistics
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timezone
from collectors.base import save_market_data

BLOCKCHAINCENTER_URL = "https://www.blockchaincenter.net/altcoin-season-index/"
COINMETRICS_URL      = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}


def _fetch_mvrv():
    """
    Compute the true MVRV Z-Score for BTC from CoinMetrics community API (free, no key).

        Z = (market_cap - realized_cap) / stdev(market_cap over full history)

    The free community tier exposes market cap (CapMrktCurUSD) and the MVRV ratio
    (CapMVRVCur) but NOT realized cap directly, so realized cap is derived from the
    identity  realized_cap = market_cap / mvrv_ratio.  The denominator is the
    standard deviation of market cap across all of history, matching the canonical
    lookintobitcoin definition (values historically ~ -0.5 to 8+).

    Returns (z_score, recorded_at, source_name) or (None, None, None).
    """
    times, mcaps, rcaps = [], [], []
    url = COINMETRICS_URL
    params = {
        "assets": "btc",
        "metrics": "CapMrktCurUSD,CapMVRVCur",
        "frequency": "1d",
        "page_size": 10000,
    }
    try:
        # Follow pagination (next_page_url) until the full history is collected.
        for _ in range(20):  # hard cap; full BTC daily history fits well under this
            resp = requests.get(url, params=params, headers=HEADERS, timeout=30)
            if resp.status_code != 200:
                print(f"onchain: MVRV CoinMetrics returned {resp.status_code}")
                return None, None, None
            body = resp.json()
            for row in body.get("data", []):
                mc = row.get("CapMrktCurUSD")
                mv = row.get("CapMVRVCur")
                if mc is None or mv is None:
                    continue
                mc = float(mc)
                mv = float(mv)
                if mv == 0:
                    continue
                times.append(row["time"])
                mcaps.append(mc)
                rcaps.append(mc / mv)  # realized_cap = market_cap / mvrv_ratio
            next_url = body.get("next_page_url")
            if not next_url:
                break
            url, params = next_url, None  # next_page_url already carries query params
    except Exception as e:
        print(f"onchain: MVRV CoinMetrics error: {e}")
        return None, None, None

    if len(mcaps) < 2:
        return None, None, None

    sd = statistics.pstdev(mcaps)
    if sd == 0:
        return None, None, None

    z = (mcaps[-1] - rcaps[-1]) / sd
    ts = datetime.fromisoformat(times[-1].replace("Z", "+00:00"))
    return z, ts, "CoinMetrics"


def _fetch_altcoin_season():
    """Scrape Blockchaincenter altcoin season index."""
    try:
        resp = requests.get(BLOCKCHAINCENTER_URL, headers=HEADERS, timeout=20)
        resp.raise_for_status()

        for pattern in [
            r'"altcoinSeason"\s*:\s*(\d+)',
            r'season[^"]{0,80}?(\d{1,3})\s*(?:/100|out of)',
            r'(?:index|score)[^"]{0,30}?[>:]\s*(\d{1,3})\b',
            r'season.*?(\d{1,3})',
        ]:
            match = re.search(pattern, resp.text[:20000], re.IGNORECASE)
            if match:
                val = int(match.group(1))
                if 0 <= val <= 100:
                    return val, datetime.now(timezone.utc)

        soup = BeautifulSoup(resp.text, "html.parser")
        for el in soup.find_all(attrs={"data-value": True}):
            try:
                val = int(el["data-value"])
                if 0 <= val <= 100:
                    return val, datetime.now(timezone.utc)
            except (ValueError, TypeError):
                continue
    except Exception as e:
        print(f"onchain: Blockchaincenter error: {e}")
    return None, None


def collect():
    rows = []

    mvrv, recorded_at, mvrv_source = _fetch_mvrv()
    if mvrv is not None:
        rows.append({
            "series_id":   "MVRV_ZSCORE",
            "metric_name": "MVRV Z-Score",
            "value":       mvrv,
            "text_value":  None,
            "source":      mvrv_source,
            "category":    "crypto",
            "tier":        2,
            "recorded_at": recorded_at,
        })
    else:
        print("onchain: MVRV not fetched from any source")

    alt_score, alt_ts = _fetch_altcoin_season()
    if alt_score is not None:
        rows.append({
            "series_id":   "ALTCOIN_SEASON",
            "metric_name": "Altcoin Season Index",
            "value":       float(alt_score),
            "text_value":  "Altcoin Season" if alt_score >= 75 else ("Bitcoin Season" if alt_score <= 25 else "Neutral"),
            "source":      "Blockchaincenter",
            "category":    "crypto",
            "tier":        2,
            "recorded_at": alt_ts,
        })
    else:
        print("onchain: Altcoin Season not fetched")

    save_market_data(rows)
    print(f"onchain: saved {len(rows)} series")


if __name__ == "__main__":
    collect()
