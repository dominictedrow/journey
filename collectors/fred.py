import time
import requests
from datetime import datetime, timezone
from collectors.base import save_market_data
from config import settings

BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

FRED_SERIES = {
    "T10Y2Y":       ("2s/10s Yield Curve",     "credit",    1),
    "BAMLH0A0HYM2": ("HY Credit Spread (OAS)", "credit",    1),
    "M2SL":         ("US M2 Money Supply",     "macro",     1),
    "DGS10":        ("10Y Treasury Yield",     "bond",      1),
    "DFII10":       ("10Y TIPS Real Yield",    "bond",      1),
    "DEXJPUS":      ("USD/JPY",                "fx",        1),
    "DEXUSAL":      ("AUD/USD",                "fx",        1),
    "WALCL":        ("Fed Balance Sheet",      "macro",     1),
}


def collect():
    if not settings.FRED_API_KEY:
        print("FRED: no API key, skipping")
        return

    rows = []
    for series_id, (label, category, tier) in FRED_SERIES.items():
        try:
            resp = requests.get(BASE_URL, params={
                "series_id":  series_id,
                "api_key":    settings.FRED_API_KEY,
                "file_type":  "json",
                "sort_order": "desc",
                "limit":      10,
            }, timeout=15)
            resp.raise_for_status()
            obs_list = resp.json().get("observations", [])
            for obs in obs_list:
                if obs.get("value") == ".":
                    continue
                try:
                    val = float(obs["value"])
                except (ValueError, KeyError):
                    continue
                recorded_at = datetime.strptime(obs["date"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
                rows.append({
                    "series_id":    series_id,
                    "metric_name":  label,
                    "value":        val,
                    "text_value":   None,
                    "source":       "FRED",
                    "category":     category,
                    "tier":         tier,
                    "recorded_at":  recorded_at,
                })
                break  # only the most recent valid observation
        except Exception as e:
            print(f"FRED {series_id} error: {e}")
        time.sleep(0.5)  # stay under FRED's rate limit (120 req/min)

    save_market_data(rows)
    print(f"FRED: saved {len(rows)} series")


if __name__ == "__main__":
    collect()
