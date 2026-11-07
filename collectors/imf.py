import requests
from datetime import datetime, timezone
from collectors.base import save_market_data

BASE_URL = "https://www.imf.org/external/datamapper/api/v1"
HEADERS = {"User-Agent": "MarketResearchBot/1.0"}

# IMF indicator -> prefix for series_id
INDICATORS = {
    "NGDP_RPCH": "IMF_GDP",  # Real GDP growth %
    "PCPIPCH":   "IMF_CPI",  # CPI inflation %
}

# IMF country code -> our suffix
COUNTRIES = {
    "USA": "US",
    "EUQ": "EU",
    "JPN": "JP",
    "CHN": "CN",
    "IND": "IN",
    "BRA": "BR",
    "MEX": "MX",
    "KOR": "KR",
    "AUS": "AU",
}

COUNTRY_LABELS = {
    "US": "United States", "EU": "Eurozone", "JP": "Japan",
    "CN": "China",        "IN": "India",    "BR": "Brazil",
    "MX": "Mexico",       "KR": "South Korea", "AU": "Australia",
}


def collect():
    rows = []
    now = datetime.now(timezone.utc)

    for indicator, id_prefix in INDICATORS.items():
        label_prefix = "IMF GDP" if id_prefix == "IMF_GDP" else "IMF CPI"
        try:
            resp = requests.get(
                f"{BASE_URL}/{indicator}/{'/'.join(COUNTRIES.keys())}",
                headers=HEADERS,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"imf: {indicator} fetch error: {e}")
            continue

        values_map = data.get("values", {}).get(indicator, {})
        for imf_code, our_code in COUNTRIES.items():
            country_vals = values_map.get(imf_code, {})
            if not country_vals:
                continue
            # Pick most recent year with a real value
            latest_year = max(
                (yr for yr, v in country_vals.items() if v is not None),
                default=None,
                key=lambda y: int(y),
            )
            if latest_year is None:
                continue
            val = country_vals[latest_year]
            if val is None:
                continue

            series_id = f"{id_prefix}_{our_code}"
            label = f"{label_prefix} — {COUNTRY_LABELS[our_code]} ({latest_year})"
            rows.append({
                "series_id":   series_id,
                "metric_name": label,
                "value":       float(val),
                "text_value":  None,
                "source":      "IMF DataMapper",
                "category":    "macro",
                "tier":        2,
                "recorded_at": datetime(int(latest_year), 1, 1, tzinfo=timezone.utc),
            })

    save_market_data(rows)
    print(f"imf: saved {len(rows)} series")


if __name__ == "__main__":
    collect()
