import requests
from datetime import datetime, timezone
from collectors.base import save_market_data
from config import settings

API_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"

BLS_SERIES = {
    "CUUR0000SA0":  ("CPI_YOY",  "CPI YoY %",        "macro", 1),
    "WPUFD49104":   ("PPI_YOY",  "PPI YoY %",        "macro", 1),
    "CES0000000001":("PAYROLLS", "Nonfarm Payrolls",  "macro", 1),
}


def collect():
    if not settings.BLS_API_KEY:
        print("BLS: no API key, skipping")
        return

    try:
        resp = requests.post(API_URL, json={
            "seriesid":        list(BLS_SERIES.keys()),
            "registrationkey": settings.BLS_API_KEY,
            "calculations":    True,
        }, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"BLS fetch error: {e}")
        return

    rows = []
    for series in data.get("Results", {}).get("series", []):
        sid = series["seriesID"]
        if sid not in BLS_SERIES:
            continue
        series_id, label, category, tier = BLS_SERIES[sid]
        datum_list = series.get("data", [])
        if not datum_list:
            continue

        # BLS returns newest first
        datum = datum_list[0]
        try:
            raw_value = float(datum["value"])
        except (ValueError, KeyError):
            continue

        # Compute YoY % for CPI and PPI; for payrolls store raw (thousands)
        if series_id in ("CPI_YOY", "PPI_YOY"):
            calcs = datum.get("calculations", {})
            pct_changes = calcs.get("pct_changes", {})
            yoy = pct_changes.get("12")
            if yoy is None:
                continue
            value = float(yoy)
        else:
            value = raw_value

        year = int(datum["year"])
        period = datum.get("period", "M01")
        month = int(period.replace("M", "")) if period.startswith("M") else 1
        recorded_at = datetime(year, month, 1, tzinfo=timezone.utc)

        rows.append({
            "series_id":   series_id,
            "metric_name": label,
            "value":       value,
            "text_value":  None,
            "source":      "BLS",
            "category":    category,
            "tier":        tier,
            "recorded_at": recorded_at,
        })

    save_market_data(rows)
    print(f"BLS: saved {len(rows)} series")


if __name__ == "__main__":
    collect()
