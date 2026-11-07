import requests
from bs4 import BeautifulSoup
from datetime import datetime, timezone
from collectors.base import save_market_data

URL = "https://farside.co.uk/bitcoin-etf-flow/"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}


def collect():
    try:
        resp = requests.get(URL, headers=HEADERS, timeout=20)
        resp.raise_for_status()
    except Exception as e:
        print(f"etf_flows: fetch error: {e}")
        return

    soup = BeautifulSoup(resp.text, "html.parser")
    tables = soup.find_all("table")
    # The ETF flow data is in the second table (index 1)
    table = next((t for t in tables if len(t.find_all("tr")) > 10), None)
    if not table:
        print("etf_flows: no data table found")
        return

    # Identify which column is "Total"
    header_row = table.find("tr")
    headers = [th.get_text(strip=True).lower() for th in header_row.find_all(["th", "td"])]
    total_col = next((i for i, h in enumerate(headers) if h == "total"), -1)

    rows_data = []
    for tr in table.find_all("tr")[1:]:  # skip header
        tds = tr.find_all("td")
        if not tds:
            continue

        date_text = tds[0].get_text(strip=True)
        if not date_text or "total" in date_text.lower():
            continue

        # Try to parse date (e.g. "28 May 2026")
        recorded_at = None
        for fmt in ("%d %b %Y", "%m/%d/%Y", "%Y-%m-%d", "%d/%m/%Y"):
            try:
                recorded_at = datetime.strptime(date_text, fmt).replace(tzinfo=timezone.utc)
                break
            except ValueError:
                continue
        if recorded_at is None:
            continue

        # Use Total column if available, otherwise sum all ETF columns
        def parse_flow(text):
            text = text.strip().replace(",", "")
            # Parentheses mean negative: (192.4) -> -192.4
            if text.startswith("(") and text.endswith(")"):
                text = "-" + text[1:-1]
            if text in ("", "-", "—", "N/A"):
                return None
            try:
                return float(text)
            except ValueError:
                return None

        if total_col > 0 and total_col < len(tds):
            net_flow = parse_flow(tds[total_col].get_text(strip=True))
        else:
            net_flow = None
            total = 0.0
            found = False
            for td in tds[1:]:
                v = parse_flow(td.get_text(strip=True))
                if v is not None:
                    total += v
                    found = True
            if found:
                net_flow = total

        if net_flow is not None:
            rows_data.append({
                "series_id":   "BTC_ETF_FLOWS_NET",
                "metric_name": "BTC ETF Net Flows",
                "value":       net_flow,
                "text_value":  None,
                "source":      "Farside",
                "category":    "crypto",
                "tier":        2,
                "recorded_at": recorded_at,
            })

    if rows_data:
        # Only save the most recent row to avoid re-inserting history on every run
        rows_data.sort(key=lambda r: r["recorded_at"], reverse=True)
        save_market_data([rows_data[0]])
        print(f"etf_flows: saved latest flow ({rows_data[0]['recorded_at'].date()}: {rows_data[0]['value']}M)")
    else:
        print("etf_flows: no parseable rows found")


if __name__ == "__main__":
    collect()
