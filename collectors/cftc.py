import csv
import io
import os
import requests
import zipfile
from datetime import datetime, timezone
from collectors.base import save_market_data

HEADERS = {"User-Agent": "MarketResearchBot/1.0"}

# Financial futures: S&P 500 (leveraged funds net = non-commercial equivalent)
FIN_TARGETS = {
    "E-MINI S&P 500": ("COT_SPX_NET", "COT S&P 500 Net", "macro", 1),
}

# Commodity futures: Gold
COMM_TARGETS = {
    "GOLD": ("COT_GOLD_NET", "COT Gold Net", "commodity", 1),
}


def _download_zip(url, tmp_path):
    resp = requests.get(url, headers=HEADERS, timeout=120, stream=True)
    resp.raise_for_status()
    with open(tmp_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=65536):
            f.write(chunk)


def _parse_zip(tmp_path, targets, long_col, short_col):
    latest = {}
    with zipfile.ZipFile(tmp_path) as zf:
        csv_name = next(
            (n for n in zf.namelist() if n.lower().endswith((".txt", ".csv"))),
            None,
        )
        if not csv_name:
            return latest
        with zf.open(csv_name) as raw:
            content = raw.read().decode("latin-1")

    reader = csv.DictReader(io.StringIO(content))
    for row in reader:
        market_name = row.get("Market_and_Exchange_Names", "").upper()
        for key, (series_id, label, category, tier) in targets.items():
            if key in market_name:
                date_str = row.get("Report_Date_as_YYYY-MM-DD", "")
                try:
                    report_date = datetime.strptime(date_str[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
                except Exception:
                    report_date = datetime.now(timezone.utc)
                try:
                    longs  = int(row.get(long_col,  "0").replace(",", "") or 0)
                    shorts = int(row.get(short_col, "0").replace(",", "") or 0)
                except ValueError:
                    continue
                net = longs - shorts
                prev = latest.get(series_id)
                if prev is None or report_date > prev["recorded_at"]:
                    latest[series_id] = {
                        "series_id":   series_id,
                        "metric_name": label,
                        "value":       float(net),
                        "text_value":  None,
                        "source":      "CFTC",
                        "category":    category,
                        "tier":        tier,
                        "recorded_at": report_date,
                    }
    return latest


def collect():
    year = datetime.now().year
    all_rows = {}

    # Financial futures: S&P 500
    fin_url = f"https://www.cftc.gov/files/dea/history/fut_fin_txt_{year}.zip"
    fin_tmp = "/tmp/cftc_fin.zip"
    try:
        _download_zip(fin_url, fin_tmp)
        all_rows.update(_parse_zip(
            fin_tmp, FIN_TARGETS,
            "Lev_Money_Positions_Long_All",
            "Lev_Money_Positions_Short_All",
        ))
    except Exception as e:
        print(f"cftc: financial futures error: {e}")
    finally:
        try:
            os.remove(fin_tmp)
        except OSError:
            pass

    # Commodity futures: Gold
    comm_url = f"https://www.cftc.gov/files/dea/history/fut_disagg_txt_{year}.zip"
    comm_tmp = "/tmp/cftc_comm.zip"
    try:
        _download_zip(comm_url, comm_tmp)
        all_rows.update(_parse_zip(
            comm_tmp, COMM_TARGETS,
            "M_Money_Positions_Long_All",
            "M_Money_Positions_Short_All",
        ))
    except Exception as e:
        print(f"cftc: commodity futures error: {e}")
    finally:
        try:
            os.remove(comm_tmp)
        except OSError:
            pass

    rows = list(all_rows.values())
    save_market_data(rows)
    print(f"cftc: saved {len(rows)} series")


if __name__ == "__main__":
    collect()
