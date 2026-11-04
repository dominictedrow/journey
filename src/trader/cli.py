"""Command line: ``trader import-data | data-report | compete | rebalance | rebalance-backtest
| paper-rebalance``.

Research and planning, plus ``paper-rebalance``: orders to Alpaca PAPER only (D-032).
No live order path exists.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from trader.competition import CompetitionConfig, run_competition
from trader.data.loaders import cache_path, import_csv, load_bars, snapshot_metadata
from trader.data.quality import quality_report
from trader.rebalance.cli import add_parsers as add_rebalance_parsers

DEFAULT_CONFIG = Path("configs/competition.toml")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="trader")
    sub = p.add_subparsers(dest="cmd", required=True)

    imp = sub.add_parser("import-data", help="validate a vendor CSV and freeze it as Parquet")
    imp.add_argument("csv", type=Path)
    imp.add_argument("--symbol", required=True)
    imp.add_argument("--cache-dir", type=Path, default=Path("../data/cache"))

    rep = sub.add_parser("data-report", help="write the data quality report")
    rep.add_argument("--cache-dir", type=Path, default=Path("../data/cache"))
    rep.add_argument("--out", type=Path, default=Path("reports/data_quality.md"))
    rep.add_argument("symbols", nargs="+")

    comp = sub.add_parser("compete", help="run the after-cost competition")
    comp.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    comp.add_argument(
        "--unseal-holdout", action="store_true",
        help="include the sealed holdout. Use once, for the final evaluation only.",
    )  # fmt: skip

    add_rebalance_parsers(sub)

    args = p.parse_args(argv)
    if hasattr(args, "func"):
        raise SystemExit(args.func(args))
    if args.cmd == "import-data":
        out = import_csv(args.csv, args.symbol, args.cache_dir)
        print(f"wrote {out}: {snapshot_metadata(out)}")
    elif args.cmd == "data-report":
        parts = ["# Data quality report\n"]
        for sym in args.symbols:
            meta = snapshot_metadata(cache_path(args.cache_dir, sym))
            bars = load_bars(args.cache_dir, sym)
            parts.append(quality_report(bars, sym, meta, equity_index=sym != "EURUSD"))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("\n".join(parts))
        print(f"wrote {args.out}")
    elif args.cmd == "compete":
        cfg = CompetitionConfig.from_toml(args.config)
        _, report = run_competition(cfg, unseal_holdout=args.unseal_holdout)
        print(f"report: {report}")


if __name__ == "__main__":
    main()
