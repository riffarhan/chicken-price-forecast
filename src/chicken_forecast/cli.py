"""Command-line entry point.

    chicken-forecast fetch      # download/refresh PIHPS, FRED, holiday calendars, weather
    chicken-forecast build      # raw daily -> clean weekly panel
    chicken-forecast train      # backtest, train final models, forecast, explain
    chicken-forecast ablation   # test which feature groups really improve accuracy
    chicken-forecast all        # fetch + build + train
"""

from __future__ import annotations

import argparse
import logging
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="chicken-forecast", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["fetch", "build", "train", "ablation", "all"])
    parser.add_argument("--force", action="store_true", help="re-download cached years too")
    parser.add_argument("--start-year", type=int, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    from chicken_forecast.config import PIHPS_START_YEAR

    if args.command in ("fetch", "all"):
        from chicken_forecast.data import calendar, fred, pihps, weather

        fred.download_all()
        calendar.download_all()
        weather.download_all()
        pihps.download_all(start_year=args.start_year or PIHPS_START_YEAR, force=args.force)
    if args.command in ("build", "all"):
        from chicken_forecast.data.build import build_weekly_panel

        build_weekly_panel()
    if args.command == "ablation":
        from chicken_forecast.ablation import run as run_ablation

        run_ablation()
    if args.command in ("train", "all"):
        from chicken_forecast.pipeline import run

        run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
