#!/usr/bin/env bash
# Weekly refresh: download the latest data, rebuild the panel, retrain, forecast.
# PIHPS publishes on working days, so run early Monday for the week that just ended:
#   0 6 * * 1 /path/to/chicken-price-forecast/scripts/run_weekly.sh >> /path/to/logs/cron.log 2>&1

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"
mkdir -p logs

if [ -x .venv/bin/chicken-forecast ]; then
    CLI=.venv/bin/chicken-forecast
else
    CLI=chicken-forecast
fi

echo "=== Chicken price forecast refresh: $(date) ==="
$CLI all
echo "=== Done: $(date) ==="
