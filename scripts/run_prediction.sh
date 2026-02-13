#!/usr/bin/env bash
# Weekly chicken price prediction pipeline.
# Schedule with cron:
#   0 6 * * 1 /path/to/chicken-price-forecast/scripts/run_prediction.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "============================================"
echo "Chicken Price Prediction Pipeline"
echo "Starting at $(date)"
echo "============================================"

cd "$PROJECT_DIR"

# Activate virtual environment if present
if [ -f venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source venv/bin/activate
fi

python predict.py
STATUS=$?

if [ $STATUS -eq 0 ]; then
    echo "SUCCESS: Pipeline completed."
else
    echo "ERROR: Pipeline failed — check logs/ for details."
fi

exit $STATUS
