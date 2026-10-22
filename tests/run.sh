#!/usr/bin/env bash
# Laya test battery.
#   tests/run.sh             fast tier: contract + stubs, one smoke prediction (default)
#   tests/run.sh --model     fast + model tier (real predictions over the measured banks)
#   tests/run.sh --bench     fast + bench tier (full measurement of the question banks, ~10 min)
#   tests/run.sh --all       fast + model + bench (everything except the container-restarting tier)
#   tests/run.sh --run-slow  adds the autostart tests, which stop and restart the container
#   --write-reports          rewrite tests/reports/*_metrics.json (use with --bench)
# Extra args go to pytest. LAYA_TEST_DISK_CACHE=1 reuses predictions across runs (never for metrics).
set -euo pipefail
cd "$(dirname "$0")/.."
exec uv run --with "mcp>=2" --with pytest --with pytest-timeout --no-project pytest "$@"
