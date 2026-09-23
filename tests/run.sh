#!/usr/bin/env bash
# Laya test battery. Default run is fast and read-only; --run-slow restarts the container.
set -euo pipefail
cd "$(dirname "$0")/.."
exec uv run --with "mcp>=2" --with pytest --no-project pytest "$@"
