"""Memo for deterministic model calls made by the test battery.

The model has no sampling, so the same (tool, arguments) under the same checkpoint
configuration gives the same answer. The in-memory layer lives for one pytest session and
is always on for plain predictions. The disk layer is opt-in (LAYA_TEST_DISK_CACHE=1) and
keyed by a salt the caller builds from the laya version, checkpoint and LAYA_LOWMEM, so a
model change invalidates it; it is never used for the quality-metric tiers by default.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Callable


def make_key(tool: str, arguments: dict, salt: str = "") -> str:
    blob = json.dumps([salt, tool, arguments], sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


class PredictionCache:
    def __init__(self, disk_dir: Path | None = None):
        self._mem: dict[str, Any] = {}
        self._disk = Path(disk_dir) if disk_dir else None
        if self._disk:
            self._disk.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def _path(self, key: str) -> Path:
        return self._disk / f"{key}.json"

    def _read_disk(self, key: str):
        if not self._disk:
            return None
        try:
            return json.loads(self._path(key).read_text())
        except (OSError, ValueError):
            return None

    def get_or_compute(self, key: str, compute: Callable[[], Any]) -> Any:
        if key in self._mem:
            self.hits += 1
            return copy.deepcopy(self._mem[key])
        cached = self._read_disk(key)
        if cached is not None:
            self.hits += 1
            self._mem[key] = cached
            return copy.deepcopy(cached)
        self.misses += 1
        value = compute()
        self._mem[key] = copy.deepcopy(value)
        if self._disk:
            self._path(key).write_text(json.dumps(value, ensure_ascii=False))
        return value
