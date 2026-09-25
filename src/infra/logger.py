"""Pipeline logger — timestamped, human-readable, tees to stdout + file.

Every substantial step gets narrated so a collaborator reading pipeline.log
after a run can reconstruct exactly what happened without re-running.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class PipelineLogger:
    def __init__(self, log_path: Path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.log_path, "w", buffering=1)
        self.start_ts = datetime.now()

    @staticmethod
    def _ts() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _emit(self, line: str) -> None:
        self._fh.write(line + "\n")
        print(line, flush=True)

    def info(self, msg: str) -> None:
        self._emit(f"[{self._ts()}] {msg}")

    def kv(self, key: str, value: Any) -> None:
        if isinstance(value, (dict, list)):
            v = json.dumps(value, indent=2, ensure_ascii=False)
        else:
            v = str(value)
        self._emit(f"[{self._ts()}]   {key}: {v}")

    def banner(self, msg: str) -> None:
        bar = "=" * 78
        self._emit("")
        self._emit(bar)
        self._emit(f"[{self._ts()}] {msg}")
        self._emit(bar)

    def section(self, msg: str) -> None:
        self._emit("")
        self._emit(f"[{self._ts()}] --- {msg} ---")

    def multiline(self, label: str, text: str) -> None:
        self._emit(f"[{self._ts()}] {label}:")
        for line in text.splitlines():
            self._emit(f"    {line}")

    def round_header(self, round_idx: int) -> None:
        self.banner(f"GEPA ROUND {round_idx:02d}")

    def elapsed(self) -> str:
        delta = datetime.now() - self.start_ts
        s = int(delta.total_seconds())
        return f"{s // 3600}h{(s % 3600) // 60:02d}m{s % 60:02d}s"

    def close(self) -> None:
        self._emit("")
        self._emit(f"[{self._ts()}] === run finished, elapsed={self.elapsed()} ===")
        self._fh.close()
