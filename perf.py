"""Structured performance logger — writes JSONL for easy pandas analysis.

Usage:
    from perf import perf

    perf.stage("triage")
    with perf.timer("llm_call", model="gpt-5.2"):
        result = await llm(...)

    # At end of pipeline:
    perf.summary()       # pretty-print to terminal
    perf.save()          # writes runs/YYYYMMDD_HHMMSS.jsonl

Load in notebook:
    import pandas as pd
    df = pd.read_json("runs/20260226_143000.jsonl", lines=True)
    df.groupby("operation")["duration_ms"].describe()
"""

import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PerfEvent:
    timestamp: float
    elapsed_s: float
    stage: str
    operation: str
    duration_ms: float
    success: bool = True
    error: str | None = None
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "timestamp": self.timestamp,
            "elapsed_s": round(self.elapsed_s, 3),
            "stage": self.stage,
            "operation": self.operation,
            "duration_ms": round(self.duration_ms, 2),
            "success": self.success,
        }
        if self.error:
            d["error"] = self.error
        d.update(self.meta)
        return d


class PerfLogger:
    def __init__(self):
        self._t0: float = 0.0
        self._events: list[PerfEvent] = []
        self._current_stage: str = ""
        self._stage_starts: dict[str, float] = {}

    def start(self):
        self._t0 = time.time()

    def stage(self, name: str):
        """Mark entry into a pipeline stage."""
        now = time.time()
        # Close previous stage
        if self._current_stage and self._current_stage in self._stage_starts:
            dur = (now - self._stage_starts[self._current_stage]) * 1000
            self._events.append(PerfEvent(
                timestamp=now,
                elapsed_s=now - self._t0,
                stage=self._current_stage,
                operation="stage_end",
                duration_ms=dur,
            ))
        self._current_stage = name
        self._stage_starts[name] = now
        self._events.append(PerfEvent(
            timestamp=now,
            elapsed_s=now - self._t0,
            stage=name,
            operation="stage_start",
            duration_ms=0,
        ))

    def event(self, operation: str, duration_ms: float, success: bool = True,
              error: str | None = None, **meta):
        """Record a single timed event."""
        now = time.time()
        self._events.append(PerfEvent(
            timestamp=now,
            elapsed_s=now - self._t0,
            stage=self._current_stage,
            operation=operation,
            duration_ms=duration_ms,
            success=success,
            error=error,
            meta=meta,
        ))

    @contextmanager
    def timer(self, operation: str, **meta):
        """Context manager that times a block and records the event."""
        t = time.time()
        err = None
        ok = True
        try:
            yield
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            ok = False
            raise
        finally:
            dur = (time.time() - t) * 1000
            self.event(operation, dur, success=ok, error=err, **meta)

    async def atimer(self, operation: str, coro, **meta):
        """Await a coroutine and record timing."""
        t = time.time()
        err = None
        ok = True
        try:
            result = await coro
            return result
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            ok = False
            raise
        finally:
            dur = (time.time() - t) * 1000
            self.event(operation, dur, success=ok, error=err, **meta)

    def finish(self):
        """Close the final stage."""
        if self._current_stage and self._current_stage in self._stage_starts:
            now = time.time()
            dur = (now - self._stage_starts[self._current_stage]) * 1000
            self._events.append(PerfEvent(
                timestamp=now,
                elapsed_s=now - self._t0,
                stage=self._current_stage,
                operation="stage_end",
                duration_ms=dur,
            ))

    def summary(self):
        """Print a human-readable timing summary to terminal."""
        print(f"\n{'─' * 60}")
        print(f"  PERFORMANCE SUMMARY")
        print(f"{'─' * 60}")

        # Per-stage durations
        stages: dict[str, float] = {}
        for ev in self._events:
            if ev.operation == "stage_end":
                stages[ev.stage] = ev.duration_ms

        total = sum(stages.values())
        if total > 0:
            for name, dur in stages.items():
                pct = dur / total * 100
                print(f"  {name:<30s} {dur/1000:>7.1f}s  ({pct:>4.1f}%)")
            print(f"  {'─' * 50}")
            print(f"  {'TOTAL':<30s} {total/1000:>7.1f}s")

        # Per-operation stats
        print(f"\n  Operation Latencies (ms):")
        print(f"  {'operation':<24s} {'count':>5s} {'min':>8s} {'median':>8s} {'p95':>8s} {'max':>8s} {'total':>8s}")
        print(f"  {'─' * 77}")

        ops: dict[str, list[float]] = {}
        for ev in self._events:
            if ev.operation in ("stage_start", "stage_end"):
                continue
            ops.setdefault(ev.operation, []).append(ev.duration_ms)

        for op, durations in sorted(ops.items()):
            durations.sort()
            n = len(durations)
            mn = durations[0]
            mx = durations[-1]
            med = durations[n // 2]
            p95 = durations[int(n * 0.95)]
            tot = sum(durations)
            print(f"  {op:<24s} {n:>5d} {mn:>8.1f} {med:>8.1f} {p95:>8.1f} {mx:>8.1f} {tot:>8.0f}")

        # Error summary
        errors = [ev for ev in self._events if not ev.success]
        if errors:
            print(f"\n  Errors: {len(errors)}")
            for ev in errors[:5]:
                print(f"    [{ev.stage}] {ev.operation}: {ev.error}")

        print(f"{'─' * 60}")

    def save(self, directory: str = "runs") -> str:
        """Write all events as JSONL. Returns the file path."""
        Path(directory).mkdir(exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(self._t0))
        path = os.path.join(directory, f"{ts}.jsonl")
        with open(path, "w") as f:
            for ev in self._events:
                f.write(json.dumps(ev.to_dict()) + "\n")
        print(f"  Perf log saved: {path} ({len(self._events)} events)")
        return path

    @property
    def events(self) -> list[PerfEvent]:
        return list(self._events)


# Module-level singleton
perf = PerfLogger()
