"""Mergeable heavy-hitter sketch built on Count-Min Sketch.

Guarantees (classic Count-Min, Cormode & Muthukrishnan):
  * estimate(key) >= true_count(key)                (never underestimates)
  * estimate(key) <= true_count(key) + epsilon * N  with probability >= 1 - delta
    where epsilon = e / width, delta = exp(-depth), N = total weight added.

The sketch state is a pure sum of updates, so:
  * insertion order does not affect the final counters;
  * sketches with identical (width, depth) merge by element-wise addition,
    which makes the result safe to ship across processes and combine.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
from array import array
from dataclasses import dataclass

_COUNTER_TYPECODE = "d"  # float64 counters
_COUNTER_BYTES = 8
_CANDIDATE_ENTRY_BYTES = 64  # conservative per-key bookkeeping estimate
_HASH_DOMAIN = b"heavy-hitter-sketch/v1"
_SERIALIZATION_VERSION = 1

STATUS_CONFIRMED = "confirmed"
STATUS_POSSIBLE = "possible"


@dataclass(frozen=True)
class KeyEstimate:
    """One ranked result row.

    estimate: Count-Min point estimate (upper bound on the true count).
    lower:    estimate - error_bound (lower bound, w.p. >= 1 - delta).
    upper:    same as estimate (Count-Min never underestimates).
    status:   "confirmed" if lower >= threshold (heavy even in the worst
              case), otherwise "possible" (estimate crosses the threshold
              but the error band does not rule out a smaller true count).
    """

    key: str
    estimate: float
    lower: float
    upper: float
    status: str


def _row_index(row: int, key_bytes: bytes, width: int) -> int:
    """Deterministic, process-independent hash of (row, key)."""
    digest = hashlib.blake2b(
        _HASH_DOMAIN + row.to_bytes(4, "little") + key_bytes,
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "little") % width


class HeavyHitterSketch:
    """Count-Min Sketch plus a bounded, deterministically pruned candidate set."""

    def __init__(self, width: int, depth: int, capacity: int):
        if width < 1:
            raise ValueError("width must be >= 1")
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.width = int(width)
        self.depth = int(depth)
        self.capacity = int(capacity)
        self._counters = array(_COUNTER_TYPECODE, [0.0]) * (self.width * self.depth)
        self._total = 0.0
        self._candidates: set[str] = set()

    @classmethod
    def from_budget(cls, budget_bytes: int, delta: float = 0.01) -> "HeavyHitterSketch":
        """Size the sketch from a byte budget.

        depth is derived from delta (delta = exp(-depth)); the budget is
        split 75% / 25% between counters and the candidate table. Tiny
        budgets are clamped to the minimum viable shape (width=2, depth=1,
        capacity=1), in which case actual usage may slightly exceed the
        budget while all error guarantees still hold.
        """
        if budget_bytes < 1:
            raise ValueError("budget_bytes must be >= 1")
        if not 0.0 < delta < 1.0:
            raise ValueError("delta must be in (0, 1)")
        depth = max(1, math.ceil(math.log(1.0 / delta)))
        candidate_budget = max(_CANDIDATE_ENTRY_BYTES, budget_bytes // 4)
        capacity = max(1, candidate_budget // _CANDIDATE_ENTRY_BYTES)
        counter_budget = max(depth * _COUNTER_BYTES, budget_bytes - candidate_budget)
        width = max(2, counter_budget // (depth * _COUNTER_BYTES))
        return cls(width=width, depth=depth, capacity=capacity)

    @property
    def epsilon(self) -> float:
        """Relative error factor: error_bound = epsilon * total_weight."""
        return math.e / self.width

    @property
    def delta(self) -> float:
        """Failure probability of the error bound."""
        return math.exp(-self.depth)

    @property
    def total(self) -> float:
        return self._total

    @property
    def error_bound(self) -> float:
        """Max over-estimation of any single key, w.p. >= 1 - delta."""
        return self.epsilon * self._total

    @property
    def memory_bytes(self) -> int:
        return (
            self.width * self.depth * _COUNTER_BYTES
            + self.capacity * _CANDIDATE_ENTRY_BYTES
        )

    def add(self, key: str, weight: float = 1.0) -> None:
        if not isinstance(key, str):
            raise TypeError("key must be a str")
        if not isinstance(weight, (int, float)) or isinstance(weight, bool):
            raise TypeError("weight must be a number")
        if weight < 0 or math.isnan(weight):
            raise ValueError("weight must be >= 0")
        key_bytes = key.encode("utf-8")
        for row in range(self.depth):
            col = _row_index(row, key_bytes, self.width)
            self._counters[row * self.width + col] += weight
        self._total += weight
        self._candidates.add(key)
        if len(self._candidates) > self.capacity:
            self._prune_candidates()

    def _prune_candidates(self) -> None:
        """Keep the top-`capacity` candidates, tie-broken by key for determinism."""
        ranked = sorted(
            ((self.estimate(key), key) for key in self._candidates),
            key=lambda item: (-item[0], item[1]),
        )
        self._candidates = {key for _, key in ranked[: self.capacity]}

    def estimate(self, key: str) -> float:
        """Point estimate of key's count; never below the true count."""
        key_bytes = key.encode("utf-8")
        best = math.inf
        for row in range(self.depth):
            col = _row_index(row, key_bytes, self.width)
            value = self._counters[row * self.width + col]
            if value < best:
                best = value
        return 0.0 if best is math.inf else best

    def topk(self, k: int, threshold: float | None = None) -> list[KeyEstimate]:
        """Ranked candidates with per-key error band and heavy-hitter status.

        threshold defaults to error_bound: a key is "confirmed" when its
        worst-case count (estimate - error_bound) still clears the bar.
        """
        if k < 0:
            raise ValueError("k must be >= 0")
        if threshold is None:
            threshold = self.error_bound
        err = self.error_bound
        ranked = sorted(
            ((self.estimate(key), key) for key in self._candidates),
            key=lambda item: (-item[0], item[1]),
        )
        results = []
        for est, key in ranked[:k]:
            lower = max(0.0, est - err)
            status = STATUS_CONFIRMED if lower >= threshold else STATUS_POSSIBLE
            results.append(
                KeyEstimate(key=key, estimate=est, lower=lower, upper=est, status=status)
            )
        return results

    def heavy_hitters(self, phi: float) -> list[KeyEstimate]:
        """All candidates whose estimate reaches phi * total_weight."""
        if not 0.0 < phi <= 1.0:
            raise ValueError("phi must be in (0, 1]")
        threshold = phi * self._total
        return [
            item
            for item in self.topk(len(self._candidates), threshold=threshold)
            if item.estimate >= threshold
        ]

    def merge(self, other: "HeavyHitterSketch") -> "HeavyHitterSketch":
        """In-place merge; other must have identical (width, depth)."""
        if not isinstance(other, HeavyHitterSketch):
            raise TypeError("can only merge another HeavyHitterSketch")
        if self.width != other.width or self.depth != other.depth:
            raise ValueError("sketch shapes differ; cannot merge")
        self._counters = array(
            _COUNTER_TYPECODE,
            (a + b for a, b in zip(self._counters, other._counters)),
        )
        self._total += other._total
        self._candidates |= other._candidates
        if len(self._candidates) > self.capacity:
            self._prune_candidates()
        return self

    @classmethod
    def merged(cls, sketches) -> "HeavyHitterSketch":
        """Merge an iterable of sketches into a new one."""
        sketches = list(sketches)
        if not sketches:
            raise ValueError("nothing to merge")
        result = cls.loads(sketches[0].dumps())
        for sketch in sketches[1:]:
            result.merge(sketch)
        return result

    def dumps(self) -> bytes:
        """Deterministic JSON serialization safe to ship across processes."""
        payload = {
            "version": _SERIALIZATION_VERSION,
            "width": self.width,
            "depth": self.depth,
            "capacity": self.capacity,
            "total": self._total,
            "counters": base64.b64encode(self._counters.tobytes()).decode("ascii"),
            "candidates": sorted(self._candidates),
        }
        return json.dumps(payload, sort_keys=True).encode("utf-8")

    @classmethod
    def loads(cls, data: bytes | str) -> "HeavyHitterSketch":
        if isinstance(data, str):
            data = data.encode("utf-8")
        payload = json.loads(data.decode("utf-8"))
        if payload.get("version") != _SERIALIZATION_VERSION:
            raise ValueError("unsupported serialization version")
        sketch = cls(
            width=payload["width"],
            depth=payload["depth"],
            capacity=payload["capacity"],
        )
        raw = base64.b64decode(payload["counters"])
        counters = array(_COUNTER_TYPECODE)
        counters.frombytes(raw)
        if len(counters) != sketch.width * sketch.depth:
            raise ValueError("corrupt counter payload")
        sketch._counters = counters
        sketch._total = float(payload["total"])
        sketch._candidates = set(payload["candidates"])
        return sketch

    def __repr__(self) -> str:
        return (
            f"HeavyHitterSketch(width={self.width}, depth={self.depth}, "
            f"capacity={self.capacity}, total={self._total})"
        )
