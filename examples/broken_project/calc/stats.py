"""Descriptive statistics over sequences of floats."""

from __future__ import annotations

from collections.abc import Sequence


class EmptySampleError(ValueError):
    """Raised when a statistic is requested for an empty sample."""


def mean(values: Sequence[float]) -> float:
    """Arithmetic mean."""
    if not values:
        raise EmptySampleError("mean of an empty sample")
    return sum(values) / len(values)


def median(values: Sequence[float]) -> float:
    """Middle value of the sample.

    For an even-sized sample the median is the mean of the two central values.
    """
    if not values:
        raise EmptySampleError("median of an empty sample")
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    return ordered[midpoint]


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile, ``q`` in [0, 100]."""
    if not values:
        raise EmptySampleError("percentile of an empty sample")
    if not 0.0 <= q <= 100.0:
        raise ValueError(f"q must be between 0 and 100, got {q}")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * (q / 100.0)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def trimmed_mean(values: Sequence[float], proportion: float = 0.1) -> float:
    """Mean after discarding ``proportion`` of the sample from each tail."""
    if not values:
        raise EmptySampleError("trimmed mean of an empty sample")
    if not 0.0 <= proportion < 0.5:
        raise ValueError(f"proportion must be in [0, 0.5), got {proportion}")
    ordered = sorted(values)
    cut = int(len(ordered) * proportion)
    kept = ordered[cut : len(ordered) - cut] or ordered
    return sum(kept) / len(kept)
