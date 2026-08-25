"""Contract tests for calc.stats."""

from __future__ import annotations

import pytest

from calc import mean, median, percentile, trimmed_mean
from calc.stats import EmptySampleError


def test_mean_of_integers() -> None:
    assert mean([1, 2, 3, 4]) == 2.5


def test_median_of_odd_sample() -> None:
    assert median([5, 1, 3]) == 3


def test_median_of_even_sample_averages_the_middle_pair() -> None:
    assert median([1, 2, 3, 4]) == 2.5


def test_median_ignores_input_order() -> None:
    assert median([10, 2, 8, 4]) == 6.0


def test_median_of_empty_sample_raises() -> None:
    with pytest.raises(EmptySampleError):
        median([])


def test_percentile_endpoints() -> None:
    assert percentile([1, 2, 3, 4], 0) == 1
    assert percentile([1, 2, 3, 4], 100) == 4


def test_percentile_interpolates() -> None:
    assert percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)


def test_trimmed_mean_drops_tails() -> None:
    assert trimmed_mean([0, 1, 2, 3, 4, 5, 6, 7, 8, 100], 0.1) == pytest.approx(4.5)
