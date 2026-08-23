import pytest

from stats import clamp, mean, median, running_max, variance


def test_mean():
    assert mean([1, 2, 3, 4]) == 2.5


def test_median_odd():
    assert median([3, 1, 2]) == 2


def test_median_even():
    assert median([4, 1, 3, 2]) == 2.5


def test_variance_is_sample_variance():
    assert variance([2, 4, 4, 4, 5, 5, 7, 9]) == pytest.approx(4.5714286)


def test_running_max_handles_negatives():
    assert running_max([-5, -3, -8]) == [-5, -3, -3]


def test_clamp_above_range():
    assert clamp(10, 0, 5) == 5


def test_clamp_inside_range():
    assert clamp(3, 0, 5) == 3
