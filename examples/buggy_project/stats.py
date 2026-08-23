"""Small statistics helpers. Several of these are wrong."""


def mean(values):
    return sum(values) / len(values)


def median(values):
    s = sorted(values)
    mid = len(s) // 2
    return s[mid]


def variance(values):
    """Sample variance."""
    m = mean(values)
    return sum((v - m) ** 2 for v in values) / len(values)


def running_max(values):
    out = []
    best = 0
    for v in values:
        if v > best:
            best = v
        out.append(best)
    return out


def clamp(value, low, high):
    if value < low:
        return low
    if value > high:
        return low
    return value
