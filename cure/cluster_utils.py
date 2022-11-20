import numpy as np


def distance(a: np.ndarray, b: np.ndarray):
    return np.linalg.norm(b - a)


def random_points(data: np.ndarray, n: int):
    maxi = len(data) - 1
    rng = np.random.default_rng()
    indices = rng.choice(maxi, n, replace=True)

    k = data.shape[1]
    selected = np.ndarray((n, k), dtype=float)
    for i, index in enumerate(indices):
        selected[i] = data[index]
    return selected


def random_points_distinct(data: np.ndarray, n: int):
    """Pick random sample of data points and also returns the remainder"""
    length = len(data)
    maxi = length - 1
    rng = np.random.default_rng()
    indices = rng.choice(maxi, n, replace=False).tolist()

    rem = length - n
    dim = data.shape[1]
    selected = np.ndarray((n, dim), dtype=float)
    remainder = np.ndarray((rem, dim), dtype=float)
    i = j = 0
    for l in range(length):
        if l in indices:
            selected[i] = data[l]
            i += 1
        else:
            remainder[j] = data[l]
            j += 1

    return selected, remainder


def point_product(points: np.ndarray):
    len = points.shape[0]
    for i in range(len):
        for j in range(i + 1, len):
            yield (points[i], points[j])


def average_distance(points: np.ndarray):
    """Calculate average distance for points."""
    points = points[:200]  # TODO better limit avg calculation
    pairs = point_product(points)
    distances = list(map(lambda p: distance(p[0], p[1]), pairs))
    return np.average(distances)
