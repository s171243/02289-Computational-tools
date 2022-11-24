import numpy as np
import sklearn

from cluster_utils import distance
from clustering import Clustering


def clustering_adapter(data: np.ndarray, labelset: np.ndarray) -> Clustering:
    """Convert a numpy array to a Clustering object."""
    return Clustering(data, labelset)


def average_radius(clustering: Clustering):
    for part in clustering.partitions:
        # ri: average radius
        # C: partition
        # c: point
        centroid = part.centroid
        sum = 0.0
        for point in part.points:
            sum += distance(centroid, point)
        ri = 1 / len(part) * sum
        yield ri


def davies_bouldin_index(clustering: Clustering) -> float:
    """Calculate the Davies-Bouldin index for a clustering.

    The Davies-Bouldin index is a measure of how well a clustering
    separates the data. The lower the index, the better the clustering.
    """

    radiuses = list(average_radius(clustering))
    k = clustering.partition_num
    sum = 0.0
    for i in range(k):
        def inner():
            for j in range(k):
                if i == j:
                    continue
                Ci = clustering.partitions[i]
                Cj = clustering.partitions[j]
                dij = distance(Ci.centroid, Cj.centroid)
                yield (radiuses[i] + radiuses[j]) / dij

        sum += max(inner())
    return 1 / k * sum


def reality_check(clustering: Clustering):
    return sklearn.metrics.davies_bouldin_score(
        X=clustering.points,
        labels=clustering.labels,
    )
