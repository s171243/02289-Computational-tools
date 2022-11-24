from functools import cached_property
from typing import List

from cluster_utils import *


class ClusterPartition:
    def __init__(self, points):
        self.points: np.ndarray = points
        self.representatives: np.ndarray = None

    @cached_property
    def centroid(self):
        sum = 0.0
        for point in self.points:
            sum += point
        return sum / len(self.points)


class Clustering:
    def __init__(self, dataset: np.ndarray, labels: np.ndarray):
        self.points: np.ndarray = dataset
        self.labels: np.ndarray = labels

    @property
    def partition_num(self) -> int:
        return self.labels.max() + 1

    @cached_property
    def partitions(self) -> List[ClusterPartition]:
        parts = map(lambda p: ClusterPartition(p), self._materialize_partitions())
        return list(parts)

    def _materialize_partitions(self):
        for k in range(self.partition_num):
            mask = self.labels == k
            yield self.points[mask]
