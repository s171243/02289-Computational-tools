from typing import Any, List, Union

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN

from cure.cluster_utils import *
from cure.clustering import Clustering, ClusterPartition


def load_data(filename: str) -> np.ndarray:
    data = np.genfromtxt(
        filename,
        skip_header=1,
        skip_footer=1,
        dtype=float,
        delimiter=';'
    )
    data = np.delete(data, -1, axis=1)
    return data


def split_data(data: np.ndarray, denominator=3):
    """Pick random sample of data points"""
    maxi = len(data) - 1
    length = int(maxi / denominator)
    sample, remainder = random_points_distinct(data, length)
    return sample, remainder


def calculate_eps(sample: np.ndarray):
    # TODO select the best clustering here
    n_clusters = 4
    labelset = KMeans(n_clusters=n_clusters, max_iter=500).fit(sample)
    clustering = Clustering(dataset=sample, labels=labelset.labels_)

    def cluster_avgs(cluster: Clustering):
        for partition in clustering.partitions:
            yield average_distance(partition.points)

    avgs = sorted(list(cluster_avgs(clustering)))
    arbitrary_choice = avgs[1]
    return arbitrary_choice


def calc_borderline(partition: ClusterPartition, borderline_num: int):
    distances = sorted(
        list(map(
            lambda p: (p, distance(p, partition.centroid)),
            partition.points
        )),
        key=lambda p: p[1]
    )

    borderline: list[np.ndarray] = [distances.pop()[0]]

    half = int(len(distances) / 2)
    # an educated guess to reduce the number of points
    possible = distances[half:]

    while len(borderline) < borderline_num:
        def repres_distance(point):
            sum = 0.0
            for r in borderline:
                sum += distance(point, r)
            return sum / len(borderline)

        ds = list(map(lambda pair: (pair[0], repres_distance(pair[0])), possible))
        ds = sorted(ds, key=lambda p: p[1])
        borderline.append(ds[-1][0])

    def closer_to_centroid(vec):
        vec_towards_centroid = partition.centroid - vec
        return vec + vec_towards_centroid / 4

    for vec in borderline:
        yield closer_to_centroid(vec)


def calc_representatives(partition: ClusterPartition, desired: int) -> np.ndarray:
    borderline_num = int(desired * 3 / 4)
    random_num = desired - borderline_num

    randoms = random_points(partition.points, random_num)
    borderline = list(calc_borderline(partition, borderline_num))
    return np.array([*randoms, *borderline])


def plot_clustering(clustering: Clustering):
    colors = ["#4EACC5", "#FF9C34", "#4E9A06", "#f4dafa", "#5d7c74", "#6edc53"]
    for partition, color in zip(clustering.partitions, colors):
        points = partition.points
        plt.plot(points[:, 0], points[:, 1], 'o', markerfacecolor=color, markeredgecolor='k', markersize=6)

        if partition.centroid is not None:
            points = partition.centroid
            plt.plot(points[0], points[1], 'o', markerfacecolor='b', markeredgecolor='k', markersize=14)

        if partition.representatives is not None:
            points = partition.representatives
            plt.plot(points[:, 0], points[:, 1], 'x', markerfacecolor='k', markeredgecolor='r', markersize=14)

    # plt.title('clustering')
    plt.show()


def cure_initial_clustering(sample: np.ndarray) -> Clustering:
    """Initial clustering
    idea:
    - build a cluster (with centroids)
    - maybe drop tiny clusters
    - then leave points per cluster that is further from the centroid than
      constant * avg(99th percentile distance(centroid, point))
    """

    # intentionally not using AgglomerativeClustering
    eps = 0.8 * calculate_eps(sample)
    labelset = DBSCAN(eps=eps, min_samples=10).fit(sample)
    return Clustering(dataset=sample, labels=labelset.labels_)


def cure_classify_remainders(clustering: Clustering, remainder: np.ndarray) -> np.ndarray:
    labels = np.ndarray(shape=(len(remainder),), dtype=int)

    def repres_lookup() -> tuple[Any, int]:
        for i, partition in enumerate(clustering.partitions):
            for representative in partition.representatives:
                yield representative, i

    # Place to that cluster that’s representative is closer to the point
    lookup_table: list[tuple[Any, int]] = list(repres_lookup())
    for i, point in enumerate(remainder):
        closest: tuple[Any, int] = min(lookup_table, key=lambda pair: distance(pair[0], point))
        # append to cluster
        labels[i] = closest[1]
    return labels


def cure_clustering(data: np.ndarray) -> Clustering:
    # Tested initial clustering
    while True:  # retry mechanism
        sample, remainder = split_data(data)
        init_clustering = cure_initial_clustering(sample)
        # sanity check
        if len(init_clustering.partitions) > 1:
            break
        else:
            print("Not enough clusters, retrying")

    # Calculate representatives
    for partition in init_clustering.partitions:
        partition.representatives = calc_representatives(partition, desired=9)

    # Debug opportunity
    # plot_clustering(init_clustering)

    remainder_labels = cure_classify_remainders(init_clustering, remainder)
    return Clustering(
        dataset=np.concatenate((sample, remainder)),
        labels=np.concatenate((init_clustering.labels, remainder_labels))
    )


#def main():
#    data = load_data('data/cluster02_1000_2_3.dat')
#    clustering = cure_clustering(data)
#    plot_clustering(clustering)
    # TODO nice to have the Davies-Bouldin index
    # TODO exporting those clusters? A table with group_id and user_id

