import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics.pairwise import pairwise_distances

""" Sample size for initial clustering """
CURE_SAMPLE = lambda size: int(size / 5) if size < 1000 else int(size / 500)

""" Number of (calculated, random) representative points """
CURE_REPRES = lambda sample_len: (10, 5)  # (calculated, random)


def random_data_split(data: np.ndarray, n: int):
    maxi = len(data)
    indices = np.random.choice(maxi, n, replace=False)

    def inverze():
        mask = np.full(maxi, True)
        mask[indices] = False
        return mask

    return data[indices], data[inverze()]


def _cure_initial_clustering(sample: np.ndarray):
    """Initial clustering
    idea:
    - do a clustering
    - drop tiny clusters
    - calculate average distances in good-looking one
    - do the sample clustering with DBSCAN
    """

    def eps_calculation():
        n_clusters = 4  # magic number
        labels = KMeans(n_clusters=n_clusters, random_state=0, max_iter=500).fit(sample).labels_
        partition = lambda k: sample[labels == k]

        def cluster_averages():
            for k in range(labels.max() + 1):
                ds = np.triu(pairwise_distances(partition(k)))
                yield np.average(ds[ds != 0])

        averages = sorted(list(cluster_averages()))
        arbitrary_choice = averages[1]
        return arbitrary_choice

    # intentionally not using AgglomerativeClustering
    correction = 1.0 if len(sample) < 1000 else 0.5
    eps = correction * eps_calculation()
    labels = DBSCAN(eps=eps, min_samples=10).fit(sample)
    return labels.labels_


def cure_classify(x: np.ndarray, representatives: np.ndarray) -> np.ndarray:
    labels = np.ndarray(shape=(len(x),), dtype=int)

    rss = np.concatenate(representatives)
    indexed_rs = [
        partition
        for partition, rs in enumerate(representatives)
        for _ in rs
    ]

    dss = pairwise_distances(x, rss)
    for i in range(len(x)):
        labels[i] = indexed_rs[dss[i].argmin()]

    return labels


def cure_representatives(x: np.ndarray):
    # Checked initial clustering
    while True:  # retry mechanism
        try:
            sample_num = CURE_SAMPLE(len(x))
            sample, remainder = random_data_split(x, sample_num)
            sample_labels = _cure_initial_clustering(sample)
            if sample_labels.max() >= 1:  # sanity check
                break
            else:
                print("Inappropriate initial clusters, retrying")
        except ValueError:
            print("Failed EPS calculation, retrying")

    sample_partition = lambda k: sample[sample_labels == k]
    centroid = lambda x: np.average(x, axis=0)

    def calc_borderline(x: np.ndarray, desired: int):
        cent = centroid(x)
        # centroid <-> point distances
        cent_point_dist: list[tuple[np.ndarray, float]] = sorted(
            zip(x, pairwise_distances(x, [cent]).flatten()),
            key=lambda p: p[1]
        )

        # furthest points from centroid
        borderline: list[np.ndarray] = [cent_point_dist.pop()[0]]

        half = int(len(cent_point_dist) / 2)
        # an educated guess to reduce the number of points
        cent_point_dist = cent_point_dist[half:]

        while len(borderline) < desired:
            def repres_distance(point):  # point <-> borderline "distances"
                return pairwise_distances(borderline, [point]).sum()

            ds = list(map(lambda pair: (pair[0], repres_distance(pair[0])), cent_point_dist))
            ds = sorted(ds, key=lambda p: p[1])
            borderline.append(ds[-1][0])

        for vec in borderline:
            vec_towards_centroid = cent - vec
            yield vec + vec_towards_centroid / 4

    def representatives(x) -> np.ndarray:
        borderline_num, randoms_num = CURE_REPRES(len(x))
        randoms, _ = random_data_split(x, randoms_num)
        borderline = list(calc_borderline(x, borderline_num))
        return np.concatenate([randoms, borderline])

    rs = [
        representatives(sample_partition(k))
        for k in range(sample_labels.max() + 1)
    ]

    return rs
