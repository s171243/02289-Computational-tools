from cluster_utils import *
from cure import cure_clustering, plot_clustering
from evaluation import davies_bouldin_index, reality_check


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


def main():
    data = load_data('data/csv_files/cluster02_1000_2_3.dat')
    clustering = cure_clustering(data)
    plot_clustering(clustering)
    i1 = davies_bouldin_index(clustering)
    i2 = reality_check(clustering)
    print(i1, i2, i2 - i1)
    # TODO exporting those clusters? A table with group_id and user_id


if __name__ == '__main__':
    main()
