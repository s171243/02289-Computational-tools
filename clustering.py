import json

import matplotlib.pyplot as plt
import pandas as pd
from pyod.models.ecod import ECOD
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score

from cure import *
from data.data_loader import load_data_raw
from data.data_preprocessing import preprocess_df, extract_additional_user_features
from data.feature_categorization import U_features


def david_bouldin(X, label=""):
    """
    cluster users and plot the elbow-graph to visualize most appropriate number of classes based on intertia
    :param X: preprocessed data
    :return:
    """
    bouldin = []
    max_clusters = 25

    # scaled inertia
    for i in range(2, max_clusters):
        kmeans = KMeans(n_clusters=i, random_state=0, max_iter=500).fit(X)
        # kmeans.labels_
        # kmeans.cluster_centers_
        bouldin.append(davies_bouldin_score(X, kmeans.labels_))

    plt.plot(list(range(2, max_clusters)), bouldin)
    plt.title("Bouldin score of kmeans: " + label)
    plt.xticks(list(range(0, max_clusters)))
    plt.xlabel("n_clusters")
    plt.ylabel("Inertia")
    plt.grid(True)
    plt.show()


def inertia_graph_kmeans(X, label=""):
    """
    cluster users and plot the elbow-graph to visualize most appropriate number of classes based on intertia
    :param X: preprocessed data
    :return:
    """
    inertias = []
    max_clusters = 25

    # scaled inertia
    for i in range(1, max_clusters):
        kmeans = KMeans(n_clusters=i, random_state=0, max_iter=500).fit(X)
        # kmeans.labels_
        # kmeans.cluster_centers_
        inertias.append(kmeans.inertia_)

    plt.plot(list(range(1, max_clusters)), inertias)
    plt.title("intertia of kmeans: " + label)
    plt.xticks(list(range(0, max_clusters)))
    plt.xlabel("n_clusters")
    plt.ylabel("Inertia")
    plt.grid(True)
    plt.show()


def scaled_inertia(X, label=""):
    """
    cluster users and plot the elbow-graph to visualize most appropriate number of classes based on intertia
    :param X: preprocessed data
    :return:
    """
    inertias = []
    max_clusters = 25

    # scaled inertia

    kmeans_initial = KMeans(n_clusters=1, random_state=0, max_iter=500).fit(X)
    initial_inertia = kmeans_initial.inertia_
    alpha = 0.02
    for i in range(1, max_clusters):
        kmeans = KMeans(n_clusters=i, random_state=0, max_iter=500).fit(X)
        # kmeans.labels_
        # kmeans.cluster_centers_
        inertias.append(kmeans.inertia_ / initial_inertia + i * alpha)

    plt.plot(list(range(1, max_clusters)), inertias)
    plt.title("Scaled inertia: " + label)
    plt.xlabel("n_clusters")
    plt.xticks(list(range(0, max_clusters)))
    plt.ylabel("Inertia")
    plt.grid(True)
    plt.show()


def visualize_with_PCA(X, optimal_clusters=3, label=""):
    """
    Visualize clusters with PCA, given the optimal amount of clusters found using the graph produced by the function user_clustering_kmeans()
    :param X:
    :param optimal_clusters:
    :return:
    """
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=0, max_iter=500).fit(X)
    y = kmeans.predict(X)
    labels = kmeans.labels_
    print(f"BOULDIN: {davies_bouldin_score(X, labels)}")
    fig = plt.figure(1, figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d", elev=15, azim=200)  # azim=110

    pca = PCA(n_components=3)
    X_reduced = pca.fit_transform(X)

    print(pca.components_, "\n", X.columns, "\n", pca.explained_variance_ratio_, pca.explained_variance_ratio_.cumsum())

    ax.scatter(
        X_reduced[:, 0],
        X_reduced[:, 1],
        X_reduced[:, 2],
        c=y,
        cmap=plt.cm.Set1,
        edgecolor="k",
        s=40,
    )

    ax.set_title("First three PCA directions: " + label)
    ax.set_xlabel("1st eigenvector")
    ax.w_xaxis.set_ticklabels([])
    ax.set_ylabel("2nd eigenvector")
    ax.w_yaxis.set_ticklabels([])
    ax.set_zlabel("3rd eigenvector")
    ax.w_zaxis.set_ticklabels([])
    plt.show()


def split_users(df):
    """
    split users based on user_grade
    :param df:
    :return:
    """
    split1 = df[df["user_grade"] < 4]
    split2 = df[df["user_grade"] == 5]
    split3 = df[df["user_grade"] == 6]
    split4 = df[df["user_grade"] == 7]
    split5 = df[df["user_grade"] > 7]
    # young = df[df["user_grade"] < 5]
    # medium = df[(df["user_grade"] >= 5) & (df["user_grade"] < 7)]
    # old = df[df["user_grade"] >= 7]
    return [split1,split2,split3,split4,split5], ["split1","split2","split3","split4","split5"]


def cluster_main():
    df_u, df_pr, df_c = load_data_raw(subset=False)
    print("data loaded")
    X = extract_additional_user_features(df_u, df_pr, df_c)
    print("Features extracted")
    user_features = U_features()
    X = preprocess_df(df=X, o_features=user_features)
    print("Data preprocessed")
    dfs, labels = split_users(X)
    print("Users split")
    SPLIT_USERS = False
    if SPLIT_USERS:
        for df, label in zip(dfs, labels):
            write_clusters(df, label=label)
            df = df.drop(['uuid'], axis=1)
            # plot_columns_of_df(X)
            # david_bouldin(df, label=label)
            inertia_graph_kmeans(df, label=label)
            scaled_inertia(df, label=label)
            visualize_with_PCA(df, label=label)
    else:

        df = X
        label = "all"
        # write_clusters(df, label)
        df = df.drop(['uuid'], axis=1)

        write_cure_cluster(df, X)
        return
        clf = ECOD()
        clf.fit(df)

        # get outlier scores
        y_train_scores = clf.decision_scores_  # raw outlier scores on the train data
        pd.DataFrame(np.asarray(y_train_scores)).to_csv("sample.csv")
        print(f"{y_train_scores=}")
        # plot_columns_of_df(X)
        david_bouldin(df, label=label)
        inertia_graph_kmeans(df, label=label)
        scaled_inertia(df, label=label)
        visualize_with_PCA(df, label=label)


def write_clusters(X, label=""):
    X_no_uuid = X.drop(['uuid'], axis=1)

    kmeans = KMeans(n_clusters=6, random_state=0, max_iter=500).fit(X_no_uuid)
    y = kmeans.predict(X_no_uuid)

    write = True
    if write:
        print(X.columns)
        X['cluster'] = pd.Series(y, index=X.index)

        users = X[["uuid", "cluster"]]
        users.to_csv(f"data/csv_files/clusters_{label}.csv")


def write_cure_cluster(df, X):
    data = df.to_numpy()
    repres = cure_representatives(data)
    labels = cure_classify(data, repres)

    n_clusters = labels.max() + 1
    partition = lambda k: data[labels == k]
    similarities = [pairwise_distances(partition(k)) for k in range(n_clusters)]
    sim_users = [X["uuid"][labels == k] for k in range(n_clusters)]

    i = 0
    for (sim, u) in zip(similarities, sim_users):
        with open(f"data/csv_files/clusters_cure_{i}.json", "w") as outfile:
            outfile.write(
                json.dumps({
                    "similarity": sim.tolist(),
                    "users": u.tolist()
                })
            )
        i += 1


#
if __name__ == "__main__":
    cluster_main()
    pass
