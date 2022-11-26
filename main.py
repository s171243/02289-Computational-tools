import time

from sklearn.metrics import davies_bouldin_score

from clustering import split_users
from cure import *
from data.data_loader import load_data_raw
from data.data_preprocessing import preprocess_df, extract_additional_user_features
from data.feature_categorization import U_features

start_time = time.time()


def log(msg):
    t = (time.time() - start_time) / 60
    print(f"[{t:.2f} min] {msg}")


# TODO just copy the code from cure.py here

def main():
    log("Loading data...")
    df_u, df_pr, df_c = load_data_raw(subset=False)

    log("Feature extraction...")
    X = extract_additional_user_features(df_u, df_pr, df_c)

    log("Preprocessing...")
    user_features = U_features()
    X = preprocess_df(df=X, o_features=user_features)

    log("Splitting users...")
    dfs, labels = split_users(X)
    SPLIT_USERS = False
    if SPLIT_USERS:
        for df, label in zip(dfs, labels):
            # INSERT CLUSTERING AND OTHER STUFF HERE
            pass
    else:
        # INSERT CLUSTERING AND OTHER STUFF HERE
        pass

    log("CURE Classification...")
    data = X.drop(columns=["uuid"]).to_numpy()
    cure_repres = cure_representatives(data)
    labels = cure_classify(data, cure_repres)
    log("CURE Classification done")

    dbi_cure = davies_bouldin_score(data, labels)
    log(f"davies_bouldin_score CURE: {dbi_cure}")

    log("Comparing with kmeans")
    kmeans = KMeans(n_clusters=5, random_state=0, max_iter=500).fit(data)
    dbi_kmeans = davies_bouldin_score(data, kmeans.labels_)
    log(f"davies_bouldin_score Kmeans: {dbi_kmeans}")

    n_clusters = labels.max() + 1
    partition = lambda k: data[labels == k]
    similarities = [pairwise_distances(partition(k)) for k in range(n_clusters)]
    # TODO users for similarity
    log("Similarity matrices calculated")


if __name__ == '__main__':
    main()
