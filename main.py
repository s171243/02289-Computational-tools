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


    SPLIT_USERS = True
    if SPLIT_USERS:
        log("Splitting users...")
        dfs, labels = split_users(X)
        del X
        for df, label in zip(dfs, labels):
            # INSERT CLUSTERING AND OTHER STUFF HERE
            log("Getting clusters for {} users...".format(label))
            get_clusters_and_similarity_matrix(df)
            pass
    else:
        # INSERT CLUSTERING AND OTHER STUFF HERE
        get_clusters_and_similarity_matrix(X)
        pass



    pass
    #TODO add segmentation
    # M, U1_ids, P1_ids = generate_utility_matrix_for_one_cluster(clusters=clusters, df_u_full=df_u, df_pr_full=df_pr, cluster_id=cluster_id)
    #
    # #U[i:(cluster.shape[0] + i), :] = M
    # #i += cluster.shape[0]
    # user_idx = 1 #U1_ids.iloc[0]
    # difficulties_for_single_user = get_psedu_problem_difficulties_for_single_user(user_idx, M)
    #
    # difficulties_for_all_users = get_psedu_problem_difficulties(M)
    #
    # recommendation_difficulty_for_single_user, recommendation_idx_single = get_recommendation(difficulties_for_single_user)
    # recommendation_difficulty_for_all_users,recommendation_idx_all = get_recommendation(difficulties_for_all_users)


def get_clusters_and_similarity_matrix(X):
    log("CURE Classification...")
    data = X.drop(columns=["uuid"]).to_numpy()
    cure_repres = cure_representatives(data)
    labels = cure_classify(data, cure_repres)
    log("CURE Classification: done")
    n_clusters = labels.max() + 1
    partition = lambda k: data[labels == k]
    similarities = [pairwise_distances(partition(k)) for k in range(n_clusters)]
    sim_users = [X["uuid"][labels == k] for k in range(n_clusters)]
    log("Similarity matrices calculated")
    return

if __name__ == '__main__':
    main()
