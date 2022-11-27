import time

import pandas as pd
from clustering import split_users
from cure import *
from data.data_loader import load_data_raw
from data.data_preprocessing import preprocess_df, extract_additional_user_features
from data.feature_categorization import U_features
from sandbox.recommender_system import  generate_utility_matrix_for_one_cluster, get_psedu_problem_difficulties_for_single_user, get_recommendation, get_psedu_problem_difficulties, split_data
start_time = time.time()


def log(msg):
    t = (time.time() - start_time) / 60
    print(f"[{t:.2f} min] {msg}")


# TODO just copy the code from cure.py here

def bind_labels_and_uuid(cluster_labels,sim_users):
    clusters = pd.DataFrame([cluster_labels, sim_users]).T
    clusters.columns = ["labels", "uuid"]
    return clusters
def main():
    log("Loading data...")
    df_u, df_pr, df_c = load_data_raw(subset=False)

    log("Feature extraction...")
    X = extract_additional_user_features(df_u, df_pr, df_c)

    log("Preprocessing...")
    user_features = U_features()
    X: pd.DataFrame = preprocess_df(df=X, o_features=user_features)

    SPLIT_USERS = True
    RUN_ALL_SPLITS = False
    split_idx = 0
    if RUN_ALL_SPLITS:
        all_split_labels = []
        all_split_similarities = []
        all_split_sim_users = []

    if SPLIT_USERS:
        log("Splitting users...")
        dfs, labels = split_users(X)
        del X
        for df, label in zip(dfs, labels):
            if split_idx < 1 or RUN_ALL_SPLITS:
                # INSERT CLUSTERING AND OTHER STUFF HERE
                log("Getting clusters for {} users...".format(label))
                cluster_labels, similarities, sim_users = get_clusters_and_similarity_matrix(df)
                split_idx += 1
                if RUN_ALL_SPLITS:
                    all_split_labels.append(cluster_labels)
                    all_split_similarities.append(similarities)
                    all_split_sim_users.append(sim_users)
            else:
                break
    else:
        # INSERT CLUSTERING AND OTHER STUFF HERE
        log("Getting clusters for all users...")
        cluster_labels, similarities, sim_users = get_clusters_and_similarity_matrix(X)


    if RUN_ALL_SPLITS:
        all_segment_clusters = [bind_labels_and_uuid(c_labels,s_users) for (c_labels,s_users) in zip(all_split_labels,all_split_sim_users)]
    else:
        clusters = bind_labels_and_uuid(cluster_labels,sim_users)

    #
    #test_index = split_data(df_u, df_pr)
    if RUN_ALL_SPLITS:
        mean_errors = []
        for cluster_idx, clusters_ in enumerate(all_segment_clusters):
            df_u_split = dfs[cluster_idx]
            df_p_split = df_pr.loc[df_pr['uuid'].isin(df_u_split['uuid'])]
            test_index = split_data(df_u_split, df_p_split)
            mean_abs_error,recommendation_difficulty_for_all_users, recommendation_idx_all = run_and_evaluate_recommender_system(clusters_, df_pr, df_u, test_index,cluster_idx)
            mean_errors.append(mean_abs_error)
        print("Mean absolute errors for the different splits {}".format(mean_errors))
    else:
        if SPLIT_USERS:
            df_u_split = dfs[0]
            df_p_split = df_pr.loc[df_pr['uuid'].isin(df_u_split['uuid'])]
            test_index = split_data(df_u_split, df_p_split)
            mean_abs_error, recommendation_difficulty_for_all_users, recommendation_idx_all = run_and_evaluate_recommender_system(clusters, df_pr, df_u, test_index)
        else:
            test_index = split_data(df_u, df_pr)
            mean_abs_error,recommendation_difficulty_for_all_users, recommendation_idx_all = run_and_evaluate_recommender_system(clusters, df_pr, df_u, test_index)
    pass
    # TODO Generate utility matrix for each cluster - and save.


def run_and_evaluate_recommender_system(clusters, df_pr, df_u, test_index,cluster_id=0):
    M, M_test, U1_ids, P1_ids = generate_utility_matrix_for_one_cluster(clusters=clusters,
                                                                        df_u_full=df_u, df_pr_full=df_pr,
                                                                        cluster_id=cluster_id, test_index=test_index)
    difficulties_for_all_users, errors_all = get_psedu_problem_difficulties(M, M_test)
    errors = [item for sublist in errors_all for item in sublist]
    mean_abs_error = np.mean(errors)
    recommendation_difficulty_for_all_users, recommendation_idx_all = get_recommendation(difficulties_for_all_users)
    print("Mean absolute error of difficulty was {} for cluster {}".format(mean_abs_error,cluster_id))
    return mean_abs_error,recommendation_difficulty_for_all_users, recommendation_idx_all

def get_clusters_and_similarity_matrix(df: pd.DataFrame):
    log("CURE Classification...")
    data = df.drop(columns=["uuid"]).to_numpy()
    cure_repres = cure_representatives(data)
    cluster_labels = cure_classify(data, cure_repres)
    log("CURE Classification: done")
    n_clusters = cluster_labels.max() + 1
    partition = lambda k: data[cluster_labels == k]
    similarities = [pairwise_distances(partition(k)) for k in range(n_clusters)]
    sim_users = [df["uuid"][cluster_labels == k] for k in range(n_clusters)]
    log("Similarity matrices calculated")
    return cluster_labels, similarities, sim_users


if __name__ == '__main__':
    main()
