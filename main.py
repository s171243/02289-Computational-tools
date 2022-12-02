import time

import pandas as pd
from sklearn.metrics import davies_bouldin_score
from tqdm import tqdm

from cure import *
from data_preprocessing import preprocess_df, extract_additional_user_features
from recommender_system import generate_utility_matrix_for_one_cluster, get_recommendation, \
    get_psedu_problem_difficulties

random.seed(3)
np.random.seed(3)
start_time = time.time()


def log(msg):
    t = (time.time() - start_time) / 60
    print(f"[{t:.2f} min] {msg}")


def load_data_raw():
    df_u = pd.read_csv('data/Info_UserData.csv')
    df_pr = pd.read_csv('data/Log_Problem.csv')
    df_ex = pd.read_csv('data/Info_Content.csv')
    return df_u, df_pr, df_ex


def bind_labels_and_uuid(cluster_labels, sim_users):
    cluster_partitions = [cluster_labels[cluster_labels == i] for i in range(cluster_labels.max() + 1)]
    clusters = pd.DataFrame([np.hstack(cluster_partitions), np.hstack(sim_users)]).T
    clusters.columns = ["labels", "uuid"]
    return clusters


def remove_problems_with_total_time_outliers(df_pr):
    limit = np.quantile(df_pr['total_sec_taken'], 0.98)
    new_df_pr = df_pr[df_pr['total_sec_taken'] < limit]
    return new_df_pr


def split_users_by_grade(df):
    split1 = df[df["user_grade"] < 4]
    split2 = df[df["user_grade"] == 5]
    split3 = df[df["user_grade"] == 6]
    split4 = df[df["user_grade"] == 7]
    split5 = df[df["user_grade"] > 7]
    return [split1, split2, split3, split4, split5], ["split1", "split2", "split3", "split4", "split5"]


class U_features():
    def __init__(self,
                 features=['user_grade', 'has_teacher_cnt', 'has_student_cnt', 'has_class_cnt',
                           "correct_percentage", "problems_attempted", "average_level", "max_level",
                           "average_hints", "avg_difficulty", "avg_learning_stage"],
                 features_to_be_time_encoded=['first_login_date_TW'], features_to_be_OR_encoded=[],
                 features_to_be_OH_encoded=['gender', 'user_city', 'is_self_coach'], features_meta=['uuid'],
                 features_to_be_scaled=['points', "correct_percentage", "time_spent", 'belongs_to_class_cnt',
                                        "badges_cnt"]):
        self.features = features
        self.features_to_be_time_encoded = features_to_be_time_encoded
        self.features_to_be_scaled = features_to_be_scaled
        self.features_to_be_OR_encoded = features_to_be_OR_encoded
        self.features_to_be_OH_encoded = features_to_be_OH_encoded
        self.features_meta = features_meta

def evaluate_clusterings(combined):
    for df, split, labels in combined:
        df = df.drop(columns=["uuid", "first_login_date_TW"])
        data = df.to_numpy()
        print(f"Split: {split} | Davies-Bouldin score: ", davies_bouldin_score(data, labels))
        centroids = [np.average(data[labels == k], axis=0) for k in range(labels.max() + 1)]

        def interesting_columns(x: np.ndarray):
            return list(filter(lambda pair: pair[1] > 0.07, zip(df.columns, x)))

        c0 = centroids[0]
        print(f"Centroid differences (for {len(centroids)} clusters):")
        for i in range(1, len(centroids)):
            c1 = centroids[i]
            diff = interesting_columns(c0 - c1)
            diff = sorted(diff, key=lambda p: p[1])[:3]
            log(diff)


def main():
    log("Loading data...")
    df_u, df_pr, df_c = load_data_raw()
    df_pr = remove_problems_with_total_time_outliers(df_pr)

    log("Feature extraction...")
    X = extract_additional_user_features(df_u, df_pr, df_c)

    log("Preprocessing...")
    user_features = U_features()
    X: pd.DataFrame = preprocess_df(df=X, o_features=user_features)

    USE_USER_USER_SIMILARITY = True
    all_split_labels = []
    all_split_similarities = []
    all_split_sim_users = []

    log("Splitting users...")
    dfs, labels = split_users_by_grade(X)
    del X
    for df, label in zip(dfs, labels):
        log("Getting clusters for {} users...".format(label))
        cluster_labels, similarities, sim_users = get_clusters_and_similarity_matrix(df)
        all_split_labels.append(cluster_labels)
        all_split_similarities.append(similarities)
        all_split_sim_users.append(sim_users)

    evaluate_clusterings(zip(dfs, labels, all_split_labels))

    # Remember: Splits divide data into dfs labels
    # and for each split we have clusters. Each split will therefore have cluster_labels, similarities and sim_users
    log("Binding clusters labels and uuids for all splits")
    all_segment_clusters = [bind_labels_and_uuid(c_labels, s_users) for (c_labels, s_users) in
                            zip(all_split_labels, all_split_sim_users)]

    # Run all splits, and all clusters
    mean_errors = []
    for split_idx, clusters_ in enumerate(all_segment_clusters):
        df_u_split = dfs[split_idx]
        df_p_split = df_pr.loc[df_pr['uuid'].isin(df_u_split['uuid'])]
        similarities_ = all_split_similarities[split_idx]
        for cluster_idx in tqdm(range(len(similarities_)), desc="running cluster"):
            mean_abs_error, errors, recommendation_difficulty_for_all_users, recommendation_idx_all, mean_difficulty = run_and_evaluate_recommender_system(
                clusters_, df_p_split, df_u_split, similarities_, cluster_idx, USE_USER_USER_SIMILARITY)
            mean_errors.append(mean_abs_error)
            with open('data/evaluation/eval_mean_errors_5splits.txt', 'a') as f:
                f.write(
                    "split_id: {},split_size: {}, cluster_id: {},cluster_u_size {}, n_errors: {}, mean_error: {}, mean_difficulty: {}, mean_recommendation_difficulty: {}\n".format(
                        split_idx, df_u_split.shape[0], cluster_idx, similarities_[cluster_idx].shape[0],
                        len(errors), np.round(mean_abs_error, 5), np.round(mean_difficulty, 5),
                        np.round(np.mean(recommendation_difficulty_for_all_users), 5)))
            with open('data/evaluation/eval_error_5splitss.txt', 'a') as f:
                f.write("split_id: {}, cluster_id: {}, errors {}\n".format(split_idx, cluster_idx, errors))
    print("Mean absolute errors for the different splits {}".format(mean_errors))



def run_and_evaluate_recommender_system(clusters, df_pr, df_u, user_user_similarities, cluster_id=0,
                                        use_user_user_similarity=False):
    M, M_test, U1_ids, P1_ids = generate_utility_matrix_for_one_cluster(clusters=clusters, df_u_full=df_u,
                                                                        df_pr_full=df_pr, cluster_id=cluster_id)

    cluster_user_user_similarity = user_user_similarities[cluster_id]

    difficulties_for_all_users, errors_all = get_psedu_problem_difficulties(M, M_test, cluster_user_user_similarity,
                                                                            use_user_user_similarity)
    errors = [item[0] for sublist in errors_all for item in sublist if len(item) > 0]
    mean_abs_error = np.mean(errors)
    recommendation_difficulty_for_all_users, recommendation_idx_all = get_recommendation(difficulties_for_all_users)
    num_users = clusters.loc[clusters['labels'] == cluster_id].shape[0]
    print("Mean absolute error of difficulty was {} for cluster {} with {} users and use_similarrity={}".format(
        mean_abs_error, cluster_id, num_users, str(use_user_user_similarity)))
    return mean_abs_error, errors, recommendation_difficulty_for_all_users, recommendation_idx_all, np.mean(
        difficulties_for_all_users[difficulties_for_all_users > 0])


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
