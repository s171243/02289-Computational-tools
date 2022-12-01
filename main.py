import time,random, pickle
import pandas as pd
from tqdm import tqdm
from clustering import split_users
from cure import *
from data.data_loader import load_data_raw
from data.data_preprocessing import preprocess_df, extract_additional_user_features
from data.feature_categorization import U_features
from sandbox.recommender_system import  generate_utility_matrix_for_one_cluster, get_psedu_problem_difficulties_for_single_user, get_recommendation, get_psedu_problem_difficulties, split_data
from os.path import exists
random.seed(3)
np.random.seed(3)
start_time = time.time()


def log(msg):
    t = (time.time() - start_time) / 60
    print(f"[{t:.2f} min] {msg}")


# TODO just copy the code from cure.py here

def bind_labels_and_uuid(cluster_labels,sim_users):
    # Something is wrong here
    #Concatenate pd.DataFrame([clusters_labels

    cluster_partitions = [cluster_labels[cluster_labels==i] for i in range(cluster_labels.max()+1)]
    clusters = pd.DataFrame([np.hstack(cluster_partitions), np.hstack(sim_users)]).T
    #clusters = pd.DataFrame([cluster_labels, sim_users]).T
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
    RUN_ALL_SPLITS = True
    USE_USER_USER_SIMILARITY = False
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
        log("Getting clusters for all users...")
        cluster_labels, similarities, sim_users = get_clusters_and_similarity_matrix(X)

    #TODO Make it work for RUN_ALL_SPLITS True False, SPLIT true False
    #Remember: Splits divide data into dfs labels
    #and for each split we have clusters. Each split will therefore have cluster_labels, similarities and sim_users
    if RUN_ALL_SPLITS:
        # TODO update arguments such that it only uses df_u_split or df_u, and df_p_split or df_p
        log("Binding clusters labels and uuids for all splits")
        all_segment_clusters = [bind_labels_and_uuid(c_labels,s_users) for (c_labels,s_users) in zip(all_split_labels,all_split_sim_users)]
    else:
        clusters = bind_labels_and_uuid(cluster_labels,sim_users)

    if RUN_ALL_SPLITS: # Run all splits, and all clusters
        mean_errors = []
        for split_idx, clusters_ in enumerate(all_segment_clusters):
            if split_idx == 1:
                break
            else:
                # TODO update arguments such that it only uses df_u_split or df_u, and df_p_split or df_p
                df_u_split = dfs[split_idx]
                df_p_split = df_pr.loc[df_pr['uuid'].isin(df_u_split['uuid'])]
                similarities_ = all_split_similarities[split_idx]
                for cluster_idx in tqdm(range(len(similarities_)),desc="running cluster"):
                    mean_abs_error,errors,recommendation_difficulty_for_all_users, recommendation_idx_all = run_and_evaluate_recommender_system(clusters_, df_p_split, df_u_split,similarities_,cluster_idx,USE_USER_USER_SIMILARITY)
                    mean_errors.append(mean_abs_error)
                    with open('data/evaluation_results/eval_mean_errors.txt', 'a') as f:
                        f.write("segment: {}, cluster: {}, n_errors {}, mean_error {}\n".format(split_idx,cluster_idx,len(errors),mean_abs_error))
                    with open('data/evaluation_results/eval_errors.txt', 'a') as f:
                        f.write("segment: {}, cluster: {}, errors {}\n".format(split_idx,cluster_idx,errors))
        print("Mean absolute errors for the different splits {}".format(mean_errors))
    else:
        if SPLIT_USERS: #Split data and run the first cluster on the first split
            # TODO update arguments such that it only uses df_u_split or df_u, and df_p_split or df_p, should there exist a for loop iterating over cluster_idx?
            #Select the first data related to third split
            df_u_split = dfs[2]
            df_p_split = df_pr.loc[df_pr['uuid'].isin(df_u_split['uuid'])]
            #Select what cluster to evaluate
            cluster_idx = 2
            mean_abs_error, errors,recommendation_difficulty_for_all_users, recommendation_idx_all = run_and_evaluate_recommender_system(clusters, df_p_split, df_u_split,similarities,cluster_idx,USE_USER_USER_SIMILARITY)
            print(mean_abs_error)
        else:
            # TODO update arguments such that it only uses df_u_split or df_u, and df_p_split or df_p, what about idx?
            cluster_idx = 0
            mean_abs_error,errors,recommendation_difficulty_for_all_users, recommendation_idx_all = run_and_evaluate_recommender_system(clusters, df_pr, df_u,similarities,cluster_idx,USE_USER_USER_SIMILARITY)
    pass
    # TODO Generate utility matrix for each cluster - and save.

def run_and_evaluate_recommender_system(clusters, df_pr, df_u,user_user_similarities,cluster_id=0,use_user_user_similarity=False):
    # fname_uniq_prob, fname_M, fnameM_test,fname_df_u_sub_uuids = 'data/pickle_files/unique_prob_ids.pkl', 'data/pickle_files/M.pkl', 'data/pickle_files/M_test.pkl', 'data/pickle_files/df_u_sub_uuids.pkl'
    # files_exists = exists(fname_uniq_prob) and exists(fname_M) and exists(fnameM_test) and exists(fname_df_u_sub_uuids)
    # if files_exists:
    #     with open(fname_uniq_prob, 'rb') as file:
    #         P1_ids = pickle.load(file)
    #     with open(fnameM_test, 'rb') as file:
    #         M_test = pickle.load(file)
    #     with open(fname_M, 'rb') as file:
    #         M = pickle.load(file)
    #     with open(fname_df_u_sub_uuids, 'rb') as file:
    #         U1_ids = pickle.load(file)
    # else:
    #     M, M_test, U1_ids, P1_ids = generate_utility_matrix_for_one_cluster(clusters=clusters,df_u_full=df_u, df_pr_full=df_pr,cluster_id=cluster_id)
    #     if not files_exists:
    #         with open(fname_uniq_prob, 'wb') as file:
    #             pickle.dump(P1_ids,file)
    #         with open(fname_M, 'wb') as file:
    #             pickle.dump(M,file)
    #         with open(fnameM_test, 'wb') as file:
    #             pickle.dump(M_test,file)
    #         with open(fname_df_u_sub_uuids, 'wb') as file:
    #             pickle.dump(U1_ids,file)
    #OUTCOMMENT FOLLOWING line if you are using the files_exists logic to load rather than compute matrices
    M, M_test, U1_ids, P1_ids = generate_utility_matrix_for_one_cluster(clusters=clusters,df_u_full=df_u, df_pr_full=df_pr,cluster_id=cluster_id)

    cluster_user_user_similarity = user_user_similarities[cluster_id]
    difficulties_for_all_users, errors_all = get_psedu_problem_difficulties(M, M_test,cluster_user_user_similarity,use_user_user_similarity)
    errors = [item for sublist in errors_all for item in sublist if len(item) > 0]
    print("trying to calculate mean with error values; ", errors)
    mean_abs_error = np.mean(errors)
    recommendation_difficulty_for_all_users, recommendation_idx_all = get_recommendation(difficulties_for_all_users)
    print("Mean absolute error of difficulty was {} for cluster {} with {} errors".format(mean_abs_error,cluster_id,len(errors)))
    return mean_abs_error,errors,recommendation_difficulty_for_all_users, recommendation_idx_all

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
