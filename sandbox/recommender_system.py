import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from clustering import user_clustering_kmeans, visualize_with_PCA
from data.data_loader import load_data_raw
from data.data_preprocessing import extract_additional_user_features, preprocess_df
from data.feature_categorization import U_features
from collections import defaultdict

def get_user_segments(users,no_implementation = True, n_segments = 10):
    """
    use segmentation implementation to extract user segments
    :return:
    """
    if no_implementation:

        users_pr_segment = len(users // n_segments)
        users_in_segments = [users[i:i+users_pr_segment] for i in range(n_segments)]
        return users_in_segments

def get_clusters():
    """
    Purpose: get user-clusters. Pre-compute and load or generate on the fly.
    :return: clusters DataFrame with uuid (user_id) and cluster_id as columns.
    """
    clusters = pd.read_csv(r'data/csv_files/clusters.csv')
    return clusters

def construct_util_matrix(users,problems):
    unique_problems = problems['upid'].unique()
    M = np.empty((len(users),len(unique_problems)))
    for i, userID in enumerate(tqdm(users['uuid'],desc="Func: construct_util_matrix()")):
        user_problems = problems.loc[problems['uuid'] == userID]
        if not user_problems.empty:
            #TODO handle outliers. Max time should probably not be defined like this or rather some outliers should be filtered away first.
            max_time = max(user_problems['total_sec_taken'])
            user_prob_idx = []
            user_difficulties = []
            for j in user_problems.index.to_list():
                problem_ID = user_problems.loc[[j],'upid'].values[0]
                problem_index = np.where(unique_problems == problem_ID)[0][0]
                user_prob_idx.append(problem_index)
                time = user_problems.loc[[j],'total_sec_taken'].values[0]
                is_correct = user_problems.loc[[j],'is_correct'].values[0]
                user_difficulties.append(get_difficulty(time,max_time,is_correct))
                M[i,problem_index] = user_difficulties[-1]
            #standardize difficulty values between [0,1]
            min_ = np.min(user_difficulties)
            max_ = np.max(user_difficulties)
            M[i,user_prob_idx] = (M[i,user_prob_idx] - min_) / (max_)

        else:
            #M = np.delete(M, i, axis=0)
            pass
    return M,unique_problems

def get_difficulty(time,max_time,is_correct,alpha=0.8):
    """
    :param time: Time (sec) spent on a given problem for a given student
    :param max_time: Maximum time (sec) a given student has used on any problem
    :param is_correct: boolean if the student solved the given problem (1) or not (0)
    :param alpha: time importance weight
    :defined in func beta: correctness importance weights (Note weights should sum to one)
    :return: Relative difficulty of a problem for a given student
    """
    beta = 1-alpha
    difficulty = 1-((max_time - time) / max_time)*alpha - beta*is_correct
    return difficulty


def get_utility_matrix_shape(clusters,df_pr,subset=False):
    if subset:
        problems = defaultdict(list)
        for userID in tqdm(clusters['uuid'],desc="Func: get_utility_matrix_shape()"):
            user_info = df_pr.loc[df_pr['uuid'] == userID]
            # for problem in user_info['upid']:
            if user_info['upid'].shape[0]:
                problems[str(user_info['upid'])] = 1
        sub_problems = df_pr['upid'].isin(problems)
        df_pr_sub = df_pr.loc[sub_problems]
        return clusters.shape[0], len(df_pr_sub['upid'].unique())
    else:
        return clusters.shape[0], len(df_pr['upid'].unique())


def generate_utility_matrix_for_one_cluster(clusters,cluster_id,df_u_full,df_pr_full):
    cluster = clusters.loc[clusters['cluster'] == cluster_id]
    sub_users = df_u_full['uuid'].isin(cluster['uuid'].to_list())
    df_u_sub = df_u_full.loc[sub_users]
    #TODO make sure that the faster implementation is correct
    #problems = []
    # for userID in tqdm(df_u_sub['uuid'],desc="Func: generate_utility_matrix_for_one_cluster()"):
    #     user_info = df_pr_full.loc[df_pr_full['uuid'] == userID]
    #     for problem in user_info['upid']:
    #         problems.append(problem)
    # sub_problems = df_pr_full['upid'].isin(problems)
    # df_pr_sub = df_pr_full.loc[sub_problems]
    df_pr_sub = df_pr_full.loc[df_pr_full['uuid'].isin(df_u_sub['uuid'])]
    M, unique_prob_ids = construct_util_matrix(df_u_sub, df_pr_sub)
    return M, df_u_sub['uuid'], unique_prob_ids


def get_psedu_problem_difficulties(M):
    # Now that we have a utility matrix, we need to fill all empty entries
    recommendations = np.zeros_like(M)  # Create copy of utility matrix, so that 'predictions' are not used when aggregating
    for user in tqdm(range(np.shape(M)[0]),desc="get_psedu_problem_difficulties()"):
        recommendations[user, :] = get_psedu_problem_difficulties_for_single_user(user, M)
    return recommendations
def get_psedu_problem_difficulties_for_single_user(user_idx, M):
    # Now that we have a utility matrix, we need to fill all empty entries
    #M2 = np.copy(M)  # Create copy of utility matrix, so that 'predictions' are not used when aggregating
    n_user = np.shape(M)[0]
    n_problems = np.shape(M)[1]
    user_recommendations = np.zeros(n_problems)
    unsolved_problems = np.argwhere(M[user_idx, :] == 0)
    relevant_user_ids = list(range(0,user_idx))+list(range(user_idx+1,n_user))
    relevant_M = M[relevant_user_ids, unsolved_problems]
    user_recommendations[unsolved_problems] = np.reshape(np.sum(relevant_M, axis=1) / np.sum(relevant_M != 0, axis=1),(-1,1))
    return user_recommendations

def get_recommendation(difficulty_matrix,quantile=0.80,recommendations_to_return = 2):
    if len(difficulty_matrix.shape) == 1:
        difficulty_matrix = np.reshape(difficulty_matrix,(1,-1))

    n_problems = np.min(np.sum(~np.isnan(difficulty_matrix),axis=1))
    sorted_indices = np.argsort(difficulty_matrix,axis=1)

    quantile_idx = int(n_problems*quantile)

    #indices of problems being recommended (based on the initial ordering in difficulty_matrix)
    problem_indices = sorted_indices[:,quantile_idx:quantile_idx+recommendations_to_return]

    #test = difficulty_matrix[:,problem_indices]
    difficulties_of_recommendations = [difficulty_matrix[user_idx, prob_idx] for (user_idx, prob_idx) in enumerate(problem_indices)]
    pass
    return difficulties_of_recommendations, problem_indices


if __name__ == "__main__":
    sub = True
    df_u, df_pr, df_c = load_data_raw(subset=sub)

    print("data loaded")
    # X = extract_additional_user_features(df_u, df_pr, df_c)
    # user_features = U_features()
    # X = preprocess_df(df=X, o_features=user_features)
    # X = X.drop(['uuid'], axis=1)
    # user_clustering_kmeans(X)
    # visualize_with_PCA(X)
    clusters = get_clusters()
    #rows,columns = get_utility_matrix_shape(clusters,df_pr,subset=sub)
    #U = np.empty((rows,columns))


    #TODO Generate utility matrix for each cluster - and save.
    i = 0
    cluster_id = 3
    M, U1_ids, P1_ids = generate_utility_matrix_for_one_cluster(clusters=clusters, df_u_full=df_u, df_pr_full=df_pr, cluster_id=cluster_id)

    #U[i:(cluster.shape[0] + i), :] = M
    #i += cluster.shape[0]
    user_idx = 1 #U1_ids.iloc[0]
    difficulties_for_single_user = get_psedu_problem_difficulties_for_single_user(user_idx, M)

    difficulties_for_all_users = get_psedu_problem_difficulties(M)

    recommendation_difficulty_for_single_user, recommendation_idx_single = get_recommendation(difficulties_for_single_user)
    recommendation_difficulty_for_all_users,recommendation_idx_all = get_recommendation(difficulties_for_all_users)


    #TODO 1. Implement difficulty function DONE
    #TODO 2. construct utility matrix per user group using 1. then standardize across user DONE
    #TODO 3. Calculate aggregate difficulty measures DONE
    #TODO 4. Pick the X'th quantile and use for recommendation. Done
    #TODO 5. Figure out how to evaluate SPLIT ON 10-20 % RECENTLY SOLVED PROBLEMS


