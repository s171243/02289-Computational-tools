# from clustering import user_clustering_kmeans, visualize_with_PCA
# from data.data_loader import load_data_raw
# from data.data_preprocessing import extract_additional_user_features, preprocess_df
# from data.feature_categorization import U_features
# from scipy import sparse
# import numpy as np
# import time
#
#
# def get_user_segments(users,no_implementation = True, n_segments = 10):
#     """
#     use segmentation implementation to extract user segments
#     :return:
#     """
#     if no_implementation:
#         users_pr_segment = users.shape[0] // n_segments
#         users_in_segments = [users.iloc[i*users_pr_segment:(1+i)*users_pr_segment,:] for i in range(n_segments)]
#         return users_in_segments
# def construct_util_matrix(uuids,df_u,df_pr):
#     """
#     extract all users problems
#     :param users:
#     :param problems:
#     :return:
#     """
#     #Could be implemented with MapReduce
#     start_t = time.time()
#     upids = df_pr[df_pr.uuid.isin([uuids])]['upid']
#     print("Took {} seconds".format(start_t-time.time()))
#
#     n_users = len(uuids)
#     n_problems = len(upids)
#     U_matrix = np.empty((n_users,n_problems))
#     pass
#     return U_matrix
#
#
#
# def get_difficulty():
#     pass
#
#
# if __name__ == "__main__":
#     df_u, df_pr, df_c = load_data_raw(subset=False)
#     print("data loaded")
#     #X = extract_additional_user_features(df_u, df_pr, df_c)
#     #user_features = U_features()
#     #df_user = preprocess_df(df=X, o_features=user_features)
#
#     users_in_segments = get_user_segments(df_u)
#     test_segment = users_in_segments[0]
#     del users_in_segments,df_c
#     #Try with a single segment
#     construct_util_matrix(test_segment,df_u,df_pr)
#     pass
#
#
#     # X = X.drop(['uuid'], axis=1)
#     # user_clustering_kmeans(X)
#     # visualize_with_PCA(X)
#
#     #TODO 1. Implement difficulty function
#     #TODO 2. construct utility matrix per user group using 1., then standardize across user
#     #TODO 3. Calculate aggregate difficulty measures
#     #TODO 4. Pick the X'th quantile and use for recommendation.
#     #TODO 5. Figure out how to evaluate
# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# from sklearn.preprocessing import StandardScaler
# from clustering import user_clustering_kmeans, visualize_with_PCA
# from data.data_loader import load_data_raw
# from data.data_preprocessing import extract_additional_user_features, preprocess_df
# from data.feature_categorization import U_features
# import pickle
# from os.path import exists
#
# def get_user_segments(users,no_implementation = True, n_segments = 10):
#     """
#     use segmentation implementation to extract user segments
#     :return:
#     """
#     if no_implementation:
#
#         users_pr_segment = len(users // n_segments)
#         users_in_segments = [users[i:i+users_pr_segment] for i in range(n_segments)]
#         return users_in_segments
#
# def construct_util_matrix(users,problems):
#     M = np.empty((len(users),len(problems['upid'].unique())))
#     for i, userID in enumerate(tqdm(users['uuid'])):
#         user_problems = problems.loc[problems['uuid'] == userID]
#         if not user_problems.empty:
#             max_time = max(user_problems['total_sec_taken'])
#             for j in user_problems.index.to_list():
#                 problem_ID = user_problems.loc[[j],'upid'].values[0]
#                 problem_index = np.where(problems['upid'].unique() == problem_ID)
#                 time = user_problems.loc[[j],'total_sec_taken'].values[0]
#                 is_correct = user_problems.loc[[j],'is_correct'].values[0]
#                 M[i,problem_index[0][0]] = get_difficulty(time,max_time,is_correct)
#         else:
#             #M = np.delete(M, i, axis=0)
#             pass
#     return M
#
# def get_difficulty(time,max_time,is_correct):
#     """
#     :param time: Time (sec) spent on a given problem for a given student
#     :param max_time: Maximum time (sec) a given student has used on any problem
#     :param is_correct: boolean if the student solved the given problem (1) or not (0)
#     :return: Relative difficulty of a problem for a given student
#     """
#     difficulty = 1-((max_time - time) / max_time)*is_correct
#     return difficulty
#
# def generate_M_matrix(cluster_id=3,subset=False):
#     df_u, df_pr, _ = load_data_raw(subset=subset)
#     # Have for loop over all clusters, for now we just consider a single cluster
#     clusters = pd.read_csv(r'data/csv_files/clusters.csv')
#     cluster = clusters.loc[clusters['cluster'] == cluster_id]
#     sub_users = df_u['uuid'].isin(cluster['uuid'].to_list())
#     df_u_sub = df_u.loc[sub_users]
#     problems = []
#     for userID in tqdm(cluster['uuid']):
#         user_info = df_pr.loc[df_pr['uuid'] == userID]
#         for problem in user_info['upid']:
#             problems.append(problem)
#     sub_problems = df_pr['upid'].isin(problems)
#     df_pr_sub = df_pr.loc[sub_problems]
#     M = construct_util_matrix(df_u_sub, df_pr_sub)
#     with open('data/pickle_files/M_matrix_subset_{}_cluster_{}.pkl'.format(subset,cluster_id), 'wb') as f:
#         pickle.dump(M, f)
# def generate_U_matrix(subset=False):
#     df_u, df_pr, df_c = load_data_raw(subset=subset)
#
#     print("data loaded")
#     X = extract_additional_user_features(df_u, df_pr, df_c)
#     user_features = U_features()
#     X = preprocess_df(df=X, o_features=user_features)
#     X = X.drop(['uuid'], axis=1)
#
#     clusters = pd.read_csv(r'data/csv_files/clusters.csv')
#     if sub:
#         problems = []
#         for userID in tqdm(clusters['uuid']):
#             user_info = df_pr.loc[df_pr['uuid'] == userID]
#             for problem in user_info['upid']:
#                 problems.append(problem)
#         sub_problems = df_pr['upid'].isin(problems)
#         df_pr_sub = df_pr.loc[sub_problems]
#         U = np.empty((clusters.shape[0], len(df_pr_sub['upid'].unique())))
#     else:
#         U = np.empty((clusters.shape[0], len(df_pr['upid'].unique())))
#     with open('data/pickle_files/U_matrix_subset_{}.pkl'.format(subset), 'wb') as f:
#         pickle.dump(U, f)
#
#
# def get_U_and_M_matrix(subset=False,generate_new_U_matrix = False):
#     """
#     Load utility matrix and subset matrix M
#     :param subset:
#     :param generate_new_U_matrix:
#     :return:
#     """
#     files_exists = exists('data/pickle_files/M_matrix_subset_{}.pkl'.format(subset)) and exists('data/pickle_files/U_matrix_subset_{}.pkl'.format(subset))
#     if not files_exists:
#         generate_U_and_M_matrix(subset=subset)
#         U, M = get_U_and_M_matrix(subset=subset)
#     else:
#         with open('data/pickle_files/U_matrix_subset_{}.pkl'.format(sub), 'rb') as f:
#             U = pickle.load(f)
#         with open('data/pickle_files/M_matrix_subset_{}.pkl'.format(sub), 'rb') as f:
#             M = pickle.load(f)
#     return U, M
#
#
# if __name__ == "__main__":
#     sub = False
#     if sub:
#         df_u, df_pr, df_c = load_data_raw(subset=True)
#     else:
#         df_u, df_pr, df_c = load_data_raw()
#     print("data loaded")
#     X = extract_additional_user_features(df_u, df_pr, df_c)
#     user_features = U_features()
#     X = preprocess_df(df=X, o_features=user_features)
#     X = X.drop(['uuid'], axis=1)
#     # user_clustering_kmeans(X)
#     # visualize_with_PCA(X)
#     clusters = pd.read_csv(r'data/csv_files/clusters.csv')
#     if sub:
#         problems = []
#         for userID in tqdm(clusters['uuid']):
#             user_info = df_pr.loc[df_pr['uuid'] == userID]
#             for problem in user_info['upid']:
#                 problems.append(problem)
#         sub_problems = df_pr['upid'].isin(problems)
#         df_pr_sub = df_pr.loc[sub_problems]
#         U = np.empty((clusters.shape[0], len(df_pr_sub['upid'].unique())))
#     else:
#         U = np.empty((clusters.shape[0], len(df_pr['upid'].unique())))
#     i = 0
#     # Have for loop over all clusters, for now we just consider a single cluster
#     cluster = clusters.loc[clusters['cluster'] == 3]
#     sub_users = df_u['uuid'].isin(cluster['uuid'].to_list())
#     df_u_sub = df_u.loc[sub_users]
#     problems = []
#     for userID in tqdm(cluster['uuid']):
#         user_info = df_pr.loc[df_pr['uuid'] == userID]
#         for problem in user_info['upid']:
#             problems.append(problem)
#     sub_problems = df_pr['upid'].isin(problems)
#     df_pr_sub = df_pr.loc[sub_problems]
#     M = construct_util_matrix(df_u_sub, df_pr_sub)
#     M = StandardScaler().fit_transform(M)  # We should instead standardize across non-empty entries only
#     U[i:(cluster.shape[0] + i), :] = M
#     i += cluster.shape[0]
#     # Now that we have a utility matrix, we need to fill all empty entries
#     M2 = np.copy(M)  # Create copy of utility matrix, so that 'predictions' are not used when aggregating
#     user_recommendations = np.zeros(np.shape(M)[0])
#     for user in np.shape(M)[0]:
#         unsolved_problems = np.argwhere(M[user, :] == 0.0)
#         # for each unsolved problem we aggregate the difficulty for all similar users (users in same cluster)
#         for prob in unsolved_problems:
#             sim_users = np.argwhere(M[:, prob] != 0.0)
#             aggregate = np.mean(M[sim_users, prob])
#             M2[user, prob] = aggregate
# #
# # #############
# #
# #
# #     print("data loaded")
# #     X = extract_additional_user_features(df_u, df_pr, df_c)
# #     user_features = U_features()
# #     X = preprocess_df(df=X, o_features=user_features)
# #     X = X.drop(['uuid'], axis=1)
# #     #user_clustering_kmeans(X)
# #     #visualize_with_PCA(X)
# #     clusters = pd.read_csv(r'data/csv_files/clusters.csv')
# #     if sub:
# #         problems = []
# #         for userID in tqdm(clusters['uuid']):
# #             user_info = df_pr.loc[df_pr['uuid'] == userID]
# #             for problem in user_info['upid']:
# #                 problems.append(problem)
# #         sub_problems = df_pr['upid'].isin(problems)
# #         df_pr_sub = df_pr.loc[sub_problems]
# #         U = np.empty((clusters.shape[0],len(df_pr_sub['upid'].unique())))
# #     else:
# #         U = np.empty((clusters.shape[0], len(df_pr['upid'].unique())))
# #     i = 0
# #     # Have for loop over all clusters, for now we just consider a single cluster
# #     cluster = clusters.loc[clusters['cluster']==3]
# #     sub_users = df_u['uuid'].isin(cluster['uuid'].to_list())
# #     df_u_sub = df_u.loc[sub_users]
# #     problems = []
# #     for userID in tqdm(cluster['uuid']):
# #         user_info = df_pr.loc[df_pr['uuid'] == userID]
# #         for problem in user_info['upid']:
# #             problems.append(problem)
# #     sub_problems = df_pr['upid'].isin(problems)
# #     df_pr_sub = df_pr.loc[sub_problems]
# #     M = construct_util_matrix(df_u_sub,df_pr_sub)
# #
# #     M = StandardScaler().fit_transform(M) # We should instead standardize across non-empty entries only
# #     U[i:(cluster.shape[0]+i),:] = M
# #     i += cluster.shape[0]
# #     # Now that we have a utility matrix, we need to fill all empty entries
# #     M2 = np.copy(M) # Create copy of utility matrix, so that 'predictions' are not used when aggregating
# #     user_recommendations = np.zeros(np.shape(M)[0])
# #     for user in np.shape(M)[0]:
# #         unsolved_problems = np.argwhere(M[user,:]==0.0)
# #         # for each unsolved problem we aggregate the difficulty for all similar users (users in same cluster)
# #         for prob in unsolved_problems:
# #             sim_users = np.argwhere(M[:,prob]!=0.0)
# #             aggregate = np.mean(M[sim_users,prob])
# #             M2[user,prob] = aggregate
# #         # user_recommendations[user] = find 75'th quantile across unsolved problems for recommodation
#
#
#     #TODO 1. Implement difficulty function DONE
#     #TODO 2. construct utility matrix per user group using 1. then standardize across user DONE
#     #TODO 3. Calculate aggregate difficulty measures DONE
#     #TODO 4. Pick the X'th quantile and use for recommendation. ALMOST
#     #TODO 5. Figure out how to evaluate SPLIT ON 10-20 % RECENTLY SOLVED PROBLEMS
#
