from clustering import user_clustering_kmeans, visualize_with_PCA
from data.data_loader import load_data_raw
from data.data_preprocessing import extract_additional_user_features, preprocess_df
from data.feature_categorization import U_features


def get_user_segments(users,no_implementation = True, n_segments = 10):
    """
    use segmentation implementation to extract user segments
    :return:
    """
    if no_implementation:

        users_pr_segment = len(users // n_segments)
        users_in_segments = [users[i:i+users_pr_segment] for i in range(n_segments)]
        return users_in_segments
def construct_util_matrix(users,problems):

def get_difficulty



if __name__ == "__main__":

    df_u, df_pr, df_c = load_data_raw(subset=True)
    print("data loaded")
    X = extract_additional_user_features(df_u, df_pr, df_c)
    user_features = U_features()
    X = preprocess_df(df=X, o_features=user_features)
    X = X.drop(['uuid'], axis=1)
    user_clustering_kmeans(X)
    visualize_with_PCA(X)

    #TODO 1. Implement difficulty function
    #TODO 2. construct utility matrix per user group using 1., then standardize across user
    #TODO 3. Calculate aggregate difficulty measures
    #TODO 4. Pick the X'th quantile and use for recommendation.
    #TODO 5. Figure out how to evaluate


