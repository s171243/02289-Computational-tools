from clustering import split_users
from cure import *
from data.data_loader import load_data_raw
from data.data_preprocessing import preprocess_df, extract_additional_user_features
from data.feature_categorization import U_features


def main():
    df_u, df_pr, df_c = load_data_raw(subset=True)
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
            # INSERT CLUSTERING AND OTHER STUFF HERE
            pass
    else:
        # INSERT CLUSTERING AND OTHER STUFF HERE
        pass

    data = X.drop(columns=["uuid"]).to_numpy()
    cure_repres = cure_representatives(data)
    print("CURE Representatives calculated")
    labels = cure_classify(data, cure_repres)
    print("CURE Classification done")

    n_clusters = labels.max() + 1
    partition = lambda k: data[labels == k]
    similarities = [pairwise_distances(partition(k)) for k in range(n_clusters)]
    # TODO users for similarity
    print("Similarity matrices calculated")


if __name__ == '__main__':
    main()
