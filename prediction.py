from data.data_preprocessing import load_and_preprocess_user_data,preprocess_df
from data.data_loader import load_data_raw
from data.feature_categorization import U_features,Ex_features,Pr_features
if __name__ == "__main__":
    df_u, df_pr, df_ex = load_data_raw(subset=True)

    user_features = U_features()
    X = preprocess_df(df=df_u, o_features=user_features)
    # X = load_and_preprocess_user_data(df_u)
    # remove meta data:
    X = X.drop(['uuid'], axis=1)
    # df_u = load_and_preprocess_user_data(full_data=True)
    # df_pr = load_and_preprocess_problem_data(full_data=False,use_ex_features=True)
    pass
