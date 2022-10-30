from data.data_preprocessing import preprocess_df, remove_outliers_by_quantile
from data.data_loader import load_data_raw
from data.feature_categorization import U_features,Ex_features,Pr_features
from data.data_visualization import plot_columns_of_df
import time
import pandas as pd


if __name__ == "__main__":
    df_u, df_pr, df_ex = load_data_raw(subset=False)
    u_features,pr_features,ex_features = U_features(), Pr_features(),Ex_features(features_to_be_OH_encoded=['level2_id','level3_id'])
    start_time = time.time()

    X_u = preprocess_df(df=df_u, o_features=u_features)
    X_pr = preprocess_df(df=df_pr, o_features=pr_features)
    X_ex = preprocess_df(df=df_ex, o_features=ex_features)
    print("seconds:\t", time.time()-start_time)

    # plot_columns_of_df(X_u)
    X_u = remove_outliers_by_quantile(X_u,columns=['points'],quantiles=[0.90])
    X_u.convert_dtypes()
    # plot_columns_of_df(X_pr)
    X_pr = remove_outliers_by_quantile(X_pr,columns=['problem_number','exercise_problem_repeat_session','total_sec_taken','total_attempt_cnt'],quantiles=[0.84,0.84,0.84,0.84])
    X_pr.convert_dtypes()



    del (df_u, df_pr, df_ex,u_features,pr_features,ex_features)
    X = X_pr.merge(X_ex,on="ucid",how="left")
    del(X_pr, X_ex)
    X = X.merge(X_u,on="uuid",how="left")
    del(X_u)
    X.dropna(inplace=True)

    X,y = X.loc[ : , X.columns != 'is_correct_True'], X['is_correct_True']
    y.to_pickle("data/pickle_files/y_w_only_level_id2.pkl")
    del y
    X.to_pickle("data/pickle_files/X_w_only_level_id2.pkl")
    # remove meta data:
    # X_u = X_u.drop(['uuid'], axis=1)
    print("seconds:\t", time.time()-start_time)

    pass
    a = 2
