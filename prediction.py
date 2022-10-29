from data.data_loader import load_data_raw

if __name__ == "__main__":
    df_u, df_pr, df_ex = load_data_raw(subset=True)

    # df_u = load_and_preprocess_user_data(full_data=True)
    # df_pr = load_and_preprocess_problem_data(full_data=False,use_ex_features=True)
    pass
