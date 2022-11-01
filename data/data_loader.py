import pandas as pd
import os
from os.path import exists


def generate_subset_csv(df, filename, fractions=0.02, folder_path="data/"):
    filepath = os.path.join(folder_path, filename + '.csv')
    subset = df.iloc[:int(df.shape[0] * 0.05), :]
    subset.to_csv(filepath, index=False)


def load_data_raw(subset=False):
    if subset:
        files_exists = exists('data/csv_files/Info_UserData_subset.csv') and exists(
            'data/csv_files/Log_Problem_subset.csv') and exists(
            'data/csv_files/Info_Content_subset.csv')
        if not files_exists:
            # Run the following to generate subset
            df_u = pd.read_csv('data/csv_files/Info_UserData.csv')
            df_pr = pd.read_csv('data/csv_files/Log_Problem.csv')
            df_ex = pd.read_csv('data/csv_files/Info_Content.csv')
            [generate_subset_csv(df, filename) for df, filename in
             zip([df_u, df_pr, df_ex], ["Info_UserData_subset", "Log_Problem_subset", "Info_Content_subset"])]

        df_u, df_pr, df_ex = load_subset_raw()
        return df_u, df_pr, df_ex
    else:
        df_u = pd.read_csv('data/csv_files/Info_UserData.csv')
        df_pr = pd.read_csv('data/csv_files/Log_Problem.csv')
        df_ex = pd.read_csv('data/csv_files/Info_Content.csv')
        return df_u, df_pr, df_ex


def load_subset_raw():
    df_u = pd.read_csv('data/csv_files/Info_UserData_subset.csv')
    df_pr = pd.read_csv('data/csv_files/Log_Problem_subset.csv')
    df_ex = pd.read_csv('data/csv_files/Info_Content_subset.csv')
    return df_u, df_pr, df_ex


if __name__ == "__main__":
    df_u, df_pr, df_ex = load_data_raw(subset=True)

    # remove outliers!
    # - time spend is more than XXX
    pass
