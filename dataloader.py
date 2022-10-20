import pandas as pd, numpy as np
import os
from os.path import exists


def generate_subset_csv(df,filename,fractions=0.02,folder_path="data/"):
    filepath = os.path.join(folder_path,filename+'.csv')
    subset = df.iloc[:int(df.shape[0] * 0.05), :]
    subset.to_csv(filepath,index=False)
#
def load_data(subset=False):
    if subset:
        files_exists = exists('data/Info_UserData_subset.csv') and exists('data/Log_Problem_subset.csv') and exists('data/Info_Content_subset.csv')
        if not files_exists:
            # Run the following to generate subset
            df_user = pd.read_csv('data/Info_UserData.csv')
            df_problem = pd.read_csv('data/Log_Problem.csv')
            df_content = pd.read_csv('data/Info_Content.csv')
            [generate_subset_csv(df, filename) for df, filename in zip([df_user, df_problem, df_content], ["Info_UserData_subset", "Log_Problem_subset", "Info_Content_subset"])]

        df_user, df_problem, df_content = load_subset()
        return df_user, df_problem, df_content
    else:
        df_user = pd.read_csv('data/Info_UserData.csv')
        df_problem = pd.read_csv('data/Log_Problem.csv')
        df_content = pd.read_csv('data/Info_Content.csv')
        return df_user, df_problem, df_content

def load_subset():
    df_user = pd.read_csv('data/Info_UserData_subset.csv')
    df_problem = pd.read_csv('data/Log_Problem_subset.csv')
    df_content = pd.read_csv('data/Info_Content_subset.csv')
    return df_user,df_problem, df_content


if __name__ == "__main__":
    df_user, df_problem, df_content = load_data(subset=True)

    #remove outliers!
    #- time spend is more than XXX
    pass


