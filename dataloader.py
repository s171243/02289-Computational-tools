import pandas as pd, numpy as np
import os
from os.path import exists

from data_preprocessing import one_hot_encode, scale


def generate_subset_csv(df,filename,fractions=0.02,folder_path="data/"):
    filepath = os.path.join(folder_path,filename+'.csv')
    subset = df.iloc[:int(df.shape[0] * 0.05), :]
    subset.to_csv(filepath,index=False)
#
def load_data_raw(subset=False):
    if subset:
        files_exists = exists('data/Info_UserData_subset.csv') and exists('data/Log_Problem_subset.csv') and exists('data/Info_Content_subset.csv')
        if not files_exists:
            # Run the following to generate subset
            df_user = pd.read_csv('data/Info_UserData.csv')
            df_problem = pd.read_csv('data/Log_Problem.csv')
            df_content = pd.read_csv('data/Info_Content.csv')
            [generate_subset_csv(df, filename) for df, filename in zip([df_user, df_problem, df_content], ["Info_UserData_subset", "Log_Problem_subset", "Info_Content_subset"])]

        df_user, df_problem, df_content = load_subset_raw()
        return df_user, df_problem, df_content
    else:
        df_user = pd.read_csv('data/Info_UserData.csv')
        df_problem = pd.read_csv('data/Log_Problem.csv')
        df_content = pd.read_csv('data/Info_Content.csv')
        return df_user, df_problem, df_content

def load_subset_raw():
    df_user = pd.read_csv('data/Info_UserData_subset.csv')
    df_problem = pd.read_csv('data/Log_Problem_subset.csv')
    df_content = pd.read_csv('data/Info_Content_subset.csv')
    return df_user,df_problem, df_content


def load_and_preprocess_user_data(full_data=False,user_features=[]):
    """
    Load user data (whole or subset), pick relevant user features, one-hot-encode (not a general implem.), scale data
    :param full_data: user dataframe
    :param user_features: features to be used and preprocessed
    :return: user dataframe ready for clustering
    """
    if full_data:
        df_user = pd.read_csv('data/Info_UserData.csv')
    else:
        df_user = pd.read_csv('data/Info_UserData_subset.csv')

    if len(user_features)==0:
        user_features = ['points', 'badges_cnt', 'user_grade', 'has_teacher_cnt', 'has_student_cnt', 'belongs_to_class_cnt',
         'has_class_cnt']

    #Select relevant clustering features (except for columns that need encoded)
    X = df_user[user_features]
    #We need gender & is_self_coach to be onehotencoded
    columns_to_be_encoded = df_user[['gender','is_self_coach']]
    columns_to_be_encoded['gender'] = columns_to_be_encoded['gender'].fillna("unspecified")
    columns_encoded, column_names, enc = one_hot_encode(columns_to_be_encoded)

    df_one_hot_encoded = pd.concat([X,columns_encoded],axis=1)
    df_one_hot_encoded_and_scaled = scale(df_one_hot_encoded)

    return df_one_hot_encoded_and_scaled



if __name__ == "__main__":
    df_user, df_problem, df_content = load_data_raw(subset=True)

    #remove outliers!
    #- time spend is more than XXX
    pass


