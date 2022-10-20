from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler,MaxAbsScaler
import pandas as pd
# from tqdm import tqdm


def scale(df, scaler="standard"):
    """
    Standardize data with scikit learn scaler
    :param df: dataframe with data to be scaled
    :param scaler: method of scaling
    :return: scaled df
    """
    if scaler == "standard":
        scaler = StandardScaler()  # z = (x - u) / s, where u is column mean, and s is std. dev
    elif scaler == "minmax":
        scaler = MinMaxScaler(feature_range=(0, 1))  # scales columns to given range.
    else:
        scaler = MaxAbsScaler()

    scaler.fit(df)
    scaled = scaler.fit_transform(df)
    return pd.DataFrame(scaled, columns=df.columns)

def one_hot_encode(df,column_names=['gender','is_self_coach']):
    """
    one-hot-encode categorical features
    :param data: dataframe with only the columns to be one-hot-encoded
    :param column_names: names of the columns to be one-hot-encoded
    :return: one-hot-encoded columns ready to be concatenated with original dataframe
    """
    enc = OneHotEncoder(drop='first')
    enc.fit(df)
    print(enc.categories_)
    column_names = enc.get_feature_names(column_names)

    encoded_data = enc.transform(df).toarray()
    df_encoded = pd.DataFrame(encoded_data, columns=column_names)
    return df_encoded, column_names, enc

def extract_user_features(df_users):
    """
    not finished.
    intendend purpose: extract user features from problmes.
    # exercise_per_level
    # avg_time_spend
    # avg_correct
    # problem_solved
    :return:
    """
    users = df_users['uucid']
    df_problems = pd.read_csv('data/Log_Problem_subset.csv')
    for user in users:
        pass