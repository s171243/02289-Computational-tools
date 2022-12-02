import warnings
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, MaxAbsScaler

from data_visualization import plot_columns_of_df

warnings.filterwarnings("ignore")


####Preproccesing steps
# 0. Load data
# 1. Categorize features and encode accordingly
# 2. Handle missing data
# 3. Visualize
# 4. Identify and remove outliers
# 5. Potentially enrich data with feature engineering

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


def unix_encode(df, column_names=[]):
    """
    Encode date to unix format (sec/min/hour since unix epoch), that can be used for clustering/prediction
    A kind of ugly implementation trying to speed up to_datetime by providing the relevant time formats
    (There is a 10-100x speed up to give correct format)
    :param df:
    :param column_names:
    :return:
    """
    time_columns = np.array([])
    t_formats = ['%Y-%m-%d', '%Y-%m-%d %H:%M:%S %Z']
    for column in column_names:
        success = False
        # temp = (pd.to_datetime(df[column].iloc[:, 0]) - pd.Timestamp("1970-01-01"))
        for t_format in t_formats:
            if not success:
                try:
                    temp = pd.to_datetime(df[column], format=t_format)  # Takes 15 sec for subset on df_pr
                    success = True
                except:
                    print("incorrect time format {}".format(t_format))

        temp -= temp.min()
        temp //= pd.Timedelta('15m')
        if time_columns.shape[0] == 0:
            time_columns = np.array(temp)
        else:
            time_columns = np.vstack((time_columns, temp))

    return pd.DataFrame(time_columns.T, columns=column_names)


def ordinal_encode(df, column_names=[]):
    """
      ordinal-encode features
      :param df: dataframe with only the columns to be one-hot-encoded
      :param column_names: names of the columns to be one-hot-encoded
      :return: one-hot-encoded columns ready to be concatenated with original dataframe
      """
    encoded_data = np.array([])
    mappings = \
        {
            'difficulty': {'easy': 0, 'normal': 1, 'unset': 1, 'hard': 2},
            'learning_stage': {'elementary': 0, 'junior': 1, 'senior': 2}
        }
    for column in column_names:
        try:
            mapping = mappings[column]
        except:
            print(
                "Please add ordinal mapping to ordinal_dict in 'ordinal_encode'-function in data/data_preprocessing.py")
        temp = list(map(lambda x: mapping[x], df[column]))
        if encoded_data.shape[0] == 0:
            encoded_data = np.array(temp)
        else:
            encoded_data = np.vstack((encoded_data, temp))
    df_encoded = pd.DataFrame(encoded_data.T, columns=column_names)
    return df_encoded, column_names


def one_hot_encode(df, column_names=['gender', 'is_self_coach']):
    """
    one-hot-encode categorical features
    :param df: dataframe with only the columns to be one-hot-encoded
    :param column_names: names of the columns to be one-hot-encoded
    :return: one-hot-encoded columns ready to be concatenated with original dataframe
    """
    enc = OneHotEncoder()
    enc.fit(df)
    # print(enc.categories_)
    column_names = enc.get_feature_names(column_names)

    encoded_data = enc.transform(df).toarray()
    df_encoded = pd.DataFrame(encoded_data, columns=column_names)
    return df_encoded, column_names, enc


def extract_additional_user_features(df_u, df_problems, df_content):
    """
    intendend purpose: extract user features from problems. Potentially using map_reduce
    # exercise_per_level
    # avg_time_spend
    # avg_correct
    # problem_solved
    :return:
    """

    # Get features based on content
    problem_content = df_problems.merge(df_content)
    encoded_df, cols = ordinal_encode(problem_content[["difficulty", "learning_stage"]],
                                      column_names=["difficulty", "learning_stage"])
    problem_content[cols] = encoded_df[cols]
    problem_content_grouped = problem_content.groupby("uuid").agg({
        "is_correct": "mean",
        "total_sec_taken": "mean",
        "upid": "count",
        "level": ["mean", "max"],
        "used_hint_cnt": "mean",
        "difficulty": "mean",
        "learning_stage": "mean",
    })
    cols = [
        "correct_percentage",
        "time_spent",
        "problems_attempted",
        "average_level", "max_level",
        "average_hints",
        "avg_difficulty",
        "avg_learning_stage",
    ]
    problem_content_grouped.columns = cols

    users = df_u.merge(problem_content_grouped, left_on="uuid", right_on="uuid")

    return users


def _extract_additional_user_features(df_u, df_problems, df_content):
    """
    intendend purpose: extract user features from problems. Potentially using map_reduce
    # exercise_per_level
    # avg_time_spend
    # avg_correct
    # problem_solved
    :return:
    """

    os.system("python sandbox/MapReduceSandbox.py data/csv_files/Log_Problem_subset.csv > data/csv_files/reduced.csv")
    reduced = pd.read_csv("data/csv_files/reduced.csv",
                          names=["uuid", "problems_attempted", "time_spent", "average_level", "correct_percentage",
                                 "max_level", "avg_learning_stage", "avg_difficulty", "average_hints"], index_col=False)


    users = df_u.merge(reduced, left_on="uuid", right_on="uuid")

    return users


def extract_additional_problem_features(df_ex):
    """
    :param df_ex:
    :return:
    """


def preprocess_df(df, o_features) -> pd.DataFrame:
    """

    :param df:
    :param o_features: object features as described in data/feature_categorization.py
    :return:
    """
    f, f_time, f_OR, f_OH, f_meta, f_scale = o_features.features, o_features.features_to_be_time_encoded, o_features.features_to_be_OR_encoded, o_features.features_to_be_OH_encoded, o_features.features_meta, o_features.features_to_be_scaled
    df_f, df_time, df_OR, df_OH, df_meta, df_scale = df[f], df[f_time], df[f_OR], df[f_OH], df[f_meta], df[f_scale]

    if f_time:  # if not empty then unix encode
        df_time = unix_encode(df_time,
                              f_time)  # Takes a significant amount of time to compute (for subset it is approx 20 seconds)

    if f_scale:
        df_scale = scale(df_scale)
        plot_columns_of_df(df_scale)

    if f_OR:
        df_OR, column_names_OR = ordinal_encode(df_OR, f_OR)

    if f_OH:
        for column in df_OH.columns:
            if df_OH[column].hasnans:
                df_OH[column] = df_OH[column].astype('string').fillna("unspecified")

        df_OH, column_names_OH, enc_OH = one_hot_encode(df_OH, f_OH)
    dfs = [df_f, df_time, df_OR, df_OH, df_meta, df_scale]
    df_counter = 0
    for df_ in dfs:
        if df_.shape[1] != 0:
            df_counter += 1
            if df_counter == 1:
                df_final = df_
            else:
                df_final = pd.concat([df_final, df_], axis=1)
    return df_final


def remove_outliers_by_quantile(df, columns, quantiles):
    for column, quantile_ in zip(columns, quantiles):
        index_names = df[(df[column] > df[column].quantile(quantile_))].index
        df.drop(index_names, inplace=True)
    return df
