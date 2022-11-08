from data.data_preprocessing import preprocess_df, remove_outliers_by_quantile
from data.data_loader import load_data_raw
from data.feature_categorization import U_features, Ex_features, Pr_features
from data.data_visualization import plot_columns_of_df, hist_plot
import time
from sklearn.metrics import confusion_matrix
from os.path import exists
import pandas as pd
from collections import Counter
import numpy as np
from sklearn import model_selection
from sklearn import preprocessing
from scipy import stats
from random import randint

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm


def get_x_and_y_for_prediction(subset=True):
    if subset:
        files_exists = exists('data/pickle_files/X_subset_w_level_id2_level_id3.pkl') and exists(
            'data/pickle_files/y_subset_w_level_id2_level_id3.pkl')
    else:
        files_exists = exists('data/pickle_files/X_w_level_id2_level_id3.pkl') and exists(
            'data/pickle_files/y_w_level_id2_level_id3.pkl')
    if not files_exists:
        generate_x_and_y_for_prediction(subset=subset)
        X, y = get_x_and_y_for_prediction(subset=subset)
    else:
        if subset:
            X = pd.read_pickle("data/pickle_files/X_subset_w_level_id2_level_id3.pkl")
            y = pd.read_pickle("data/pickle_files/y_subset_w_level_id2_level_id3.pkl")
        else:
            X = pd.read_pickle("data/pickle_files/X_w_level_id2_level_id3.pkl")
            y = pd.read_pickle("data/pickle_files/y_w_level_id2_level_id3.pkl")
    return X, y


def generate_x_and_y_for_prediction(subset=True):
    df_u, df_pr, df_ex = load_data_raw(subset=subset)
    u_features, pr_features, ex_features = U_features(features=['points','badges_cnt','user_grade','has_teacher_cnt','has_student_cnt','belongs_to_class_cnt','has_class_cnt'],features_to_be_time_encoded=['first_login_date_TW'],features_to_be_OR_encoded=[],features_to_be_OH_encoded=['gender','user_city','is_self_coach'],features_meta=['uuid']), Pr_features(), Ex_features(
        features_to_be_OH_encoded=['level2_id', 'level3_id'])
    # start_time = time.time()

    X_u = preprocess_df(df=df_u, o_features=u_features)
    X_pr = preprocess_df(df=df_pr, o_features=pr_features)
    X_ex = preprocess_df(df=df_ex, o_features=ex_features)

    # plot_columns_of_df(X_u)
    X_u = remove_outliers_by_quantile(X_u, columns=['points'], quantiles=[0.84])
    X_u.convert_dtypes()
    # plot_columns_of_df(X_pr)
    X_pr = remove_outliers_by_quantile(X_pr,
                                       columns=['problem_number', 'exercise_problem_repeat_session', 'total_sec_taken',
                                                'total_attempt_cnt'], quantiles=[0.82, 0.82, 0.82, 0.82])
    X_pr.convert_dtypes()

    del (df_u, df_pr, df_ex, u_features, pr_features, ex_features)
    X = X_pr.merge(X_ex, on="ucid", how="left")
    del (X_pr, X_ex)
    X = X.merge(X_u, on="uuid", how="left")
    del (X_u)
    X.dropna(inplace=True)
    X, y = X.loc[:, X.columns != 'is_correct_True'], X['is_correct_True']
    if subset:
        y.to_pickle("data/pickle_files/y_subset_w_level_id2_level_id3.pkl")
    else:
        y.to_pickle("data/pickle_files/y_w_level_id2_level_id3.pkl")
    del y
    if subset:
        X.to_pickle("data/pickle_files/X_subset_w_level_id2_level_id3.pkl")
    else:
        X.to_pickle("data/pickle_files/X_w_level_id2_level_id3.pkl")


def confint(vector, interval):
    # Standard deviation of sample
    vec_sd = np.std(vector)
    # Sample size
    n = len(vector)
    # Mean of sample
    vec_mean = np.mean(vector)
    # Error according to t distribution
    error = 1.984217 * vec_sd / np.sqrt(n)
    # Confidence interval as a vector
    result = np.array([vec_mean - error, vec_mean + error])
    return (result)


if __name__ == "__main__":
    sub = False
    X, y = get_x_and_y_for_prediction(subset=sub)
    frac = 0.1
    print("data is loaded")
    np.random.seed(0)
    if not sub:
        chosen_idx = np.random.choice(int(X.shape[0]), replace=False, size= int(X.shape[0] * frac))
        X = X.iloc[chosen_idx]
        y = y.iloc[chosen_idx]
    X_meta = X[['uuid','ucid','upid']]

    X.drop(['uuid','ucid','upid'],axis=1,inplace=True)
    y = y.to_numpy()
    X_0 = X.iloc[0]
    X = preprocessing.scale(X)
    K = 10
    kf = model_selection.KFold(n_splits=K, shuffle=True)

    # svm_predict = np.array([])
    # GNB_predict = np.array([])
    # clf_predict = np.array([])
    lreg_predict = np.array([])
    # DecisionTree_predict = np.array([])
    RF_predict = np.array([])

    opt_lambda = np.array([])
    y_true = np.array([])

    c = 0
    val_lg = np.array([])
    # val_nn = np.array([])
    # val_sv = np.array([])
    val_RF = np.array([])


    yhat = []
    y_true = []
    indices = []
    for train_index, test_index in tqdm(kf.split(X)):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        y_true = np.append(y_true, y_test)
        indices = np.append(indices, test_index)

        print("mean of y_test: \t {}".format(y_test.mean()))

        # support vector machine
        # m_svm = SVC(gamma='auto', kernel="linear")
        # m_svm.fit(X_train, y_train)
        # svm_predict = np.append(svm_predict, m_svm.predict(X_test))

        # m_DecisionTree = DecisionTreeClassifier()
        # m_DecisionTree.fit(X_train, y_train)
        # DecisionTree_predict = np.append(DecisionTree_predict, m_DecisionTree.predict(X_test))

        # clf = MLPClassifier(activation='relu', solver='adam', alpha=1e-5, hidden_layer_sizes=(200, 200, 10),
        #                     max_iter=300,
        #                     momentum=0.9)
        # clf.fit(X_train, y_train)
        # clf_predict = np.append(clf_predict, clf.predict(X_test))

        # m_gaus = GaussianNB()
        # m_gaus.fit(X_train, y_train)
        # GNB_predict = np.append(GNB_predict, m_gaus.predict(X_test))

        ranFor = RandomForestClassifier(n_estimators=100)
        ranFor.fit(X_train, y_train)
        RF_predict = np.append(RF_predict, ranFor.predict(X_test))

        lreg = LogisticRegression(penalty="l2", solver="saga", multi_class="auto", max_iter=500)
        lreg.fit(X_train, y_train)
        lreg_predict = np.append(lreg_predict, lreg.predict(X_test))

        # print("svm_acc:", np.mQean(y_true == svm_predict))
        # print("Dec:", np.mean(y_true == DecisionTree_predict))
        # print("MLP:", np.mean(y_true == clf_predict))
        # print("GNB:", np.mean(y_true == GNB_predict))
        print("RandForest:", np.mean(y_true == RF_predict))
        print("LogReg:", np.mean(y_true == lreg_predict))


        val_lg = np.append(val_lg, np.mean(y_true == lreg_predict))
        val_RF = np.append(val_RF, np.mean(y_true == RF_predict))
        # val_nn = np.append(val_nn, np.mean(y_true == clf_predict))
        # val_sv = np.append(val_sv, np.mean(y_true == svm_predict))

    # print("svm_acc:", np.mean(y_true == svm_predict))
    # print("Dec:", np.mean(y_true == DecisionTree_predict))
    # print("MLP:", np.mean(y_true == clf_predict))
    # print("GNB:", np.mean(y_true == GNB_predict))
    print("LG:", np.mean(y_true == lreg_predict))
    print("RF:", np.mean(y_true == RF_predict))


    # print("CI nn: ", confint(val_nn, 0.95))
    print("CI lg: ", confint(val_lg, 0.95))
    print("CI RF: ", confint(val_RF, 0.95))
    # print("CI sv: ", confint(val_sv, 0.95))

    [print("{},{}".format(column, round(coef, 4))) for (coef, column) in zip(lreg.coef_.squeeze(), list(X_0.keys()))]
    x_meta_predicted = X_meta.iloc[indices]
    wrong_pred = [y_true == lreg_predict]
    uuuid_user_w_wrong_pred = X_meta[~ (y_true == lreg_predict)]['uuid'].unique()
    unique_users_w_wrong_pred = len(uuuid_user_w_wrong_pred)
    unique_users = len(X_meta['uuid'].unique())
    print("{} out of {} has incorrect preds, {}".format(unique_users_w_wrong_pred,unique_users_w_wrong_pred, (unique_users_w_wrong_pred/unique_users_w_wrong_pred)*100 ))
    no_of_incorrect_per_user = Counter(X_meta[~ (y_true == lreg_predict)]['uuid'])
    hist_plot(no_of_incorrect_per_user.values(),"number of incorrects pr. user")
    con_matrix = confusion_matrix(y_true, lreg_predict)
    tn, fp, fn, tp = con_matrix.ravel()
    print("Confusion matrix \n{}".format(con_matrix))
    pass
    # print(np.std(y_true == svm_predict),np.std(y_true == DecisionTree_predict),np.std(y_true==clf_predict),np.std(y_true==GNB_predict))
