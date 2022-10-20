from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from class_definitions import User,Problem,Content
from data_preprocessing import scale, one_hot_encode




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

def user_clustering_kmeans(X):
    """
    cluster users and plot the elbow-graph to visualize most appropriate number of classes based on intertia
    :param X: preprocessed data
    :return:
    """
    inertias = []
    max_clusters = 25
    for i in range(1,max_clusters):
        kmeans = KMeans(n_clusters=i, random_state=0,max_iter=500).fit(X)
        # kmeans.labels_
        # kmeans.cluster_centers_
        inertias.append(kmeans.inertia_)
    plt.plot(list(range(1,max_clusters)),inertias)
    plt.title("intertia of kmeans")
    plt.xlabel("n_clusters")
    plt.ylabel("Inertia")
    plt.grid(True)
    plt.show()

def visualize_with_PCA(X,optimal_clusters=7):
    """
    Visualize clusters with PCA, given the optimal amount of clusters found using the graph produced by the function user_clustering_kmeans()
    :param X:
    :param optimal_clusters:
    :return:
    """
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=0, max_iter=500).fit(X)
    y = kmeans.predict(X)

    fig = plt.figure(1, figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d", elev=-150, azim=200) #azim=110

    pca = PCA(n_components=3)
    X_reduced = pca.fit_transform(X)
    print(pca.components_,"\n",X.columns,"\n",pca.explained_variance_ratio_)

    ax.scatter(
        X_reduced[:, 0],
        X_reduced[:, 1],
        X_reduced[:, 2],
        c=y,
        cmap=plt.cm.Set1,
        edgecolor="k",
        s=40,
    )

    ax.set_title("First three PCA directions")
    ax.set_xlabel("1st eigenvector")
    ax.w_xaxis.set_ticklabels([])
    ax.set_ylabel("2nd eigenvector")
    ax.w_yaxis.set_ticklabels([])
    ax.set_zlabel("3rd eigenvector")
    ax.w_zaxis.set_ticklabels([])
    plt.show()

def cluster_main():
    X = load_and_preprocess_user_data(full_data=True,enrich=False)
    user_clustering_kmeans(X)
    visualize_with_PCA(X)

if __name__ == "__main__":
    cluster_main()