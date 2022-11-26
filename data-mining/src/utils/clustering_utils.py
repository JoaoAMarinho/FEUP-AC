from matplotlib.pyplot import cm
from matplotlib import colors
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
import plotly.graph_objects as go
from itertools import product
import scipy.cluster.hierarchy as sch
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import utils.clean_utils as cu

DEBUG = True
#######
# Utils
#######

def pca_analysis(data):
    n_components = len(data.columns)

    pca = PCA(n_components=n_components)
    pca.fit(data)
    variance = pca.explained_variance_ratio_ 
    var=np.cumsum(np.round(variance, 3)*100)
    plt.figure(figsize=(12,6))
    plt.ylabel('% Variance Explained')
    plt.xlabel('# of Features')
    plt.title('PCA Analysis')
    plt.ylim(0,100.5)
    plt.plot(var)
    plt.show()

def dbscan_param_tuning_silhouette(df_scale):
    eps_values = np.arange(0.2,1.5,0.1) 
    min_samples = np.arange(2,5) 
    dbscan_params = list(product(eps_values, min_samples))
    no_of_clusters = []
    sil_score = []
    epsvalues = []
    min_samp = []

    for p in dbscan_params:
        dbscan_cluster = DBSCAN(eps=p[0], min_samples=p[1]).fit(df_scale)
        epsvalues.append(p[0])
        min_samp.append(p[1])
        if len(np.unique(dbscan_cluster.labels_)) > 1:
            no_of_clusters.append(len(np.unique(dbscan_cluster.labels_)))
            sil_score.append(silhouette_score(df_scale, dbscan_cluster.labels_))

    eps_min = list(zip(no_of_clusters, sil_score, epsvalues, min_samp))
    eps_min_df = pd.DataFrame(eps_min, columns=['no_of_clusters', 'silhouette_score', 'epsilon_values', 'minimum_points'])
    print(eps_min_df)

def pca(n_components, data):
    pca_cols = []
    for i in range(n_components):
        pca_cols.append('pc'+str(i+1))

    pca = PCA(n_components=n_components)
    pca.fit(data)
    pca_scale = pca.transform(data)
    pca_df = pd.DataFrame(pca_scale, columns=pca_cols)

    return pca_df

def elbow_method(df):
    df_copy = df.copy()

    nr_clusters = []
    inertias = []
    range_values = np.arange(2,8)

    for k in range_values:
        kmeans = KMeans(k)
        kmeans.fit(df_copy)
        nr_clusters.append(k)
        inertias.append(kmeans.inertia_)

    plt.rcParams.update({'font.size': 22})
    plt.plot(nr_clusters, inertias, "-o", color='#138f8d')
    plt.title('Evolution of Inertia with number of clusters')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.show()

def silhouette_score(df, labels):
    return metrics.silhouette_score(df, labels, metric='euclidean')

##########
# Features
##########

def merge_transactions_clients(dfs):
    [ accounts, disp, clients, districts, cards, trans ] = dfs

    clients = cu.clean_clients(clients, clean=False)
    accounts = cu.clean_accounts(accounts, clean=False)
    districts = cu.clean_districts(districts, clean=False)
    disp = cu.clean_disp(disp, clean=False)
    transactions = cu.clean_transactions(trans, clean=False, op=False, k_symbol=True)
    cards = cu.clean_cards(cards, disp, clean=False)

    df = pd.merge(clients, disp, on='client_id', how="left")
    df = pd.merge(df, accounts, on='account_id', how="left", suffixes=('_client', '_account'))
    df = pd.merge(df, districts, left_on='client_district_id', right_on='code')
    df = pd.merge(df, transactions, how="left", on="account_id")
    df = pd.merge(df, cards, how="left", on="account_id")

    return df


############
# Algorithms
############

def clustering_agglomerative(df, n_clusters=2, linkage='average', n_components=2):
    df_copy = df.copy()

    # Create Dendrogram
    cmap = cm.rainbow(np.linspace(0, 1, 10))
    sch.set_link_color_palette([colors.rgb2hex(rgb[:3]) for rgb in cmap])
    _ = sch.dendrogram(sch.linkage(df_copy, method=linkage))
    plt.savefig('dendogram.jpg')
    plt.clf()

    # Scaling
    scaler = MinMaxScaler()
    scaler.fit(df_copy)
    X_scale = scaler.transform(df_copy)
    df_copy = pd.DataFrame(X_scale, columns=df_copy.columns)

    # Reduce dimensionality with PCA
    if len(df_copy.columns) > 3:
        # Analyse PCA
        if DEBUG:
            pca_analysis(df_copy)
        df_pca = pca(n_components, df_copy)

        if DEBUG:
            elbow_method(df_pca)

        # Apply Agglomerative Clustering
        hc = AgglomerativeClustering(n_clusters=n_clusters, affinity = 'euclidean', linkage = linkage)
        data = df_pca.values
        hc.fit_predict(data)
        print(f'Agglomerative Silhouette Score: \n{silhouette_score(df_pca, hc.labels_)}')

        # Plot results
        if n_components == 2:
            Scene = dict(xaxis = dict(title  = 'PC1'),yaxis = dict(title  = 'PC2'))
            trace = go.Scatter(x=df_pca.iloc[:,0], y=df_pca.iloc[:,1], mode='markers',marker=dict(color = hc.labels_, colorscale='rainbow', size = 6, line = dict(width = 0)))
        else: 
            Scene = dict(xaxis = dict(title  = 'PC1'),yaxis = dict(title  = 'PC2'), zaxis= dict(title  = 'PC3'))
            trace = go.Scatter3d(x=df_pca.iloc[:,0], y=df_pca.iloc[:,1], z=df_pca.iloc[:,2], mode='markers',marker=dict(color = hc.labels_, colorscale='rainbow', size = 6, line = dict(width = 0)))
        layout = go.Layout(scene = Scene, height = 1000,width = 1000)
        data = [trace]
        fig = go.Figure(data = data, layout = layout)
        fig.update_layout(title="Agglomerative Clusters",font=dict(size=15))
        fig.show()

    else:
        if DEBUG:
            elbow_method(df_copy)

        # Apply Agglomerative Clustering
        hc = AgglomerativeClustering(n_clusters=n_clusters, affinity = 'euclidean', linkage = linkage)
        data = df_copy.values
        hc.fit_predict(data)
        print(f'Agglomerative Silhouette Score: \n{silhouette_score(df_copy, hc.labels_)}')

        # Plot results
        if len(df_copy.columns) == 2:
            Scene = dict(xaxis = dict(title = df_copy.columns[0]),yaxis = dict(title  = df_copy.columns[1]))
            trace = go.Scatter(x=df_copy.iloc[:,0], y=df_copy.iloc[:,1], mode='markers',marker=dict(color = hc.labels_, colorscale='bluered', size = 8, line = dict(width = 0)))
        else: 
            Scene = dict(xaxis = dict(title = df_copy.columns[0]),yaxis = dict(title = df_copy.columns[1]), zaxis= dict(title = df_copy.columns[2]))
            trace = go.Scatter3d(x=df_copy.iloc[:,0], y=df_copy.iloc[:,1], z=df_copy.iloc[:,2], mode='markers',marker=dict(color = hc.labels_, colorscale='bluered', size = 8, line = dict(width = 0)))
        layout = go.Layout(scene = Scene, height = 1000,width = 1000)
        data = [trace]
        fig = go.Figure(data = data, layout = layout)
        fig.update_layout(title="Agglomerative Clusters",font=dict(size=15))
        fig.show()

def clustering_kmeans(df, n_clusters=3, init_method='k-means++', n_components=2):
    df_copy = df.copy()

    # Scaling
    scaler = MinMaxScaler()
    scaler.fit(df_copy)
    X_scale = scaler.transform(df_copy)
    df_copy = pd.DataFrame(X_scale, columns=df_copy.columns)

    # Reduce dimensionality with PCA
    if len(df_copy.columns) > 3:
        # Analyse PCA
        if DEBUG:
            pca_analysis(df)
        df_pca = pca(n_components, df)
        if DEBUG:
            elbow_method(df_pca)

        # Compute K-Means
        kmeans = KMeans(n_clusters=n_clusters, init=init_method)
        kmeans.fit_predict(df_pca)
                
        print("KMeans Clusters: \n", Counter(kmeans.labels_))
        print("KMeans Centers: \n", kmeans.cluster_centers_)
        print(f'KMeans Inertia: \n{kmeans.inertia_}')
        print(f'KMeans Silhouette Score: \n{silhouette_score(df_pca, kmeans.labels_)}')

        # Plot results
        if n_components == 2:
            Scene = dict(xaxis = dict(title  = 'PC1'),yaxis = dict(title  = 'PC2'))
            trace = go.Scatter(x=df_pca.iloc[:,0], y=df_pca.iloc[:,1], mode='markers',marker=dict(color = kmeans.labels_, colorscale='bluered', size = 10, line = dict(width = 0)))
        else: 
            Scene = dict(xaxis = dict(title  = 'PC1'),yaxis = dict(title  = 'PC2'), zaxis= dict(title  = 'PC3'))
            trace = go.Scatter3d(x=df_pca.iloc[:,0], y=df_pca.iloc[:,1], z=df_pca.iloc[:,2], mode='markers',marker=dict(color = kmeans.labels_, colorscale='rainbow', size = 10, line = dict(width = 0)))
        layout = go.Layout(scene = Scene, height = 1000,width = 1000)
        data = [trace]
        fig = go.Figure(data = data, layout = layout)
        fig.update_layout(title="KMeans Clusters",font=dict(size=16))
        fig.show()

        clusters_pca_scale = pd.concat([df_pca, pd.DataFrame({'cluster':kmeans.labels_})], axis=1)
        cluster_pca_profile = pd.merge(df_copy, clusters_pca_scale['cluster'], left_index=True, right_index=True)

        if DEBUG:
            print(cluster_pca_profile.groupby('cluster').mean())
            print(cluster_pca_profile.groupby('cluster').min())
            print(cluster_pca_profile.groupby('cluster').max())
            print(cluster_pca_profile.groupby('cluster').std())
            
    else:
        if DEBUG:
            elbow_method(df_copy)

        # Compute K-Means
        kmeans = KMeans(n_clusters=n_clusters, init=init_method)
        kmeans.fit_predict(df_copy)

        print("KMeans Clusters: \n", Counter(kmeans.labels_))
        print("KMeans Centers: \n", kmeans.cluster_centers_)
        print(f'KMeans Inertia: \n{kmeans.inertia_}')
        print(f'KMeans Silhouette Score: \n{silhouette_score(df_copy, kmeans.labels_)}')
        print()

        # Plot results
        if len(df_copy.columns) == 2:
            Scene = dict(xaxis = dict(title = df_copy.columns[0]),yaxis = dict(title  = df_copy.columns[1]))
            trace = go.Scatter(x=df_copy.iloc[:,0], y=df_copy.iloc[:,1], mode='markers',marker=dict(color=kmeans.labels_, colorscale='rainbow', size = 7, line = dict(width = 0)))
        else: 
            Scene = dict(xaxis = dict(title = df_copy.columns[0]),yaxis = dict(title = df_copy.columns[1]), zaxis= dict(title = df_copy.columns[2]))
            trace = go.Scatter3d(x=df_copy.iloc[:,0], y=df_copy.iloc[:,1], z=df_copy.iloc[:,2], mode='markers',marker=dict(color=kmeans.labels_, colorscale='rainbow', size = 7, line = dict(width = 0)))
        layout = go.Layout(scene = Scene, height = 1000,width = 1000)
        data = [trace]
        fig = go.Figure(data = data, layout = layout)
        fig.update_layout(title="KMeans Clusters", font=dict(size=16))
        fig.show()

        df_copy.insert(loc=0, column='cluster', value=kmeans.labels_)
        print(df_copy.describe())

def clustering_kmedoids(df, n_clusters=3, init_method='k-medoids++', n_components=2):
    df_copy = df.copy()
    
    scaler = MinMaxScaler()
    scaler.fit(df_copy)
    X_scale = scaler.transform(df_copy)
    df_copy = pd.DataFrame(X_scale, columns=df_copy.columns)
    
    # Reduce dimensionality with PCA
    if len(df_copy.columns) > 3:
        # Analyse PCA
        if DEBUG:
            pca_analysis(df)
        df_pca = pca(n_components, df_copy)
        if DEBUG:
            elbow_method(df_pca)

        # Apply K-Medoids
        kmedoids = KMedoids(n_clusters=n_clusters, method='pam', init=init_method)
        kmedoids.fit(df_pca)
 
        print("\nKMedoids Clusters: \n", Counter(kmedoids.labels_))
        print("KMedoids Centers: \n", kmedoids.cluster_centers_)
        print(f'KMedoids Inertia: \n{kmedoids.inertia_}')
        print(f'KMedoids Silhouette Score: \n{silhouette_score(df_pca, kmedoids.labels_)}')

        # Plot results
        if n_components == 2:
            Scene = dict(xaxis = dict(title  = 'PC1'),yaxis = dict(title  = 'PC2'))
            trace = go.Scatter(x=df_pca.iloc[:,0], y=df_pca.iloc[:,1], mode='markers',marker=dict(color = kmedoids.labels_, colorscale='rainbow', size = 6, line = dict(width = 0)))
        else: 
            Scene = dict(xaxis = dict(title  = 'PC1'),yaxis = dict(title  = 'PC2'), zaxis= dict(title  = 'PC3'))
            trace = go.Scatter3d(x=df_pca.iloc[:,0], y=df_pca.iloc[:,1], z=df_pca.iloc[:,2], mode='markers',marker=dict(color = kmedoids.labels_, colorscale='rainbow', size = 6, line = dict(width = 0)))
        layout = go.Layout(scene = Scene, height = 1000,width = 1000)
        data = [trace]
        fig = go.Figure(data = data, layout = layout)
        fig.update_layout(title="KMedoids Clusters",font=dict(size=16))
        fig.show()
    else:
        if DEBUG:
            elbow_method(df_copy)

        # Apply K-Medoids
        kmedoids = KMedoids(n_clusters=n_clusters, method='pam', init=init_method)
        kmedoids.fit_predict(df_copy)

        print("KMedoids Clusters: \n", Counter(kmedoids.labels_))
        print("KMedoids Centers: \n", kmedoids.cluster_centers_)
        print(f'KMedoids Inertia: {kmedoids.inertia_}')
        print(f'KMedoids Silhouette Score: {silhouette_score(df_copy, kmedoids.labels_)}')

        # Plot results
        if len(df_copy.columns) == 2:
            Scene = dict(xaxis = dict(title = df_copy.columns[0]),yaxis = dict(title  = df_copy.columns[1]))
            trace = go.Scatter(x=df_copy.iloc[:,0], y=df_copy.iloc[:,1], mode='markers',marker=dict(color=kmedoids.labels_, colorscale='rainbow', size = 7, line = dict(width = 0)))
        else: 
            Scene = dict(xaxis = dict(title = df_copy.columns[0]),yaxis = dict(title = df_copy.columns[1]), zaxis= dict(title = df_copy.columns[2]))
            trace = go.Scatter3d(x=df_copy.iloc[:,0], y=df_copy.iloc[:,1], z=df_copy.iloc[:,2], mode='markers',marker=dict(color=kmedoids.labels_, colorscale='rainbow', size = 7, line = dict(width = 0)))
        layout = go.Layout(scene = Scene, height = 1000,width = 1000)
        data = [trace]
        fig = go.Figure(data = data, layout = layout)
        fig.update_layout(title="KMedoids Clusters", font=dict(size=16))
        fig.show()

def clustering_dbscan(df, eps=0.9, min_samples=4, n_components=2):
    df_copy = df.copy()

    # Scaling
    scaler = MinMaxScaler()
    scaler.fit(df_copy)
    X_scale = scaler.transform(df_copy)
    df_copy = pd.DataFrame(X_scale, columns=df_copy.columns)

    # Reduce dimensionality with PCA
    if len(df_copy.columns) > 3:
        # Analyse PCA
        if DEBUG:
            pca_analysis(df)
        df_pca = pca(n_components, df)
        if DEBUG:
            dbscan_param_tuning_silhouette(df_pca)

        # Apply DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(df_pca)
        labels = dbscan.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        print('Estimated number of clusters: %d' % n_clusters_)
        print('Estimated number of noise points: %d' % n_noise_)
        print("Silhouette Coefficient: %0.3f" % silhouette_score(df_pca, labels))

        # Plot results
        labels = dbscan.labels_
        if n_components == 2:
            Scene = dict(xaxis = dict(title  = 'PC1'),yaxis = dict(title  = 'PC2'))
            trace = go.Scatter(x=df_pca.iloc[:,0], y=df_pca.iloc[:,1], mode='markers',marker=dict(color = labels, colorscale='rainbow', size = 6, line = dict(width = 0)))
        else: 
            Scene = dict(xaxis = dict(title  = 'PC1'),yaxis = dict(title  = 'PC2'), zaxis= dict(title  = 'PC3'))
            trace = go.Scatter3d(x=df_pca.iloc[:,0], y=df_pca.iloc[:,1], z=df_pca.iloc[:,2], mode='markers',marker=dict(color = labels, colorscale='rainbow', size = 6, line = dict(width = 0)))
        layout = go.Layout(scene = Scene, height = 1000,width = 1000)
        data = [trace]
        fig = go.Figure(data = data, layout = layout)
        fig.update_layout(title="'DBSCAN Clusters Derived from PCA'", font=dict(size=12,))
        fig.show()

    else:
        if DEBUG:
            dbscan_param_tuning_silhouette(df_copy)

        # Apply DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(df_copy)
        labels = dbscan.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        print('Estimated number of clusters: %d' % n_clusters_)
        print('Estimated number of noise points: %d' % n_noise_)
        print("Silhouette Coefficient: %0.3f" % silhouette_score(df_copy, labels))

        # Plot results
        labels = dbscan.labels_
        if len(df_copy.columns) == 2:
            Scene = dict(xaxis = dict(title = df_copy.columns[0]),yaxis = dict(title  = df_copy.columns[1]))
            trace = go.Scatter(x=df_copy.iloc[:,0], y=df_copy.iloc[:,1], mode='markers',marker=dict(color = labels, colorscale='rainbow', size = 7, line = dict(width = 0)))
        else: 
            Scene = dict(xaxis = dict(title = df_copy.columns[0]),yaxis = dict(title = df_copy.columns[1]), zaxis= dict(title = df_copy.columns[2]))
            trace = go.Scatter3d(x=df_copy.iloc[:,0], y=df_copy.iloc[:,1], z=df_copy.iloc[:,2], mode='markers',marker=dict(color = labels, colorscale='rainbow', size = 7, line = dict(width = 0)))
        layout = go.Layout(scene = Scene, height = 1000,width = 1000)
        data = [trace]
        fig = go.Figure(data = data, layout = layout)
        fig.update_layout(title="'DBSCAN Clusters'", font=dict(size=12,))
        fig.show()


###########
# Clusters
###########

def clustering_economic():

    # Build dataframe
    df =  merge_datasets(db, False, True)
    df['age'] = df['birth_date'].apply(lambda x: calculate_age(x))
    df = extract_features(df)

    # CLUSTERING 1
    df1 = df[['avg_amount_credit', 'avg_amount_withdrawal', 'average_salary']]
    clustering_dbscan(df1, 0.2, 4)
    clustering_kmeans(df1)
    clustering_agglomerative(df1,3)

    # CLUSTERING 2
    df2 = df[['avg_amount_credit', 'average_salary', 'avg_balance']]
    clustering_agglomerative(df2, 4)
    clustering_kmedoids(df2, 2)
    clustering_kmeans(df2, 3)
    clustering_kmedoids(df2, 3)

    # CLUSTERING 3
    df3 = df[['min_balance', 'avg_balance', 'avg_amount_withdrawal', 'std_balance', 'avg_amount_total', 'credit_ratio']]
    clustering_kmeans(df3, 4, 'k-means++', 2)
    clustering_kmedoids(df3, 4, 'k-medoids++', 2)

    # CLUSTERING 4 - for all clients that have transactions
    df4 =  merge_transactions_clients(db)
    df4 = df4[['avg_amount_credit', 'average_salary', 'avg_amount_withdrawal', 'avg_balance']]
    df4.dropna(inplace=True)
    clustering_dbscan(df4, 0.2, 2, 2)
    clustering_kmeans(df4, 3)
    clustering_kmedoids(df4, 3)


def clustering_demographic(dfs, df=None):
    if df is None:
        df = merge_transactions_clients(dfs)

    # CLUSTERING 1
    df1 = df[['avg_balance', 'average_salary', 'age_at_loan']]
    clustering_kmeans(df1)
    clustering_kmedoids(df1)

    # CLUSTERING 2
    # df2 = df[[ 'no._of_municipalities_with_inhabitants_<_499', 'no._of_municipalities_with_inhabitants_2000-9999', 'average_salary']]
    # df2.dropna(inplace=True)
    # clustering_kmeans(df2)
    # clustering_kmedoids(df2, 2)

    # CLUSTERING 3
    # df3 = df[['no._of_municipalities_with_inhabitants_2000-9999', 'avg_commited_crimes', 'ratio_entrepreneurs']]
    # clustering_kmeans(df3)