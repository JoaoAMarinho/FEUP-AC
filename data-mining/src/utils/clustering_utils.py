from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from collections import Counter
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import utils.clean_utils as cu
from IPython.display import Markdown, display


#######
# Utils
#######

def pca(df, cols, table):
    """
    Splits data into N dimensions. Adapted from https://andrewmourcos.github.io/blog/2019/06/06/PCA.html
    """

    # Scale data
    Sc = StandardScaler()
    X = Sc.fit_transform(df[cols])

    # Divide into components
    pca = PCA(n_components=2, random_state=0)
    components = pca.fit_transform(X)
    df_comp = pd.concat([pd.DataFrame(data=components, columns=['Comp1', 'Comp2']), df[['status']]],axis = 1)
    sns.scatterplot(data=df_comp, hue='status', x='Comp1', y='Comp2')
    plt.title(f"PCA {table} info with status")
    plt.show()
    return df_comp[['Comp1', 'Comp2']]

def elbow_method(components):
    """
    Create 10 K-Mean models while varying the number of clusters (k)
    To get best k
    """

    inertias = []
    for k in range(1,10):
        kmeans = KMeans(n_clusters=k, random_state=0)
        # Fit model to samples
        kmeans.fit(components.iloc[:,:3])
        # Append the inertia to the list of inertias
        inertias.append(kmeans.inertia_)

    sns.lineplot(x=range(1,10), y=inertias, marker='o')
    plt.title("Evolution of Inertia with number of clusters")
    plt.xlabel("Number of clusters k")
    plt.ylabel("Inertia")
    plt.show()


##########
# Features
##########

def merge_transactions_clients(dfs):
    [ accounts, disp, clients, districts, cards, trans ] = dfs

    clients = cu.clean_clients(clients)
    accounts = cu.clean_accounts(accounts)
    districts = cu.clean_districts(districts)
    disp = cu.clean_disp(disp)
    transactions = cu.clean_transactions(trans, op=False, k_symbol=True)
    cards = cu.clean_cards(cards, disp)

    df = pd.merge(clients, disp, on='client_id', how='left')
    df = pd.merge(df, accounts, on='account_id', how='left', suffixes=('_client', '_account'))
    df = pd.merge(df, districts, left_on='client_district_id', right_on='code')
    df = pd.merge(df, transactions, how='left', on='account_id')
    df = pd.merge(df, cards, how='left', on='account_id')
    df['age'] = df['birth_date'].apply(lambda x: cu.calculate_age(x))
    df.dropna(inplace=True)

    return df


############
# Algorithms
############

def clustering_kmeans(df, components, N, title):
    # Split into clusters
    kmeans = KMeans(n_clusters=N, random_state=0)
    kmeans.fit(components.iloc[:,:2])

    labels = kmeans.predict(components.iloc[:,:2])

    sns.scatterplot(data=components, x='Comp1', y='Comp2', hue=labels, palette=sns.color_palette('tab10', N))
    plt.title(title)
    plt.show()

    display(Markdown(f"**KMeans Clusters:**\n- {Counter(kmeans.labels_)}"))
    display(Markdown(f"**KMeans Centers:**\n- {kmeans.cluster_centers_}"))
    display(Markdown(f"**KMeans Inertia:**\n- {kmeans.inertia_}"))
    display(Markdown(f"**KMeans Silhouette Score:**\n- {silhouette_score(components, kmeans.labels_)}"))

    clusters_pca_scale = pd.concat([components, pd.DataFrame({'cluster': kmeans.labels_})], axis=1)
    cluster_pca_profile = pd.merge(df, clusters_pca_scale['cluster'], left_index=True, right_index=True)

    return cluster_pca_profile.groupby('cluster').mean()
    # print(cluster_pca_profile.groupby('cluster').min())
    # print(cluster_pca_profile.groupby('cluster').max())
    # print(cluster_pca_profile.groupby('cluster').std())


###########
# Clusters
###########

def clustering_clients(df):
    client_columns = ['gender', 'age', 'average_salary', 'criminality_growth', 'avg_balance']

    components = pca(df, client_columns, 'Clients')
    elbow_method(components)
    return clustering_kmeans(df[client_columns], components, 3, "Client's clusters")

def clustering_economic(df):
    economic_columns = ['avg_amount_credit', 'avg_amount_withdrawal', 'average_salary']
    economic_columns2 = ['min_balance', 'avg_balance', 'avg_amount_withdrawal', 'std_balance', 'avg_amount_total', 'credit_ratio']

    # CLUSTERING - for all clients that have transactions
    # df4 =  merge_transactions_clients(db)
    # df4.dropna(inplace=True)
    # clustering_kmeans(df4, 3)


def clustering_demographic(df):
    demographic_columns = ['gender', 'age', 'average_salary', 'criminality_growth', 'avg_balance']
