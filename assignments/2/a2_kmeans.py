import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from models.k_means.kmeans import KMeans
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def kmeans_analysis():

    df = pd.read_feather('../../data/external/word-embeddings.feather')

    words=df['words'].values
    x=df['vit'].to_numpy()
    X=[]

    for i in range(x.shape[0]):
        res = []
        for j in x[i]:
            res.append(j)
        X.append(res)
        
    X = np.asarray(X)

    k=3
    km=KMeans(k=k,max_iters=100)
    km.fit(X)

    wcss=[]
    ks=[]

    for k in range(1,10):
        km=KMeans(k=k,max_iters=100)
        km.fit(X)
        ks.append(k)
        wcss.append(km.getCost(X))
        del km

    plt.plot(ks, wcss, marker='o') 
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('WCSS')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True) 
    # plt.savefig('./figures/elbow_plot_512dim_kmeans.jpg')
    plt.show()

    kmeans1=KMeans(k=5,max_iters=1000)
    kmeans1.fit(X)

    print("Clusters of the 200 words for kmeans1 are:")
    print(kmeans1.assign_clusters(X))