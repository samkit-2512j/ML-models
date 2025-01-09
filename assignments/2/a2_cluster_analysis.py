import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from models.PCA.pca import Pca
from models.k_means.kmeans import KMeans
from models.gmm.gmm import GMM
from sklearn.mixture import GaussianMixture

def cluster_analysis():
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

    '''KMeans Cluster Analysis'''

    kmeans1=5
    kmeans2=5
    kmeans3=6

    k1=KMeans(kmeans1)
    k2=KMeans(kmeans2)
    k3=KMeans(kmeans3)

    k1.fit(X)
    k2.fit(X)
    k3.fit(X)

    clus1=k1.assign_clusters(X)
    clusters1={}

    for i in range(kmeans1):
        clusters1[i]=[]
        for j,k in enumerate(clus1):
            if k==i:
                clusters1[i].append(words[j])

    with open("kmeans.txt","w") as f:
        pass

    with open("kmeans.txt", "a") as f:
        f.write(f"kmeans1 \n")
        for key, value in clusters1.items():
            f.write(f"{key} : {', '.join(value)}\n\n")

    clus2=k2.assign_clusters(X)
    clusters2={}

    for i in range(kmeans2):
        clusters2[i]=[]
        for j,k in enumerate(clus2):
            if k==i:
                clusters2[i].append(words[j])

    with open("kmeans.txt", "a") as f:
        f.write(f"k2 \n")
        for key, value in clusters2.items():
            f.write(f"{key} : {', '.join(value)}\n\n")

    clus3=k3.assign_clusters(X)
    clusters3={}

    for i in range(kmeans3):
        clusters3[i]=[]
        for j,k in enumerate(clus3):
            if k==i:
                clusters3[i].append(words[j])

    with open("kmeans.txt", "a") as f:
        f.write(f"kmeans3 \n")
        for key, value in clusters3.items():
            f.write(f"{key} : {', '.join(value)}\n\n")


    '''GMM Cluster Analysis'''

    kgmm1=1
    k2=5
    kgmm3=3

    gm1=GaussianMixture(n_components=kgmm1,random_state=0)
    gm2=GaussianMixture(n_components=k2,random_state=0)
    gm3=GaussianMixture(n_components=kgmm3,random_state=0)

    gm1.fit(X)
    gm2.fit(X)
    gm3.fit(X)

    clus1=gm1.predict(X)
    clusters1={}

    for i in range(kgmm1):
        clusters1[i]=[]
        for j,k in enumerate(clus1):
            if k==i:
                clusters1[i].append(words[j])

    with open("gmm.txt","w") as f:
        pass

    with open("gmm.txt", "a") as f:
        f.write(f"kgmm1 \n")
        for key, value in clusters1.items():
            f.write(f"{key} : {', '.join(value)}\n\n")

    clus2=gm2.predict(X)
    clusters2={}

    for i in range(k2):
        clusters2[i]=[]
        for j,k in enumerate(clus2):
            if k==i:
                clusters2[i].append(words[j])

    with open("gmm.txt", "a") as f:
        f.write(f"k2 \n")
        for key, value in clusters2.items():
            f.write(f"{key} : {', '.join(value)}\n\n")

    clus3=gm3.predict(X)
    clusters3={}

    for i in range(kgmm3):
        clusters3[i]=[]
        for j,k in enumerate(clus3):
            if k==i:
                clusters3[i].append(words[j])

    with open("gmm.txt", "a") as f:
        f.write(f"kgmm3 \n")
        for key, value in clusters3.items():
            f.write(f"{key} : {', '.join(value)}\n\n")

    '''Kgmm vs Kmeans'''

    kgmm=k2
    kmeans=kmeans3

    km=KMeans(kmeans)
    gm=GaussianMixture(n_components=kgmm,random_state=0)
    gm.fit(X)

    km.fit(X)
    clus1=km.assign_clusters(X)
    clusters1={}

    for i in range(kmeans):
        clusters1[i]=[]
        for j,k in enumerate(clus1):
            if k==i:
                clusters1[i].append(words[j])

    with open("gmm_vs_kmeans.txt","w") as f:
        pass

    with open("gmm_vs_kmeans.txt", "a") as f:
        f.write(f"kmeans \n")
        for key, value in clusters1.items():
            f.write(f"{key} : {', '.join(value)}\n\n")

    clus2=gm.predict(X)
    clusters2={}

    for i in range(kgmm):
        clusters2[i]=[]
        for j,k in enumerate(clus2):
            if k==i:
                clusters2[i].append(words[j])

    with open("gmm_vs_kmeans.txt", "a") as f:
        f.write(f"kgmm \n")
        for key, value in clusters2.items():
            f.write(f"{key} : {', '.join(value)}\n\n")

    print("Clusters have been written in the text files.")