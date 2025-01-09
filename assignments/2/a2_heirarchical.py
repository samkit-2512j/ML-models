import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from models.PCA.pca import Pca
from models.k_means.kmeans import KMeans
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster.hierarchy import linkage,dendrogram, fcluster
import itertools

def heirarchical_analysis():
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

    linkages=['single','complete','average','ward']
    metrics=['euclidean','cityblock']

    for l,m in itertools.product(linkages,metrics):
        if l=='ward' and m=='cityblock':
            continue

        linkage_mat=linkage(X,method=l,metric=m)

        plt.figure(figsize=(20,20))
        dendrogram(linkage_mat,labels=np.linspace(1, 200, 200).astype('int'), leaf_rotation=90, leaf_font_size=10)
        plt.title(f"Hierarchical clustering Dendogram with {l} and {m}")
        plt.xlabel('Sample')
        plt.ylabel('Distance')
        plt.savefig(f'./figures/hieararchical_{l}_{m}.jpg')
        plt.show()

    kbest1=6

    linkage_mat=linkage(X,method='ward',metric='euclidean')
    clusters=fcluster(linkage_mat,kbest1,criterion='maxclust')

    with open("heirarchical.txt","w") as f:
        pass

    clusters1={}

    for i in range(1,kbest1+1):
        clusters1[i]=[]
        for j,k in enumerate(clusters):
            if k==i:
                clusters1[i].append(words[j])

    with open("heirarchical.txt","w") as f:
        pass

    with open("heirarchical.txt", "a") as f:
        f.write(f"kbest1 \n")
        for key, value in clusters1.items():
            f.write(f"{key} : {', '.join(value)}\n\n")

    clusters1={}

    kbest2=5
    linkage_mat=linkage(X,method='ward',metric='euclidean')
    clusters=fcluster(linkage_mat,kbest2,criterion='maxclust')

    for i in range(1,kbest2+1):
        clusters1[i]=[]
        for j,k in enumerate(clusters):
            if k==i:
                clusters1[i].append(words[j])

    with open("heirarchical.txt", "a") as f:
        f.write(f"kbest2 \n")
        for key, value in clusters1.items():
            f.write(f"{key} : {', '.join(value)}\n\n")

    print("Analysis of observations is in report.")