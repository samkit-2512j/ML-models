import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from models.PCA.pca import Pca
from models.k_means.kmeans import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

def pca_analysis():
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

    pca_2d=Pca(2)
    pca_3d=Pca(3)

    pca_2d.fit(X)
    X_2d=pca_2d.transform(X)

    pca_3d.fit(X)
    X_3d=pca_3d.transform(X)

    def plot_2D(X):
        plt.figure(figsize=(8, 6))
        plt.scatter(X[:, 0], X[:, 1], c='blue', edgecolor='k', s=50)
        plt.title("PCA Reduced Data (2D)")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)
        plt.savefig('./figures/pca_2D.jpg')
        plt.show()

    def plot_3D(X):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c='blue', edgecolor='k', s=50)
        
        ax.set_title("PCA Reduced Data (3D)")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        plt.savefig('./figures/pca_3d.jpg')
        plt.show()

    '''Plotting the 2D and 3D PC's'''

    plot_2D(X_2d)


    plot_3D(X_3d)

    print("chackPCA for 2D, returns reconstruction error and validity:")
    print(pca_2d.checkPCA(X))
    print("chackPCA for 3D, returns reconstruction error and validity:")
    print(pca_3d.checkPCA(X))

    '''Elbow Graph for 2D data'''
    def elbow_plot(X,dim):
        ks,wcss=[],[]

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
        plt.savefig(f'./figures/elbow_plot_{dim}dim_kmeans.jpg')
        plt.show()

    elbow_plot(X_2d,dim=2)

    km_2d=KMeans(k=5)
    km_2d.fit(X_2d)
    clusters=km_2d.assign_clusters(X_2d)

    def plot_2d_clusters(X, clusters):
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']
        
        plt.figure()
        for i in range(max(clusters)+1):
            plt.scatter(X[clusters == i, 0], X[clusters == i, 1], 
                        color=colors[i], label=f'Cluster {i+1}')
        
        plt.xlabel('X axis')
        plt.ylabel('Y axis')
        plt.title('2D Cluster Plot')
        plt.legend()
        plt.savefig('./figures/k2_2d_kmeans.jpg')
        plt.show()

    plot_2d_clusters(X_2d,clusters)

    '''Scree plot'''

    pca_2d.scree_plot(X)

    best_pca=Pca(n_components=5)
    best_pca.fit(X)
    best_x=best_pca.transform(X)


    # print(best_pca.checkPCA(X))

    '''Kmeans3 clustering'''
    kmeans3=6
    km=KMeans(k=kmeans3)
    km.fit(best_x)
    clusters=km.assign_clusters(best_x)

    def plot_3d_clusters(X, clusters):
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in range(max(clusters)+1):
            ax.scatter(X[clusters == i, 0], X[clusters == i, 1], X[clusters == i, 2], 
                    color=colors[i], label=f'Cluster {i+1}')
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        ax.set_title('3D Cluster Plot')
        ax.legend()
        plt.savefig('./figures/kmeans3_clustering.jpg')
        plt.show()

    plot_3d_clusters(best_x,clusters)

