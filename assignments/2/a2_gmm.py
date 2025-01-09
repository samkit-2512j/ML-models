import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from models.gmm.gmm import GMM
from models.PCA.pca import Pca

from sklearn.mixture import GaussianMixture

def gmm_analysis():
    df = pd.read_feather('../../data/external/word-embeddings.feather')

    # print(df['words'].unique().shape)

    words=df['words'].values
    x=df['vit'].to_numpy()
    X=[]

    for i in range(x.shape[0]):
        res = []
        for j in x[i]:
            res.append(j)
        X.append(res)
        
    X = np.asarray(X)


    pc=Pca(5)
    pc.fit(X)
    x_5d=pc.transform(X)

    '''4.2'''
    gm=GaussianMixture(n_components=5,random_state=0)
    gm.fit(X)

    print("Means of the Gaussians in 512D data")
    print(gm.means_)



    def aic_bic_plot(X,name):
        aic_values, bic_values=[], []
        for n in range(1,11):
            gmm_plot = GaussianMixture(n_components=n)
            gmm_plot.fit(X)
            # gmm_plot=GMM(n_components=n)
            # gmm_plot.fit(X)
            
            aic_values.append(gmm_plot.aic(X))
            bic_values.append(gmm_plot.bic(X))

        # Plotting AIC and BIC
        plt.figure(figsize=(8, 6))
        plt.plot(np.linspace(1,10,10), aic_values, label='AIC', marker='o')
        plt.plot(np.linspace(1,10,10), bic_values, label='BIC', marker='o')

        # Labeling the plot
        plt.title('AIC and BIC vs Number of Components in GMM')
        plt.xlabel('Number of Components')
        plt.ylabel('Information Criterion')
        plt.legend()
        # plt.savefig(f'./figures/{name}.jpg')
        plt.show()

    aic_bic_plot(X,'aic_bic_gmm_512d')


    '''6.3'''
    pc=Pca(2)
    pc.fit(X)
    x_2d=pc.transform(X)

    k2=5
    gmm_class=GMM(k2)
    gmm_class.fit(x_2d)

    # plot_2d_clusters()

    print("GMM on 2D data with k2 clusters")

    print("Means of the Gaussians:")
    print(gmm_class.means)
    print("Covariance of the Gaussians:")
    print(gmm_class.covariances)
    print("Weights of the Gaussians:")
    print(gmm_class.weights)

    '''6.4'''

    aic_bic_plot(x_2d,'aic_bic_gmm_2d')

    kgmm3=3

    gmm_class=GMM(kgmm3)
    gmm_class.fit(x_5d)

    print("GMM on kgmm3 number of clusters")

    print("Means of the Gaussians:")
    print(gmm_class.means)
    print("Covariance of the Gaussians:")
    print(gmm_class.covariances)
    print("Weights of the Gaussians:")
    print(gmm_class.weights)