import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class Pca:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None 
        self.mean = None 

    def scree_plot(self, X, save=False, name=None):
        e_vec, e_val=self.fit(X,sp=True)
        total_variance = np.sum(e_val)
        self.explained_variance_ratio = e_val[:20] / total_variance

        plt.figure(figsize=(8, 6))
        plt.plot(np.arange(1, len(self.explained_variance_ratio) + 1), self.explained_variance_ratio, 'bo-', linewidth=2)
        plt.title("Scree Plot")
        plt.xlabel("Principal Component")
        plt.ylabel("Explained Variance Ratio")
        plt.grid(True)
        if save:
            plt.savefig(f'../../assignments/2/figures/{name}.jpg')
        plt.show()


    def fit(self, X, sp=False):
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        covariance_matrix = np.cov(X_centered,rowvar=False)
        
        e_val, e_vec = np.linalg.eigh(covariance_matrix)
        
        sorted_indices = np.argsort(-e_val)
        e_vec = e_vec[:, sorted_indices]
        e_val = e_val[sorted_indices]
        
        self.components = e_vec[:, :self.n_components]
        if sp:
            return e_vec,e_val
        
    
    def transform(self, X):
        
        X_centered = X - self.mean
        
        return np.dot(X_centered, self.components)
    
    def checkPCA(self, X):
        X_transformed = self.transform(X)
        x_orig = X_transformed @ self.components.T + self.mean

        '''Euclidean norm error'''
        # recon_error = np.sqrt(np.sum((X - x_orig)**2))  
        '''MSE'''
        recon_error=np.mean((X-x_orig)**2) 
        if recon_error>0.04:
            return recon_error,False
        return recon_error,True
