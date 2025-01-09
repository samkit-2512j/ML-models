import numpy as np

class PCAAutoencoder:
    def __init__(self, n_components):
        self.k = n_components

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        covariance_matrix = np.cov(X_centered,rowvar=False)
        
        e_val, e_vec = np.linalg.eigh(covariance_matrix)
        
        sorted_indices = np.argsort(-e_val)
        self.e_vec = e_vec[:, sorted_indices]
        self.e_val = e_val[sorted_indices]
        
        self.components = self.e_vec[:, :self.k]

    def encode(self, X):
        X = X - np.mean(X, axis=0)
        transformed_data = X @ self.e_vec[:, 0:self.k]
        return transformed_data
    
    def forward(self, transformed):
        return transformed @ self.e_vec[:, 0:self.k].T + self.mean
    
    def checkPCA(self, X, threshold = 0.05, p=False):
        X_transformed = self.encode(X)
        x_orig = self.forward(X_transformed)

        '''Euclidean norm error'''
        # recon_error = np.sqrt(np.sum((X - x_orig)**2))  
        '''MSE'''
        recon_error=np.mean((X-x_orig)**2) 
        if p:
            print('MSE Loss:', recon_error)
        if recon_error>threshold:
            return recon_error
        return recon_error