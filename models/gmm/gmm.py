import numpy as np
from scipy.stats import multivariate_normal

class GMM:
    def __init__(self, n_components, n_iter=100):
        self.n_components = n_components
        self.n_iter = n_iter
        self.weights = None
        self.means = None
        self.covariances = None
        
    def fit(self, X):
        sample_count, feature_dim = X.shape
        
        self.weights = np.full(self.n_components, 1.0 / self.n_components)
        random_indices = np.random.choice(sample_count, self.n_components, replace=False)
        self.means = X[random_indices]
        self.covariances = [np.identity(feature_dim) for _ in range(self.n_components)]
        
        prev_log_likelihood = float('-inf')
        
        for iteration in range(self.n_iter):
            # E-step
            responsibilities = self.e_step(X)
            
            # M-step
            self.m_step(X, responsibilities)
            
            #log-likelihood
            curr_log_likelihood = self.getLikelihood(X)
    
    def e_step(self, X):
        responsibility_matrix = np.zeros((X.shape[0], self.n_components))
        
        for component in range(self.n_components):
            responsibility_matrix[:, component] = self.weights[component] * multivariate_normal.pdf(
                X, mean=self.means[component], cov=self.covariances[component]
            )
        
        row_sums = responsibility_matrix.sum(axis=1, keepdims=True)
        responsibility_matrix /= row_sums
        return responsibility_matrix
    
    def m_step(self, X, responsibilities):
        component_sample_counts = responsibilities.sum(axis=0)

        '''Used LLM to obtain the following piece of code'''
        # Prompt: Optimise the maximisation step of GMM given in the code (provided my code)
        
        self.weights = component_sample_counts / X.shape[0]
        self.means = np.dot(responsibilities.T, X) / component_sample_counts[:, np.newaxis]
        
        for component in range(self.n_components):
            deviation = X - self.means[component]
            self.covariances[component] = np.dot(responsibilities[:, component] * deviation.T, deviation) / component_sample_counts[component]
    
    def getParams(self):
        return self.weights, self.means, self.covariances
    
    def getMembership(self, X):
        return self.e_step(X)
    
    def getLikelihood(self, X):
        likelihood_matrix = np.zeros((X.shape[0], self.n_components))
        
        for component in range(self.n_components):
            likelihood_matrix[:, component] = self.weights[component] * multivariate_normal.pdf(
                X, mean=self.means[component], cov=self.covariances[component]
            )
        
        return np.sum(np.log(np.sum(likelihood_matrix, axis=1)))
    
    def AIC(self, X):
        sample_count, feature_dim = X.shape
        param_count = self.n_components * (1 + feature_dim + feature_dim * (feature_dim + 1) / 2) - 1
        log_likelihood = self.getLikelihood(X)
        return 2 * param_count - 2 * log_likelihood

    def BIC(self, X):
        sample_count, feature_dim = X.shape
        param_count = self.n_components * (1 + feature_dim + feature_dim * (feature_dim + 1) / 2) - 1
        log_likelihood = self.getLikelihood(X)
        return np.log(sample_count) * param_count - 2 * log_likelihood