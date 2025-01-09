import numpy as np
from collections import Counter
from numpy.linalg import norm
from tqdm import tqdm

class KNN:
    def __init__(self, k=5, dist_metric='euclidean',p=0.75):
        self.k = k
        self.dist_metric=dist_metric
        self.distances={
            'euclidean' : self.euclidean_distance,
            'manhattan' : self.manhattan_distance,
            'cosine_similarity' : self.cosine_similarity_distance,
            'minkowski' : self.minkowski_distance,
        }
        self.p=p

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        # y_pred = np.array([self._predict(x) for x in X])
        y_pred = np.array([self._predict(x) for x in tqdm(X, desc="Predicting", unit="sample")])
        return y_pred

    def _predict(self, x):
        dist_func=self.distances[self.dist_metric]
        distances = dist_func(self.X_train,x)
        
        # k closest indices
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_indices]
        
        # Return the most common genre
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
    
    def euclidean_distance(self,x_train, x):
        return np.sqrt(np.sum((x_train - x) ** 2, axis=1))

    def manhattan_distance(self,x_train, x):
        return np.sum(np.abs(x_train - x), axis=1)

    def cosine_similarity_distance(self,x_train, x):
        return np.sum(x_train * x, axis=1) / (norm(x_train, axis=-1) * norm(x, axis=-1))

    def minkowski_distance(self,x_train, x):
        return np.sum(np.abs(x_train - x) ** self.p, axis=1) ** (1 / self.p)
