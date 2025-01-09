import numpy as np

class KMeans:
    def __init__(self, k, max_iters=100, tol=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None

    def kmeans_plus_plus_init(self, X):
        '''Used LLM for this function to make kmeans initialization more accurate'''
        #prompt: (given the kmeans code) write a function to use kmeans++ to initialize the model
        n_samples, n_features = X.shape
        centroids = []

        random_index = np.random.choice(n_samples)
        centroids.append(X[random_index])

        for p in range(1, self.k):
            distances = np.min([np.linalg.norm(X - centroid, axis=1)**2 for centroid in centroids], axis=0)

            probabilities = distances / np.sum(distances)

            new_centroid_index = np.random.choice(n_samples, p=probabilities)
            centroids.append(X[new_centroid_index])

        return np.array(centroids)

    def fit(self, X):

        self.centroids = self.kmeans_plus_plus_init(X)

        for _ in range(self.max_iters):
            labels = self.assign_clusters(X)
            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.k)])

            if np.linalg.norm(new_centroids - self.centroids) < self.tol:
                break

            self.centroids = new_centroids

    def predict(self, X):
        return self.assign_clusters(X)

    def getCost(self, X):
        labels = self.predict(X)
        wcss = 0
        for i in range(self.k):
            cluster_points = X[labels == i]
            wcss += np.sum((cluster_points - self.centroids[i])**2)
        return wcss

    def assign_clusters(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)


