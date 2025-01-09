import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

class KDE:
    
    def __init__(self, kernel='gaussian', bandwidth='silverman'):
        self.kernel_type = kernel
        self.bandwidth = bandwidth
        self.data = None
        self.n_samples = None
        self.n_dimensions = None
        
    def kernel_function(self, x):
        if self.kernel_type == 'gaussian':
            return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x ** 2)
        
        elif self.kernel_type == 'box':
            return np.where(np.abs(x) <= 1, 0.5, 0)
        
        elif self.kernel_type == 'triangular':
            return np.where(np.abs(x) <= 1, 1 - np.abs(x), 0)
        
    def silverman_bandwidth(self, data):
        n = len(data)
        d = data.shape[1]
        std = np.std(data, axis=0)
        return ((4 / (d + 2)) ** (1 / (d + 4))) * std * (n ** (-1 / (d + 4)))
    
    def scott_bandwidth(self, data):
        n = len(data)
        d = data.shape[1]
        std = np.std(data, axis=0)
        return std * (n ** (-1 / (d + 4)))
    
    def fit(self, data):
        self.data = np.array(data)
        self.n_samples, self.n_dimensions = self.data.shape
        
        if self.bandwidth == 'scott':
            self.bandwidth = self.scott_bandwidth(self.data)
        else:
            self.bandwidth = self.silverman_bandwidth(self.data)
            
        self.bandwidth = np.array(self.bandwidth)
    
    def predict(self, x):
            
        x = np.array(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
            
        n_points = x.shape[0]
        density = np.zeros(n_points)
        
        for i in range(n_points):
            scaled_dist = (x[i] - self.data) / self.bandwidth
            
            kernel_values = self.kernel_function(scaled_dist)
            
            point_density = np.prod(kernel_values, axis=1)
            
            density[i] = np.mean(point_density) / np.prod(self.bandwidth)
            
        return density
    
    def visualize(self, density = None):
        
        if density is not None:
            plt.figure(figsize=(8, 6))
            plt.scatter(self.data[:, 0], self.data[:, 1], c=density, cmap='viridis', s=8, edgecolor='k')
            plt.colorbar(label='Density')
            plt.title('2D KDE Estimation with Gaussian Kernel', fontsize=16)
            plt.xlabel('X-axis', fontsize=14)
            plt.ylabel('Y-axis', fontsize=14)
            plt.show()

        x_min, x_max = self.data[:, 0].min() - 1, self.data[:, 0].max() + 1
        y_min, y_max = self.data[:, 1].min() - 1, self.data[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                            np.linspace(y_min, y_max, 100))

        grid_density = self.predict(np.c_[xx.ravel(), yy.ravel()])
        grid_density = grid_density.reshape(xx.shape)
        if density is not None:
            plt.figure(figsize=(8, 6))
            # plt.scatter(self.data[:, 0], self.data[:, 1], c=density, cmap='viridis', s=8, edgecolor='k')
            plt.contourf(xx, yy, grid_density, 20, cmap='viridis', alpha=0.5)
            plt.colorbar(label='Density')
            plt.title('2D KDE Estimation with Gaussian Kernel', fontsize=16)
            plt.xlabel('X-axis', fontsize=14)
            plt.ylabel('Y-axis', fontsize=14)
            plt.show()
