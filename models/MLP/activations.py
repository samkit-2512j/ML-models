import numpy as np

class sigmoid:
    def __init__(self):
        pass

    def func(self, x):
        return 1 / (1 + np.exp(-x))
    
    def derivative(self,x):
        a = self.func(x)
        return a * (1 - a)
    
    def name(self):
        return "sigmoid"
    

class relu:
    def __init__(Self):
        pass

    def func(self,x):
        return np.maximum(0, x)
    
    def derivative(self,x):
        return np.where(x > 0, 1, 0)
    
    def name(self):
        return "relu"
    

class tanh:
    def __init__(self):
        pass

    def func(self,x):
        return np.tanh(x)
    
    def derivative(self,x):
        return 1 - np.tanh(x)**2
    
    def name(self):
        return "tanh"
    

class linear:
    def __init__(self):
        pass

    def func(self,x):
        return x
    
    def derivative(self,x):
        return np.ones_like(x)
    
    def name(self):
        return "linear"
    
class softmax:
    def __init__(self):
        pass

    def func(self,x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def derivative(self,x):
        a = self.func(x)
        return a * (1 - a)
    
    def name(self):
        return "softmax"