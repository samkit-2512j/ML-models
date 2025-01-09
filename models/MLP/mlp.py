import numpy as np
from typing import List, Callable, Literal
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from models.MLP.activations import *
from tqdm import tqdm
import wandb
from performance_measures.performance_measures import PerfMeeasures

class MLP:
    def __init__(self, hidden_layers, learning_rate= 0.01, 
                 activation= sigmoid,
                 optimizer= 'sgd',
                 batch_size= 32,
                 epochs= 100,
                 patience=10,
                 early_stopping=False,
                 model = 'classifier',
                 loss_func = 'bce',
                 logit = False):
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.activation = activation()
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs
        self.weights = []
        self.biases = []
        self.loss_history=[]
        self.val_loss_history = []
        self.patience = patience
        self.early_stopping = early_stopping
        self.model = model
        self.logit=logit
        if model == 'classifier':
            self.act=softmax()
        elif model == 'multi-label':
            self.act=sigmoid()
        else:
            self.act=linear()
        if logit:
            self.sig=sigmoid()
        self.loss_func=loss_func


    def init_params(self, input_size, output_size):
        self.layers = [input_size] + self.hidden_layers + [output_size]
        print(self.layers)
        for i in range(1, len(self.layers)):
            # self.weights.append(np.random.randn(self.layers[i-1], self.layers[i]) * 0.01)
            # self.biases.append(np.zeros((1, self.layers[i])))
            self.weights.append(np.random.randn(self.layers[i-1], self.layers[i]) * np.sqrt(2. / self.layers[i-1]))
            self.biases.append(np.zeros((1, self.layers[i])))

    def mse_loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    def categorical_cross_entropy(self, y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        ret = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
        return ret
    
    def binary_cross_entropy(self, y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        ret = -np.mean(np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred),axis=1))
        return ret

    def forward_prop(self, X):
        activations = [X]
        zs=[]
        for i in range(len(self.weights) - 1):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            zs.append(z)
            a = self.activation.func(z)
            activations.append(a)

        z = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        zs.append(z)
        a=self.act.func(z)
        activations.append(a)
        return activations, zs

    def back_prop(self, X, y):
        m = X.shape[0] # m=1 for sgd
        activations,zs = self.forward_prop(X)
        
        dW = [np.zeros_like(w) for w in self.weights]
        db = [np.zeros_like(b) for b in self.biases]
        
        delta=0
        if self.loss_func == 'mse':
            if len(y.shape)<=1:
                delta = 2 * (activations[-1] - y.reshape(-1,1))
            else:
                delta = 2 * (activations[-1] - y)

        else:
            if self.logit:
                delta = (activations[-1] - y.reshape(-1,1))
            else:
                delta = activations[-1] - y


        dW[-1] = np.dot(activations[-2].T, delta) / m
        db[-1] = np.sum(delta, axis=0, keepdims=True) / m
        
        for i in range(len(self.weights) - 2, -1, -1):
            delta = np.dot(delta, self.weights[i+1].T) * self.activation.derivative(zs[i])
            dW[i] = np.dot(activations[i].T, delta) / m
            db[i] = np.sum(delta, axis=0, keepdims=True) / m
        
        return dW, db

    def update_params(self, dW, db):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * dW[i]
            self.biases[i] -= self.learning_rate * db[i]

    def fit(self, X, y, X_val, y_val,output_size):

        if self.model == 'classifier':
            y = np.eye(output_size)[y]
            y_val = np.eye(output_size)[y_val]
        
        elif self.model == 'multi-label':
            y_onehot=[]
            for l in y:
                zeros=[0]*output_size
                for i in l:
                    zeros[i]=1
                y_onehot.append(zeros)
            y=np.array(y_onehot)
            y_onehot=[]
            for l in y_val:
                zeros=[0]*output_size
                for i in l:
                    zeros[i]=1
                y_onehot.append(zeros)

            y_val=np.array(y_onehot)

        input_size = X.shape[1]
        output_size = y.shape[1] if len(y.shape) > 1 else 1
        self.init_params(input_size, output_size)

        best_loss = float('inf')
        patience_counter = 0

        probs, _ = self.forward_prop(X)
        y_pred=probs[-1]
        if self.loss_func == 'cce':
            loss=self.categorical_cross_entropy(y,y_pred)
        elif self.loss_func == 'mse':
            loss=self.mse_loss(y,y_pred)
        else:
            loss=self.binary_cross_entropy(y,y_pred)
        self.loss_history.append(loss)
        
        for epoch in tqdm(range(self.epochs), desc="Training", unit="epoch"):
            if self.optimizer == 'sgd':
                for i in range(X.shape[0]):
                    dW, db = self.back_prop(X[i:i+1], y[i:i+1])
                    self.update_params(dW, db)
            elif self.optimizer == 'batch':
                dW, db = self.back_prop(X, y)
                self.update_params(dW, db)
            else:  # mini-batch
                for i in range(0, X.shape[0], self.batch_size):
                    batch_X = X[i:i+self.batch_size]
                    batch_y = y[i:i+self.batch_size]
                    dW, db = self.back_prop(batch_X, batch_y)
                    self.update_params(dW, db)
            
            # Calculate loss
            probs, _ = self.forward_prop(X)
            y_pred=probs[-1]

            val_probs, _= self.forward_prop(X_val)
            y_pred_val = val_probs[-1]

            if self.loss_func == 'cce':
                loss_val=self.categorical_cross_entropy(y_val, y_pred_val)
                loss=self.categorical_cross_entropy(y,y_pred)
            elif self.loss_func == 'mse':
                loss_val=self.mse_loss(y_val, y_pred_val)
                loss=self.mse_loss(y,y_pred)
            else:
                loss_val=self.binary_cross_entropy(y_val, y_pred_val)
                loss=self.binary_cross_entropy(y,y_pred)

            self.loss_history.append(loss)

            if self.early_stopping:
                if loss_val < best_loss:
                    best_loss = loss_val
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        print(f"Early stopping at epoch {epoch}")
                        break

    def probabilities(self, X):
        if self.model == 'regressor':
            return self.predict(X)
        probs, _ =self.forward_prop(X)
        probs=probs[-1]
        return probs
    
    def predict(self, X, threshold = 0.5):
        probs, _ =self.forward_prop(X)
        y_pred=probs[-1]
        if self.logit:
            y_pred=self.sig.func(x=y_pred)
            return np.where(y_pred>threshold,1,0)
        if self.model == 'classifier':
            return np.argmax(y_pred,axis=1)
        elif self.model == 'multi-label':
            return (y_pred >= threshold).astype(int)
        else:
            return y_pred
    
    def gradient_check(self, X, y, epsilon=1e-7):
        X = np.array(X)

        if self.model == 'classifier':
            y = np.array(y)
            y = np.eye(self.layers[-1])[y]
        elif self.model == 'multi-label':
            y_onehot=[]
            for l in y:
                zeros=[0]*8
                for i in l:
                    zeros[i]=1
                y_onehot.append(zeros)

            y=np.array(y_onehot)

        y = np.array(y)

        # y_pred, _ = self.forward_prop(X)
        dW, db = self.back_prop(X, y)

        if self.loss_func == 'mse':
            lfunc=self.mse_loss
        elif self.loss_func == 'cce':
            lfunc=self.categorical_cross_entropy
        else:
            lfunc=self.binary_cross_entropy
        
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            dW_numerical = np.zeros_like(w)
            db_numerical = np.zeros_like(b)
            
            it = np.nditer(w, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                index = it.multi_index
                old_value = w[index]
                w[index] = old_value + epsilon
                y_pred_plus, _ = self.forward_prop(X)
                y_pred_plus = y_pred_plus[-1]
                w[index] = old_value - epsilon
                y_pred_minus, _ = self.forward_prop(X)
                y_pred_minus = y_pred_minus[-1]
                w[index] = old_value
                dW_numerical[index] = (lfunc(y, y_pred_plus) - lfunc(y, y_pred_minus)) / (2 * epsilon)
                it.iternext()
            
            it = np.nditer(b, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                index = it.multi_index
                old_value = b[index]
                b[index] = old_value + epsilon
                y_pred_plus, _ = self.forward_prop(X)
                y_pred_plus = y_pred_plus[-1]
                b[index] = old_value - epsilon
                y_pred_minus, _ = self.forward_prop(X)
                y_pred_minus = y_pred_minus[-1]
                b[index] = old_value
                db_numerical[index] = (lfunc(y, y_pred_plus) - lfunc(y, y_pred_minus)) / (2 * epsilon)
                it.iternext()
            
            eps=1e-9
            print(f"Layer {i+1}:")
            print(f"  Weights - Max difference: {np.max(np.abs(dW[i] - dW_numerical))}")
            print(f"  Biases - Max difference: {np.max(np.abs(db[i] - db_numerical))}")
            print(f"  Relative difference (weights): {np.linalg.norm(dW[i] - dW_numerical) / (np.linalg.norm(dW[i]) + np.linalg.norm(dW_numerical)+eps)}")
            print(f"  Relative difference (biases): {np.linalg.norm(db[i] - db_numerical) / (np.linalg.norm(db[i]) + np.linalg.norm(db_numerical)+eps)}")