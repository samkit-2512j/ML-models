import numpy as np
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from models.MLP.mlp import MLP
from models.MLP.activations import *

class Autoencoder:
    def __init__(self, input_dim, encoding_dim, hidden_layers=None, 
                 learning_rate=0.01, activation=sigmoid, optimizer='mini-batch', 
                 batch_size=32, epochs=100, patience=10, early_stopping=False):
        
        if hidden_layers is None:
            hidden_layers = []
        
        encoder_layers = hidden_layers + [encoding_dim]
        self.encoder_index = len(hidden_layers) + 1
        decoder_layers = hidden_layers[::-1]
        self.layers = encoder_layers + decoder_layers
        self.autoencoder = MLP(hidden_layers=self.layers,
                           learning_rate=learning_rate,
                           activation=activation,
                           optimizer=optimizer,
                           batch_size=batch_size,
                           epochs=epochs,
                           patience=patience,
                           early_stopping=early_stopping,
                           model='regressor',
                           loss_func='mse')

    def fit(self, X, X_val=None, verbose=True):
        self.autoencoder.fit(X, X, X_val, X_val, output_size=X.shape[1])
        
        encoded, _ = self.autoencoder.forward_prop(X)
        encoded_X = encoded[self.encoder_index]
        if X_val is not None:
            encoded, _ = self.autoencoder.forward_prop(X_val)
            encoded_X_val = encoded[self.encoder_index]

        else:
            encoded_X_val = None

        if verbose:
            print("Autoencoder training completed.")

    def encode(self, X):
        return self.encoder.predict(X)

    def decode(self, X):
        return self.decoder.predict(X)

    def reconstruct(self, X):
        encoded, _ = self.autoencoder.forward_prop(X)
        return encoded[-1]
    
    def get_latent(self, X):
        encoded, _ = self.autoencoder.forward_prop(X)
        encoded_X = encoded[self.encoder_index]
        return encoded_X
    
    def reconstruction_error(self, X, metric='mse'):
        reconstructed_X = self.reconstruct(X)
        
        error = np.mean((X - reconstructed_X) ** 2)
        return error
        