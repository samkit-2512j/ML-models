import numpy as np
import pandas as pd
from autoencoder_knn import *
from mse_vs_bce import *
from classifier_analysis import *
from mlp_classifier import *
from classifier_hyperparameter import *
from mlp_regression import *
from mlp_spotify import *
from multi_label_hyperparameter import *
from multi_label_mlp import *
from regressor_hyperparameter import *

'''MLP and MLPW are the same class except that MLPW has wandb integrated with it'''

# Run single label classifier
mlp_classifier()

# Classifier hyperparameter tuning
classifier_hyperparameter()

# Effect of parameter changes
classifier_analysis()

#Run multilabel classifier
multi_label_mlp()

# Multi label hyperparameter tuning
multi_label_hyperparameter()

#Run MLP Regression
mlp_regregression()

# Regressor hyperparameter tuning
regression_hyperparameter()

#MSE vs BCE
mse_vs_bce()

# Autoencoders + knn
spotify_1()

#MLP on spotify1
spotify_1_mlp()



