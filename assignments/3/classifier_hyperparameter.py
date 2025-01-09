import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import wandb
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from models.MLP.activations import *
from models.MLP.mlp_wandb import MLPW

def setup():
    sweep_config = {
        "method": "bayes",
        "name": "classification Sweep",
        "metric": {"goal": "maximize", "name": "val_acc"},
        "parameters": {
            "hidden_layers": {"values": [[64, 32], [128, 64], [32,16]]},
            "batch_size": {"values": [32, 64, 128, 256]},
            "epochs": {"values": [100,500,1000]},
            "optimizer": {"values": ['sgd','mini-batch','batch']},
            "loss_func": {"values": ["cce"]},
            "activations": {"values": ['tanh', 'relu', 'sigmoid']},
            "model": {"values": ["classifier"]},
            "learning_rate": {"max": 0.1, "min": 0.0001},
            'early_stopping': {"values": [True]}
        }
    }      

    params = {
        'hidden_layers': [32,16],
        'learning_rate' : 1e-3,
        'batch_size':64,
        'epochs': 100,
        'optimizer':"mini-batch",
        'loss_func':'cce',
        'activation':tanh,
        "model":"classifier",
        'early_stopping': True
    }
    
    sweep_id = wandb.sweep(sweep=sweep_config, project="Classifier_hyperparametrs")


    return sweep_id


def run():
    params = {
        'hidden_layers': [32,16],
        'learning_rate' : 1e-3,
        'batch_size':64,
        'epochs': 100,
        'optimizer':"mini-batch",
        'loss_func':'cce',
        'activation':tanh,
        "model":"classifier",
        'early_stopping': True
    }


    with wandb.init(config=params):
        config = wandb.config
        path = "../../data/external/WineQT.csv"
        
        activation_fn_mapping = {
            "linear": linear,
            "relu": relu,
            "sigmoid": sigmoid,
            "softmax": softmax,
            "tanh": tanh
        }

        model = MLPW(
            hidden_layers=config.hidden_layers,
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            epochs=config.epochs,
            optimizer=config.optimizer,
            loss_func=config.loss_func,
            activation=activation_fn_mapping[config.activations],
            model=config.model,
            early_stopping=config.early_stopping
        )

        print(model)

        df=pd.read_csv("../../data/external/WineQT.csv")
        df = df.sample(frac=1).reset_index(drop=True)

        def split_Dataset(X,y):
            train_ratio = 0.8
            val_ratio = 0.1
            test_ratio = 0.1

            indices = np.arange(X.shape[0])

            train_size = int(train_ratio * X.shape[0])
            val_size = int(val_ratio * X.shape[0])
            test_size = X.shape[0] - train_size - val_size
            train_indices = indices[:train_size]
            val_indices = indices[train_size:train_size + val_size]
            test_indices = indices[train_size + val_size:]

            X_train, y_train = X[train_indices], y[train_indices]
            X_val, y_val = X[val_indices], y[val_indices]
            X_test, y_test = X[test_indices], y[test_indices]

            return X_train,y_train,X_val,y_val,X_test,y_test


        features=df.drop(columns='quality')
        features=features.drop(columns=['Id'])
        cols=features.columns
        features[cols] = features[cols].apply(lambda x: (x - x.mean()) / x.std())
        classes=df['quality']
        y=classes.values
        y=np.array(y)
        y_vals=y
        y=y-3

        x=features.values
        x=np.array(x)

        x_train,y_train,x_val,y_val,x_test,y_test=split_Dataset(x,y)

        model.fit(x_train,y_train,x_val,y_val,6, log_wandb=True)


def classifier_hyperparameter():
    sweep_id = setup()
    wandb.agent(sweep_id=sweep_id,function=run)