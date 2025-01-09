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
        "name": "regression Sweep",
        "metric": {"goal": "maximize", "name": "val_MSE_loss"},
        "parameters": {
            "hidden_layers": {"values": [[64, 32], [128, 64], [32,16]]},
            "batch_size": {"values": [32, 64, 128, 256]},
            "epochs": {"values": [100,500,1000]},
            "optimizer": {"values": ['sgd','mini-batch','batch']},
            "loss_func": {"values": ["mse"]},
            "activations": {"values": ['tanh', 'relu', 'sigmoid']},
            "model": {"values": ["regressor"]},
            "learning_rate": {"max": 0.1, "min": 0.0001},
            'early_stopping': {"values": [True]}
        }
    }      

    params = {
        'hidden_layers': [32,16],
        'learning_rate' : 0.01,
        'batch_size':64,
        'epochs': 100,
        'optimizer':"mini-batch",
        'loss_func':'mse',
        'activation':tanh,
        "model":"regressor",
        'early_stopping': True
    }
    
    sweep_id = wandb.sweep(sweep=sweep_config, project="Regressor_hyperparametrs")


    return sweep_id


def run():
    params = {
        'hidden_layers': [32,16],
        'learning_rate' : 0.01,
        'batch_size':64,
        'epochs': 100,
        'optimizer':"mini-batch",
        'loss_func':'mse',
        'activation':tanh,
        "model":"regressor",
        'early_stopping': True
    }


    with wandb.init(config=params):
        config = wandb.config
        
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

        df=pd.read_csv("../../data/external/HousingData.csv")

        df = df.fillna(df.mean())

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

        y=df['MEDV'].values
        df=df.drop(columns='MEDV')
        cols=df.select_dtypes(include=['int','float']).columns
        df[cols] = df[cols].apply(lambda x: (x - x.mean()) / x.std())
        X=df.values

        X_train,y_train,X_val,y_val,X_test,y_test = split_Dataset(X,y)

        model.fit(X_train,y_train,X_val,y_val,1, log_wandb=True)


def regression_hyperparameter():
    sweep_id = setup()
    wandb.agent(sweep_id=sweep_id,function=run)