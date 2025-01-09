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
from sklearn.preprocessing import LabelEncoder

def setup():
    sweep_config = {
        "method": "bayes",
        "name": "multi-label Sweep",
        "metric": {"goal": "maximize", "name": "val_acc"},
        "parameters": {
            "hidden_layers": {"values": [[64, 32], [128, 64], [32,16]]},
            "batch_size": {"values": [32, 64, 128, 256]},
            "epochs": {"values": [100,500,1000]},
            "optimizer": {"values": ['sgd','mini-batch','batch']},
            "loss_func": {"values": ["bce"]},
            "activations": {"values": ['tanh', 'relu', 'sigmoid']},
            "model": {"values": ["multi-label"]},
            "learning_rate": {"max": 0.1, "min": 0.001},
            'early_stopping': {"values": [True]}
        }
    }      

    params = {
        'hidden_layers': [32,16],
        'learning_rate' : 0.01,
        'batch_size':64,
        'epochs': 100,
        'optimizer':"mini-batch",
        'loss_func':'cce',
        'activation':tanh,
        "model":"classifier",
        'early_stopping': True
    }
    
    sweep_id = wandb.sweep(sweep=sweep_config, project="Multi-Label_hyperparametrs")


    return sweep_id


def run():
    params = {
        'hidden_layers': [32,16],
        'learning_rate' : 0.01,
        'batch_size':64,
        'epochs': 100,
        'optimizer':"mini-batch",
        'loss_func':'bce',
        'activation':tanh,
        "model":"multi-label",
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

        df=pd.read_csv("../../data/external/advertisement.csv")

        def split_Dataset(X,y):
            train_ratio = 0.8
            val_ratio = 0.1
            test_ratio = 0.1

            indices = np.arange(X.shape[0])

            train_size = int(train_ratio * X.shape[0])
            val_size = int(val_ratio * X.shape[0])
            test_size = X.shape[0] - train_size - val_size
            X_train, y_train = X[:train_size], y[:train_size]
            X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
            X_test, y_test = X[train_size+val_size:], y[train_size+val_size :]

            return X_train,y_train,X_val,y_val,X_test,y_test

        labels_raw=np.array(df['labels'])
        labels = []
        for i in range(len(labels_raw)):
            words=labels_raw[i].split()
            labels.append(words)
        flattened_list = [item for sublist in labels for item in sublist]
        unique_list = list(set(flattened_list))

        print(unique_list)

        df = df.sample(frac=1).reset_index(drop=True)
   

        labels_raw=np.array(df['labels'])
        labels = []
        for i in range(len(labels_raw)):
            words=labels_raw[i].split()
            labels.append(words)

        features=df.drop(columns='labels')
       
        cols=features.select_dtypes(include=['int','float']).columns
        df[cols] = features[cols].apply(lambda x: (x - x.mean()) / x.std())
        df=df.drop(columns=['labels','city'])

     

        columns_to_encode = ['gender', 'education', 'married', 'occupation', 'most bought item']
        le = LabelEncoder()
        for col in columns_to_encode:
            df[col] = le.fit_transform(df[col])

        label_dict = {word: idx for idx, word in enumerate(unique_list)}
        y_encoded=[]
        for l in labels:
            y_encoded.append([label_dict[word] for word in l])
        num_labels=len(unique_list)

        X=df.values

        X_train,y_train,X_val,y_val,X_test,y_test = split_Dataset(X,y_encoded)
        model.fit(X_train,y_train,X_val,y_val,len(unique_list), log_wandb=True)


def multi_label_hyperparameter():
    sweep_id = setup()
    wandb.agent(sweep_id=sweep_id,function=run)