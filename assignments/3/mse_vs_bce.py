import numpy as np
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from models.MLP.mlp import MLP
from models.MLP.mlp_wandb import MLPW
from models.MLP.activations import *
from performance_measures.performance_measures import PerfMeeasures
import matplotlib.pyplot as plt

def mse_vs_bce():

    df=pd.read_csv("../../data/external/diabetes.csv")
    df = df.sample(frac=1).reset_index(drop=True)

    y = df['Outcome'].values
    df=df.drop(columns='Outcome')
    cols=df.columns
    df[cols] = df[cols].apply(lambda x: (x - x.mean()) / x.std())
    X=df.values

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

    X_train,y_train,X_val,y_val,X_test,y_test = split_Dataset(X,y)


    mlp_mse = MLP(hidden_layers = [16,32,8,4],
                activation = tanh,
                optimizer='mini-batch',
                batch_size = 64,
                early_stopping = True,
                patience = 50,
                model = 'regressor', loss_func = 'mse', logit = True)

    mlp_bce = MLP(hidden_layers = [16,32,8,4],
                activation = tanh,
                optimizer='mini-batch',
                batch_size = 64,
                early_stopping = True,
                patience = 50,
                model = 'regressor', loss_func = 'bce', logit = True)

    mlp_mse.fit(X_train,y_train.reshape(-1,1),X_val,y_val, 1)

    y_pred_mse = mlp_mse.predict(X_train)

    pf_mse=PerfMeeasures(y_train,y_pred_mse.reshape(1,-1),2)

    print("Acc for MSE:",pf_mse.accuracy())

    mlp_bce.fit(X_train,y_train,X_val,y_val, 1)

    y_pred_bce = mlp_bce.predict(X_train)

    pf=PerfMeeasures(y_train,y_pred_bce.reshape(1,-1),2)

    print("Acc for BCE:",pf.accuracy())

    loss_bce = np.array(mlp_bce.loss_history)
    loss_mse = np.array(mlp_mse.loss_history)


    plt.plot(loss_mse)
    plt.xlabel('Epochs')
    plt.ylabel('BCE Loss')
    plt.title('BCE Loss vs Epochs')
    plt.grid()
    plt.savefig('./figures/diabetes_bce.jpg')
    plt.clf()

    plt.plot(loss_bce)
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.title('MSE Loss vs Epochs')
    plt.grid()
    plt.savefig('./figures/diabetes_mse.jpg')

    loss_mse = mlp_mse.mse_loss(y_test, mlp_mse.predict(X_test).reshape(1,-1))
    loss_bce = mlp_bce.binary_cross_entropy(y_test, mlp_bce.predict(X_test).reshape(1,-1))

    # plt.plot(loss_mse, label='mse', color='b')
    # plt.plot(loss_bce, label='bce', color='r')
    # plt.xlabel('data points')
    # plt.ylabel('loss')
    # plt.title('MSE vs BCE loss')
    # plt.legend()
    # plt.show()