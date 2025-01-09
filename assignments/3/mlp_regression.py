import numpy as np
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from models.MLP.mlp_reg import MLPRegression
from models.MLP.mlp import MLP
from models.MLP.mlp_wandb import MLPW
from models.MLP.activations import *
from performance_measures.performance_measures import PerfMeeasures

def mlp_regregression():

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

    # mlp=MLPRegression([16,32,8,4],activation=relu,optimizer='mini-batch',epochs=1000,early_stopping=True,patience=75)
    # mlp=MLP([64,80],activation=relu, learning_rate=0.01, optimizer='mini-batch',epochs=1000,early_stopping=True,patience=75,model='regressor',loss_func='mse')
    # mlp=MLP([16,32,8,4],activation=relu,optimizer='mini-batch',epochs=1000,early_stopping=True,patience=75,model='regressor',loss_func='mse')
    # mlp=MLPW([16,32,8,4],activation=relu,optimizer='mini-batch',epochs=1000,early_stopping=False,patience=75,model='regressor',loss_func='mse')
    mlp=MLPW([32,16], learning_rate=0.066466, activation=relu,optimizer='mini-batch',batch_size=256,epochs=1000,early_stopping=False,patience=75,model='regressor',loss_func='mse')

    def r2_score(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return (ss_res / ss_tot)

    def mae(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    def rmse(y_true, y_pred):
        return np.sqrt(mlp.mse_loss(y_true, y_pred))

    mlp.fit(X_train,y_train,X_val,y_val,1)

    print("Initial loss:", mlp.loss_history[0])
    print("Final loss:", mlp.loss_history[-1])

    y_pred=mlp.predict(X_test)
    y_pred_val=mlp.predict(X_val)

    print("R2 score val: ",r2_score(y_val,y_pred))
    print("R2 score test: ",r2_score(y_test,y_pred))
    print("MAE on test: ",mae(y_test,y_pred))
    print("RMSE on test: ",rmse(y_test,y_pred))
    print("Final error on test: ", np.mean((y_test-y_pred)**2))

    print("Output of gradient check method:")
    mlp.gradient_check(X_train,y_train)


