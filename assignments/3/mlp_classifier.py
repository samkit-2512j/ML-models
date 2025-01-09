import numpy as np
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from models.MLP.mlp_mc import MLPClassifier_MC
from models.MLP.mlp import MLP
from models.MLP.mlp_wandb import MLPW
from models.MLP.activations import *
from performance_measures.performance_measures import PerfMeeasures

def mlp_classifier():

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

    # mlp=MLPClassifier_MC([32,16],0.01,relu,'mini_batch',epochs=1000,early_stopping=True)
    # mlp=MLPW([32,16],0.01,relu,'mini_batch',epochs=1000,early_stopping=True,model='classifier',loss_func='cce')
    mlp=MLPW([64,32],0.052055,tanh,'mini_batch',epochs=1000,early_stopping=True,model='classifier',loss_func='cce')

    mlp.fit(x_train,y_train,x_val, y_val, 6, log_wandb=False)
    prob=mlp.probabilities(x_test)

    p= mlp.predict(x_test)
    print(p)
    print(y_test)

    pf=PerfMeeasures(y_test,p,6)
    print("Accuracy:",pf.accuracy())
    print("Precision:",pf.class_precision())
    print("Recall:",pf.class_recall())
    print("F1 score:",pf.f1_score_all())
    print("Final loss:",mlp.loss_history[-1])

    mlp.gradient_check(x_train,y_train)
