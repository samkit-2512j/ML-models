import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from models.MLP.mlp_wandb import MLPW
from models.MLP.activations import *
from performance_measures.performance_measures import PerfMeeasures

def classifier_analysis():
    # Load and preprocess dataset
    df = pd.read_csv("../../data/external/WineQT.csv")
    df = df.sample(frac=1).reset_index(drop=True)

    def split_Dataset(X, y):
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

        return X_train, y_train, X_val, y_val, X_test, y_test

    # Dataset preparation
    features = df.drop(columns='quality')
    features = features.drop(columns=['Id'])
    cols = features.columns
    features[cols] = features[cols].apply(lambda x: (x - x.mean()) / x.std())
    classes = df['quality']
    y = np.array(classes.values)
    y = y - 3  # Normalize labels

    x = features.values
    x_train, y_train, x_val, y_val, x_test, y_test = split_Dataset(x, y)

    # Lists of hyperparameters
    activations = [tanh, relu, sigmoid, linear]
    learning_rates = [0.01, 0.052055, 0.001, 0.1]
    batch_sizes = [32, 64, 128, 256]

    # For tracking losses during training
    losses_activation = {}
    losses_lr = {}
    losses_batch_size = {}

    # 1. Effect of Non-linearity: Vary activation functions
    plt.figure(figsize=(10, 6))
    for activation in activations:
        mlp = MLPW([64, 32], 0.01, activation, 'mini_batch', epochs=500, early_stopping=True, model='classifier', loss_func='cce')
        mlp.fit(x_train, y_train, x_val, y_val, 6, log_wandb=False)
        losses_activation[activation.__name__] = mlp.loss_history
        plt.plot(mlp.loss_history, label=activation.__name__)

    plt.title("Effect of Activation Functions on Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('./figures/classifier_act_analysis.jpg')
    plt.show()

    # 2. Effect of Learning Rate: Vary learning rates
    plt.figure(figsize=(10, 6))
    for lr in learning_rates:
        mlp = MLPW([64, 32], lr, tanh, 'mini_batch', epochs=500, early_stopping=True, model='classifier', loss_func='cce')
        mlp.fit(x_train, y_train, x_val, y_val, 6, log_wandb=False)
        losses_lr[lr] = mlp.loss_history
        plt.plot(mlp.loss_history, label=f'LR: {lr}')

    plt.title("Effect of Learning Rate on Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('./figures/classifier_lr_analysis.jpg')
    plt.show()

    # 3. Effect of Batch Size: Vary batch sizes
    plt.figure(figsize=(10, 6))
    for batch_size in batch_sizes:
        mlp = MLPW([64, 32], 0.01, tanh, 'mini_batch', epochs=500, batch_size=batch_size, early_stopping=True, model='classifier', loss_func='cce')
        mlp.fit(x_train, y_train, x_val, y_val, 6, log_wandb=False)
        losses_batch_size[batch_size] = mlp.loss_history
        plt.plot(mlp.loss_history, label=f'Batch Size: {batch_size}')

    plt.title("Effect of Batch Size on Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('./figures/classifier_bs_analysis.jpg')
    plt.show()

    # Performance evaluation
    p = mlp.predict(x_test)
    pf = PerfMeeasures(y_test, p, 6)
    print("Accuracy:", pf.accuracy())
    print("Precision:", pf.class_precision())
    print("Recall:", pf.class_recall())
    print("F1 score:", pf.f1_score_all())
    print("Final loss:", mlp.loss_history[-1])
