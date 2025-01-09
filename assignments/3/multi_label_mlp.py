import numpy as np
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from models.MLP.mlp import MLP
from models.MLP.mlp_wandb import MLPW
from models.MLP.activations import *
from performance_measures.performance_measures import PerfMeeasures
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import multilabel_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def multi_label_mlp():

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

    # mlp=MLP(hidden_layers=[20,20],activation=relu,optimizer='mini_batch',epochs=1000,early_stopping=False, model='multi-label', loss_func='bce')
    mlp=MLPW(hidden_layers=[20,20],activation=relu,optimizer='mini_batch',epochs=1000,early_stopping=True, patience=75, model='multi-label', loss_func='bce')

    mlp.fit(X_train,y_train, X_val, y_val, len(unique_list))

    # mlp.fit(X_train,y_train, X_val, y_val, len(unique_list), log_wandb=True)
    probs=mlp.probabilities(X_test)
    y_pred=mlp.predict(X_test,threshold=0.3)

    print("Initial Loss: ", mlp.loss_history[0])
    print("Final loss:",mlp.loss_history[-1])

    y_onehot=[]
    for l in y_test:
        zeros=[0]*num_labels
        for i in l:
            zeros[i]=1
        y_onehot.append(zeros)

    y_test=np.array(y_onehot)

    def exact_match_ratio(y_true, y_pred):
        return np.mean(np.all(y_true == y_pred, axis=1))

    def accuracy(y_true, y_pred):
        intersection = np.logical_and(y_true, y_pred).sum(axis=1)
        union = np.logical_or(y_true, y_pred).sum(axis=1)
        iou = intersection / union

        return np.mean(iou)

    def hamming_loss(y_true, y_pred):
        return np.mean(y_true != y_pred)

    def precision_score(y_true, y_pred):
        true_positives = np.logical_and(y_true, y_pred).sum(axis=1)
        predicted_positives = y_pred.sum(axis=1)
        mask = predicted_positives != 0
        precision = np.zeros(y_true.shape[0])
        precision[mask] = true_positives[mask] / predicted_positives[mask]
        
        return np.mean(precision)

    def recall_score(y_true, y_pred):
        true_positives = np.logical_and(y_true, y_pred).sum(axis=1)
        actual_positives = y_true.sum(axis=1)
        mask = actual_positives != 0
        recall = np.zeros(y_true.shape[0])
        recall[mask] = true_positives[mask] / actual_positives[mask]
        
        return np.mean(recall)

    def f1_score(y_true, y_pred):
        true_positives = np.logical_and(y_true, y_pred).sum(axis=1)
        actual_positives = y_true.sum(axis=1)
        predicted_positives = y_pred.sum(axis=1)
        mask = (actual_positives + predicted_positives) != 0
        f1 = np.zeros(y_true.shape[0])
        f1[mask] = (2 * true_positives[mask]) / (actual_positives[mask] + predicted_positives[mask])
        
        return np.mean(f1)

    def calculate_metrics(y_true, y_pred):
        exact_mr = exact_match_ratio(y_true, y_pred)
        acc=accuracy(y_true,y_pred)
        hammingloss = hamming_loss(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        return exact_mr, acc,hammingloss, precision, recall, f1

    exact_mr, acc, hammingloss, precision, recall, f1 = calculate_metrics(y_test, y_pred)

    print(f"Exact Match Ratio: {exact_mr}")
    print("Accuracy: ",acc)
    print("Hamming Loss:",hammingloss)
    print("Total Precision:", precision)
    print("Total Recall:", recall)
    print(f"Total F1-score: {f1}")

    mlp.gradient_check(X_train,y_train)

    def plot_confusion_matrix(conf_matrix, labels):
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix for Multi-Label Classification')
        # plt.savefig('./figures/multi_label_confmatrix.jpg')
        plt.show()

    # Convert y_true and y_pred to binary format for each label
    y_true = np.array(y_test)
    y_pred = np.array(y_pred)

    # Generate confusion matrix for each label
    conf_matrix = multilabel_confusion_matrix(y_true, y_pred)

    # Summing up the confusion matrices across all labels to create an 8x8 matrix
    # This aggregates label-specific confusion matrices into one final matrix
    final_conf_matrix = np.zeros((num_labels, num_labels), dtype=int)

    for i in range(num_labels):
        true_positives = conf_matrix[i][1, 1]
        false_negatives = conf_matrix[i][1, 0]
        false_positives = conf_matrix[i][0, 1]
        true_negatives = conf_matrix[i][0, 0]
        
        final_conf_matrix[i][i] = true_positives
        for j in range(num_labels):
            if i != j:
                final_conf_matrix[i][j] += false_positives
                final_conf_matrix[j][i] += false_negatives

    # Plot the confusion matrix
    plot_confusion_matrix(final_conf_matrix, unique_list)



