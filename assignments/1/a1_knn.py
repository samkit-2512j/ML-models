import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import seaborn as sns
import heapq
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from models.knn.knn import KNN
import math

from performance_measures.performance_measures import PerfMeeasures

def Boxplot(df):
        for column in df.columns:
            df.boxplot(column=column)
            plt.title("boxplot for label :: {}".format(column))
            plt.ylabel("values")
            plt.xlabel("Labels")
            plt.savefig(f"./figures/knn/{column}_box.jpg")
            plt.clf()


# def Boxplot(df):
#     num_columns = len(df.columns)
    
#     # Calculate number of rows and columns based on the number of features
#     num_rows = math.ceil(math.sqrt(num_columns))
#     num_cols = math.ceil(num_columns / num_rows)
    
#     fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(5 * num_cols, 5 * num_rows))
#     axes = axes.flatten()  # Flatten the array of axes for easy indexing

#     for i, column in enumerate(df.columns):
#         df.boxplot(column=column, ax=axes[i])
#         axes[i].set_title(f"Boxplot for label: {column}")
#         axes[i].set_ylabel("Values")
#         axes[i].set_xlabel("Labels")

#     # Hide any unused subplots
#     for j in range(i + 1, len(axes)):
#         axes[j].axis('off')

#     plt.tight_layout()
#     plt.savefig(f"./figures/knn/combined_boxplots.jpg")
#     plt.show()


def correlation(df,name):
    corr_mat = df.select_dtypes(include=["int", "float"]).corr()
    plt.figure(figsize=(20, 20), facecolor='#F2EAC5', edgecolor='black')
    ax = plt.axes()
    ax.set_facecolor('#F2EAC5')
    sns.heatmap(corr_mat, annot=True, cmap='coolwarm', linewidths=0.5, annot_kws={"size": 10})
    plt.title('Correlation Analysis')
    # plt.show()
    plt.savefig(f'./figures/knn/corr_{name}.jpg')
    plt.clf()

def hist_chart(table,name):
    num_cols = table[table.columns[(table.dtypes == 'float64') | (table.dtypes == 'int64')]]
    sns.set_style('darkgrid')
    sns.set_theme(rc={"axes.facecolor":"#F2EAC5","figure.facecolor":"#F2EAC5"})
    num_cols.hist(figsize=(20,15), bins=30, xlabelsize=8, ylabelsize=8)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'./figures/knn/hist_{name}.jpg')
    plt.clf()

def find_best_combination(X_train,X_val,y_train,y_val):
    max_acc=0
    best_k=0
    best_metric=''
    for k in range(1,10):
        print(k)
        for i in ['euclidean','manhattan','cosine_similarity','minkowski']:
            print(i)
            model=KNN(k,i)
            model.fit(X_train,y_train)
            y_pred=model.predict(X_val)
            pf=PerfMeeasures(y_val,y_pred)
            acc=pf.accuracy()
            if acc>max_acc:
                best_k=k
                best_metric= i
                max_acc=acc

    return best_k,best_metric

def find_best_combinations(X_train, X_val, y_train, y_val, top_n=10):
    # Initialize a list to store the combinations and their accuracies
    top_combinations = []
    
    # Iterate over possible values of k
    for k in range(1, 10):
        print(k)
        
        # Iterate over different distance metrics
        for i in ['euclidean', 'manhattan', 'cosine_similarity', 'minkowski']:
            print(i)
            
            # Initialize and train the KNN model
            model = KNN(k, i)
            model.fit(X_train, y_train)
            
            # Predict on the validation set
            y_pred = model.predict(X_val)
            
            # Calculate accuracy
            pf = PerfMeeasures(y_val, y_pred)
            acc = pf.accuracy()
            
            # Store the combination and accuracy in the list
            top_combinations.append((k, i, acc))
            
            # Sort the list by accuracy and keep only the top N combinations
            top_combinations = sorted(top_combinations, key=lambda x: x[2], reverse=True)[:top_n]
    
    # Return the top N combinations
    return top_combinations



def normalise(X_train,X_test,X_val):
    m=X_train.mean()
    sd=X_train.std()

    X_train_norm=(X_train-m)/sd
    X_val_norm=(X_val-m)/sd
    X_test_norm=(X_test-m)/sd

    return X_train_norm,X_val_norm,X_test_norm

def split(x,y,artist=False,art=None):
    # Calculate the indices for the split
    train_size = int(0.8 * len(x))
    val_size = int(0.1 * len(x))
    test_size = len(x) - train_size - val_size

    if artist and art is not None:
        artist_train=art[:train_size]
        artist_val=art[train_size:train_size+val_size]
        artist_test=art[train_size+val_size:]
        return artist_train,artist_val,artist_test

    # Split the data
    train_x = x[:train_size]
    val_x = x[train_size:train_size + val_size]
    test_x = x[train_size + val_size:]
    y_train=y[:train_size]
    y_val=y[train_size:train_size+val_size]
    y_test=y[train_size+val_size:]

    return train_x,y_train,val_x,y_val,test_x,y_test

def label_encode_col(df,col):
    # encoded,_=pd.factorize(df[col])
    # df[col]=encoded
    df[col],_=pd.factorize(df[col])


def data_preprocess_sp_1(df):
    hist_chart(df,"before")
    Boxplot(df.select_dtypes(include=['int','float']))
    df=df.dropna(axis=0)
    df['explicit']=df['explicit'].astype(int)

    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    final_df=df_shuffled.drop_duplicates(subset=['track_id'],keep='first')

    print(final_df.info())

    final_df.drop(columns=['track_id','album_name','track_name'],inplace=True)

    for i in ['artists','track_genre']:
        label_encode_col(final_df,i)

    final_df=final_df.select_dtypes(include=['int','float'])
    correlation(final_df,"after")
    # print(final_df.head())
    y = final_df['track_genre'].values
    X = final_df.drop(columns=['artists']).values
    art=final_df['artists'].values

    X_train,y_train,X_val,y_val,X_test,y_test=split(X,y)
    art_train,art_val,art_test=split(X,y,artist=True,art=art)

    X_train,X_val,X_test=normalise(X_train,X_val,X_test)

    # print(X_test.shape , art_test.shape)
    X_train=np.concatenate((X_train,art_train.reshape(-1,1)),axis=1)
    X_val=np.concatenate((X_val,art_val.reshape(-1,1)),axis=1)
    X_test=np.concatenate((X_test,art_test.reshape(-1,1)),axis=1)

    print(X_train.shape,X_test.shape,X_val.shape)

    return X_train,X_val,X_test,y_train,y_val,y_test

def data_preprocess_sp_2(df_train,df_test,df_val):

    print(df_train['artists'].shape , df_train.shape)

    df_train=df_train.dropna(axis=0)
    df_test=df_test.dropna(axis=0)
    df_val=df_val.dropna(axis=0)

    df_train['explicit']=df_train['explicit'].astype(int)
    df_test['explicit']=df_test['explicit'].astype(int)
    df_val['explicit']=df_val['explicit'].astype(int)


    df_shuffled_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)
    df_shuffled_test = df_test.sample(frac=1, random_state=42).reset_index(drop=True)
    df_shuffled_val = df_val.sample(frac=1, random_state=42).reset_index(drop=True)

    final_df_train=df_shuffled_train.drop_duplicates(subset=['track_id'],keep='first')
    final_df_test=df_shuffled_test.drop_duplicates(subset=['track_id'],keep='first')
    final_df_val=df_shuffled_val.drop_duplicates(subset=['track_id'],keep='first')

    final_df_train.drop(columns=['track_id','album_name','track_name'],inplace=True)
    final_df_test.drop(columns=['track_id','album_name','track_name'],inplace=True)
    final_df_val.drop(columns=['track_id','album_name','track_name'],inplace=True)

    final_df_train=final_df_train.select_dtypes(include=['int','float'])
    final_df_test=final_df_test.select_dtypes(include=['int','float'])
    final_df_val=final_df_val.select_dtypes(include=['int','float'])

    # print(final_df.head())
    X_train=final_df_train.drop(columns=['artists']).values
    y_train=final_df_train['track_genre'].values

    X_test=final_df_test.drop(columns=['artists']).values
    y_test=final_df_test['track_genre'].values
    X_val=final_df_val.drop(columns=['artists']).values
    y_val=final_df_val['track_genre'].values


    art_train=final_df_train['artists'].values
    art_test=final_df_test['artists'].values
    art_val=final_df_val['artists'].values

    print(art_train.shape,art_test.shape,art_val.shape)

    X_train,X_val,X_test=normalise(X_train,X_test,X_val)
    print(X_train.shape,X_test.shape,X_val.shape)

    print("Hello ",X_test.shape , art_test.shape)

    X_train=np.concatenate((X_train,art_train.reshape(-1,1)),axis=1)
    X_val=np.concatenate((X_val,art_val.reshape(-1,1)),axis=1)
    X_test=np.concatenate((X_test,art_test.reshape(-1,1)),axis=1)

    return X_train,X_val,X_test,y_train,y_val,y_test


def spotify_1():
    file_path='../../data/external/spotify.csv'
    df=pd.read_csv(file_path,index_col=0)
    X_train,X_val,X_test,y_train,y_val,y_test=data_preprocess_sp_1(df)
    top10=find_best_combination(X_train,X_val,y_train,y_val)
    n=top10[0][0]
    best_metric=top10[0][1]
    print(top10)
    print(f"Best k={n} and best metric={best_metric}")
    
    knn=KNN(k=n,dist_metric=best_metric)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)
    pf=PerfMeeasures(y_test,y_pred)
    accuracy=pf.accuracy()


    print(f"For neighbors: {n}, accuracy is: {accuracy:.4f}")
    return n,best_metric

def label_encode_combined(df_train,df_test,df_validate):
    df_combined = pd.concat([df_train, df_test, df_validate], axis=0, ignore_index=True)

    train_size = df_train.shape[0]
    test_size = df_test.shape[0]
    validate_size = df_validate.shape[0]

    for i in ['artists','track_genre']:
        label_encode_col(df_combined,i)

    df_train = df_combined.iloc[:train_size, :]
    df_test = df_combined.iloc[train_size:train_size + test_size, :]
    df_validate = df_combined.iloc[train_size + test_size:, :]

    return df_train,df_validate,df_test    

def spotify_2(n,best_metric):
    file_path1='../../data/external/spotify-2/test.csv'
    file_path2='../../data/external/spotify-2/train.csv'
    file_path3='../../data/external/spotify-2/validate.csv'

    df_train=pd.read_csv(file_path1,index_col=0)
    df_test=pd.read_csv(file_path2,index_col=0)
    df_validate=pd.read_csv(file_path3,index_col=0)

    df_train,df_validate,df_test=label_encode_combined(df_train,df_test,df_validate)

    print(df_train.shape, df_test.shape, df_validate.shape)

    X_train,X_val,X_test,y_train,y_val,y_test=data_preprocess_sp_2(df_train,df_test,df_validate)

    # knn = KNeighborsClassifier(n_neighbors=n)
    knn=KNN(k=n,dist_metric=best_metric)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)
    accuracy = np.mean(y_test == y_pred)
    # accuracies.append(accuracy)

    print(f"For neighbors: {n}, accuracy is: {accuracy:.4f}")


def knn_main():
    n,best_metric=spotify_1()
    spotify_2(n,best_metric)
