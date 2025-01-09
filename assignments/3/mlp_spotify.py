import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from models.PCA.pca import Pca
from models.autoencoder.autoencoder import Autoencoder
from models.MLP.activations import *
from performance_measures.performance_measures import PerfMeeasures
from models.MLP.mlp import MLP

def label_encode_col(df,col):
    encoded_labels, _ = pd.factorize(df[col])
    df[col] = encoded_labels    
    return df

def normalise(X_train,X_test,X_val):
    m=X_train.mean()
    sd=X_train.std()

    X_train_norm=(X_train-m)/sd
    X_val_norm=(X_val-m)/sd
    X_test_norm=(X_test-m)/sd

    return X_train_norm,X_val_norm,X_test_norm

def drop_unimportant_cols(df,col_list):
        for col in col_list:
            if(col in df.columns):
                df.drop(columns=col,inplace=True)

def drop_duplicate_genres(df):
    df.drop_duplicates(subset= ['track_id'],keep='first',inplace = True)
    return df

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


def normalize_dataframe(df,exclude_columns):
    columns_to_normalize = df.columns.difference(exclude_columns)
    
    normalized_df = df.copy()
    
    mean = df[columns_to_normalize].mean()
    std = df[columns_to_normalize].std()
    
    std = std.replace(0, 1)
    
    normalized_df[columns_to_normalize] = (df[columns_to_normalize] - mean) / std
    
    return normalized_df

def reduce_dimensions(X,k):
    pc=Pca(k)
    pc.fit(X)
    x_red=pc.transform(X)
    print(pc.checkPCA(X))
    return x_red



def spotify_1_mlp():
    file_path='../../data/external/spotify.csv'
    df=pd.read_csv(file_path,index_col=0)

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    df=drop_duplicate_genres(df)
    drop_unimportant_cols(df,['track_id','album_name','track_name','duration_ms','explicit','artists'])

    for i in ['track_genre']:
        df=label_encode_col(df, i)

    track_genres=df['track_genre']
    drop_unimportant_cols(df,['track_genre'])


    df=normalize_dataframe(df,[])
    print(df.head())

    y=track_genres.values
    X=df.select_dtypes(include=['int','float']).values
    
    X_train,y_train,X_val,y_val,X_test,y_test=split(X,y)

    mlp = MLP(hidden_layers=[32,64,256,128], activation=relu, 
              optimizer='mini-batch',batch_size=256, patience = 100, early_stopping=True,
              model='classifier', loss_func='cce')
    
    mlp.fit(X_train,y_train,X_val,y_val,len(np.unique(y)))

    y_pred_orig = mlp.predict(X_test)
    pf_orig=PerfMeeasures(y_test,y_pred_orig,total_features=len(np.unique(y)))
    accuracy=pf_orig.accuracy()
    recall=pf_orig.class_recall()
    precision=pf_orig.class_precision()
    f1score=pf_orig.f1_score_all(mode='macro')
    print(f"Original accuracy is: {accuracy}")
    print(f"Original recall is: {np.mean(recall)}")
    print(f"Original precision is: {np.mean(precision)}")
    print(f"Original F1-score is: {f1score}")

if __name__ == '__main__':
    spotify_1_mlp()