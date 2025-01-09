import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from models.PCA.pca import Pca
from models.k_means.kmeans import KMeans
from models.knn.knn import KNN
from performance_measures.performance_measures import PerfMeeasures

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



def spotify_1():
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

    Y=track_genres.values
    X=df.select_dtypes(include=['int','float']).values
    X_train,y_train,X_val,y_val,X_test,y_test=split(X,Y)

    n=5
    best_metric='minkowski'

    # print(X_train.shape)
    pc_train=Pca(7)
    pc_train.fit(X_train)
    pc_train.scree_plot(X_train,save=False,name='knn_scree_plot')
    x_train_red=reduce_dimensions(X_train,7)
    x_val_red=reduce_dimensions(X_val,7)
    x_test_red=reduce_dimensions(X_test,7)

    x_train_red,x_val_red,x_test_red=normalise(x_train_red,x_test_red,x_val_red)


    print(X_train.shape)
    knn=KNN(k=n,dist_metric=best_metric)
    knn.fit(X_train, y_train)

    y_pred_orig = knn.predict(X_test)
    pf_orig=PerfMeeasures(y_test,y_pred_orig,total_features=len(np.unique(y_test)))
    accuracy=pf_orig.accuracy()
    recall=pf_orig.class_recall()
    precision=pf_orig.class_precision()
    f1score=pf_orig.f1_score_all(mode='macro')
    time_taken_orig=pf_orig.measure_the_inference_time_for_plotting(knn,X_test)
    print(f"Original accuracy is: {accuracy:.4f}")
    print(f"Original recall is: {np.mean(recall)}")
    print(f"Original precision is: {np.mean(precision)}")
    print(f"Original F1-score is: {f1score:.4f}")

    
    print(x_train_red.shape)
    knn_pca=KNN(k=n,dist_metric=best_metric)
    knn_pca.fit(x_train_red, y_train)

    y_pred_pca = knn_pca.predict(x_test_red)
    pf_pca=PerfMeeasures(y_test,y_pred_pca)
    accuracy=pf_pca.accuracy()
    recall=pf_pca.class_recall()
    precision=pf_pca.class_precision()
    f1score=pf_pca.f1_score_all(mode='macro')
    time_taken_pca=pf_pca.measure_the_inference_time_for_plotting(knn_pca,x_test_red)
    print(f"PCA accuracy is: {accuracy:.4f}")
    print(f"PCA recall is: {np.mean(recall)}")
    print(f"PCA precision is: {np.mean(precision)}")
    print(f"PCA F1-score is: {f1score:.4f}")

    labels = ['Complete Dataset', 'Reduced Dimensional Dataset']
    times = [time_taken_orig, time_taken_pca]
    plt.bar(labels, times, color=['blue', 'green'])
    plt.ylabel('Inference Time')
    plt.title('Inference Time Plot')
    plt.savefig('./figures/inference_time.jpg')
    plt.show()