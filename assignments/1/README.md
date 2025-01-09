# Assignment 1 Report - Samkit Jain

## K-Nearest Neighbours

### 2.2 Task 1
#### Correlation Matrix
![Correlation Matrix](./figures/knn/corr_after.jpg)
The above plot is a correlation matrix of the dataset after label encoding the categorical data.

Observations:
- There is a strong positive correlation between "loudness" and "energy".

- There is a strong negative correlation between "acousticness" and "energy".

- There is a strong negative correlation between "acousticness" and "loudness".

- There is a negative correlation between "instrumentalness" and "loudness".

- There is a weak positive correlation between "valence" (positiveness) and "danceability".

- There is a weak positive correlation between "speechiness" and "explicit".

#### Histogram Plot

Histogram Plot before data pre-processing:
![Histogram Plot](./figures/knn/hist_before.jpg)

Observations:
- The distributions in the columns danceability, tempo, and valence are almost normal.
- The loudness column has a skew to the left, with the majority of the tracks having noise levels between -15 and -5 dB.

- Songs with low values appear in the right-skewed distributions of the speechiness, acousticness, instrumentalness, and liveness columns.

- A large number of songs have a popularity score of 0, while the other songs are mostly within the normal range.

- The duration_ms column displays a distribution that is biased to the right; the longest song is around 5 million ms (83 minutes) long, while the majority of songs last less than 500,000 ms (8 minutes).

- Songs mostly with values between 0.4 and 0.9 make up the growing distribution of the energy column, which runs from 0 to 1.

- The values in the key column are evenly distributed and range from 0 to 11.

- Most songs have a "mode" value of 1.

- The most common "time_signature" value is 5.

- The values in columns like "danceability", "energy", "speechiness", etc. range from 0 to 1.

#### Boxplots

![acousticness](./figures/knn/acousticness_box.jpg)
![danceability](./figures/knn/danceability_box.jpg)
![duration](./figures/knn/duration_ms_box.jpg)
![energy](./figures/knn/energy_box.jpg)
![intrumentalness](./figures/knn/instrumentalness_box.jpg)
![key](./figures/knn/key_box.jpg)
![liveness](./figures/knn/liveness_box.jpg)
![loudness](./figures/knn/loudness_box.jpg)
![mode](./figures/knn/mode_box.jpg)
![popularity](./figures/knn/popularity_box.jpg)
![speechiness](./figures/knn/speechiness_box.jpg)
![tempo](./figures/knn/tempo_box.jpg)
![time_sig](./figures/knn/time_signature_box.jpg)
![valence](./figures/knn/valence_box.jpg)






### 2.3.1 Task 2

Model description:
```
class KNN:
    def __init__(self, k=5, dist_metric='euclidean',p=0.75):
        Initializes the class

    def fit(self, X, y):
        Fits the training data

    def predict(self, X):
        Predicts output data 

    def _predict(self, x):
        Private function that calculates the output for the data
    
    def euclidean_distance(self,x_train, x):
        Computes Euclidian Distance

    def manhattan_distance(self,x_train, x):
        Computes Manhattan Distance

    def cosine_similarity_distance(self,x_train, x):
        Computes Cosine Similarity Distance

    def minkowski_distance(self,x_train, x):
        Computes Minkowski Distance
```
The f1_score, accuracy precision, recall and other metrics have been implemented in the PerfMeasures class.

#### 2.4.1 Task 3

#### 2.5.1 Task 4
1. To vectorize the calculations of distaces I used libraries that made direct calculations from the training data set to each data point in X_val. I also used `np.argsort()` to make the searching for nearest k neighbours faster.
2. 
![](./figures/knn/time_70.jpg)
We observe that the optimised code runs faster than the non-optimised code for any given ratio of training and validation data.
This can also be observed in the below plot
3. 
![](./figures/knn/time_comp.jpg)

### Second data Set
For the best fitting values of k and distance metrics I achieved an accuracy about 31.3% . 



## Linear Regression with L1 and L2 Regularization
### Theory

#### Linear Regression

Linear regression seeks to model the relationship between a dependent variable $ y $ and an independent variable $ X $ through the equation:

$ \hat{y} = X \cdot w + b $

where:
- $ \hat{y} $ is the predicted output.
- $ X $ is the input data.
- $ w $ is the weight vector.
- $ b $ is the bias term.

#### Loss Function

The goal of linear regression is to minimize the difference between the predicted output $ \hat{y} $ and the actual output $ y $. This difference is quantified using the Mean Squared Error (MSE):

$ \text{MSE}(w, b) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 $

#### Regularization

To prevent overfitting, regularization terms can be added to the loss function. Regularization discourages the model from learning excessively complex relationships, which can lead to poor generalization.

##### L2 Regularization

L2 regularization, also known as Ridge regression, penalizes the sum of the squared weights:

$ \text{Loss}_{L2} = \text{MSE}(w, b) + \lambda \cdot \frac{1}{2} \sum_{j=1}^{m} w_j^2 $

##### L1 Regularization

L1 regularization, also known as Lasso regression, penalizes the sum of the absolute values of the weights:

$ \text{Loss}_{L1} = \text{MSE}(w, b) + \lambda \cdot \sum_{j=1}^{m} |w_j| $

#### Training Process

The model is trained by minimizing the loss function, which includes the MSE and a regularization term. The weights and bias are updated iteratively using gradient descent:

$
\begin{aligned}
dw &= \frac{1}{n} X^T \cdot (\hat{y} - y) + \frac{\lambda}{n} \cdot \text{RegularizationTerm}(w) \\
db &= \frac{2}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)
\end{aligned}
$

#### Gradient Calculation

- **L2 Regularization**: The gradient $ dw $ includes a term $ \lambda \cdot w $.
- **L1 Regularization**: The gradient $ dw $ includes a term $ \lambda \cdot \text{sign}(w) $.

### 3.1 Linear Regression class

```
class lr:
    def __init__(self, learning_rate=0.01, epochs=10000, regularization_param=0,reg_type='L2',make_gif=False):
        Initializes the linear regression class

    def poly(self,X, degree):
        Concatenates power of the training data to the dataset
    
    def split(self,x,y,k=1):
        Splits the data into train,val and test data based on the degree of the data

    def fit(self, X, y, X_val=None, y_val=None): #Gradient Descent
        Fits a linear or polynomial line to the training data

    def validate(self,X_val,y_val):
        Checks loss on X_val prediction
    
    def predict(self, X): #Y = W.X + B
        Predicts y using the current weights and bias

    def mse(self, y, y_pred): 
        Computes Mean Square Error

    def var(self, y_pred): 
        Computes Variance
    
    def sd(self,y_pred):
        Computes standard deviation
    
    def plot_linear_regression_results(self,x, y, y_pred,i,mse_vals,sd_vals,var_vals):
       Plots the variance, mse, std and predicted line in a 4x4 plot
```

### 3.1.1 Degree 1

![](./figures/linear_reg/deg_1.jpg)

Plot for degree 1 and no regularization yields the above output.

###3.1.2 Degree>1

![](./figures/linear_reg/k_2.jpg)
![](./figures/linear_reg/k_3.jpg)
![](./figures/linear_reg/k_4.jpg)
![](./figures/linear_reg/k_5.jpg)

###3.1.3 Animation

![](./figures/linear_reg/linreg_k_4.gif)

The above gif is for a degree of 4.

### 3.2.1 Regularization

![](./figures/linear_reg/reg_L1.jpg)
The above plot is for L1 regularization with a regularization penalty ($\lambda$) = 25.
![](./figures/linear_reg/reg_L2.jpg)
The above plot is for L2 regularization with regularization penalty ($\lambda$) = 25.