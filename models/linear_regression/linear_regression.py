import numpy as np
import matplotlib.pyplot as plt

class lr:
    def __init__(self, learning_rate=0.01, epochs=10000, regularization_param=0,reg_type='L2',make_gif=False):
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = None
        self.epochs = epochs
        self.lambda_reg = regularization_param
        self.make_gif=make_gif
        self.reg_type=reg_type

    def poly(self,X, degree):
        X_poly = np.hstack([X**p for p in range(1, degree + 1)]) #makes it of the form X^p, X^p-1, ...
        return X_poly
    
    def split(self,x,y,k=1):
        self.k=k
        train_size = int(0.8 * len(x))
        val_size = int(0.1 * len(x))

        x=self.poly(x,k) #fitting higher degrees of x

        X_train, y_train = x[:train_size], y[:train_size] #0-80%
        X_val, y_val = x[train_size:train_size + val_size], y[train_size:train_size + val_size] #80-90%
        X_test, y_test = x[train_size + val_size:], y[train_size + val_size:] #90-100%
    
        return X_train, y_train, X_val, y_val, X_test, y_test

    def fit(self, X, y, X_val=None, y_val=None): #Gradient Descent
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)

        self.bias = 0.01  # usually small value

        mod_val=50 #for number of images to generate for gif

        i = 0
        mse_vals,sd_vals,var_vals=[],[],[]

        while i < self.epochs:
            y_predicted = X @ self.weights + self.bias  # y_predicted = WX + b
            res = y_predicted - y #error

            if self.make_gif==True and i%mod_val==0 and i<=mod_val*100:
                self.plot_linear_regression_results(X,y,y_predicted,i,mse_vals,sd_vals,var_vals) #generating image for each epoch

            dw=0

            if self.reg_type=="L2":
                dw = (X.T @ res) / n_samples + (self.lambda_reg / n_samples) * self.weights  # dw = 1/n * (res).xT + lambda*w/n
            else:
                dw = (X.T @ res) / n_samples + (self.lambda_reg / n_samples) * np.sign(self.weights) # dw = 1/n*(res).xT + lambda*sign(w)/n

            db = 2*np.mean(res)  # db = sum(res)/n
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db


            # if i%500==0:
            #     print(f' on iter {i} loss: {loss}')
            
            i += 1

    def validate(self,X_val,y_val):
        y_val_pred=X_val@self.weights + self.bias
        loss = np.mean((y_val_pred - y_val) ** 2)
        return loss
    
    def predict(self, X): #Y = W.X + B
        return np.dot(X, self.weights) + self.bias

    def mse(self, y, y_pred): #Mean Square Error
        return np.mean((y - y_pred) ** 2)

    def var(self, y_pred): #Variance
        return np.var(y_pred)
    
    def sd(self,y_pred): #Standard deviation
        return np.std(y_pred)
    
    def plot_linear_regression_results(self,x, y, y_pred,i,mse_vals,sd_vals,var_vals):
        # Create a 2x2 subplot structure (4 plots in total)
        fig, axs = plt.subplots(2, 2, figsize=(8, 8))  # 4x4 inches per subplot


        #Plot code given by LLM

        # Plot 1: MSE over iterations/epochs (assuming mse() returns an array/list)
        mse_value = self.mse(y, y_pred)
        mse_vals.append(mse_value)
        axs[0, 0].plot(mse_vals, marker='o', color='blue')
        axs[0, 0].set_title('Mean Squared Error')
        axs[0, 0].set_xlabel('Iteration/Epoch')
        axs[0, 0].set_ylabel('MSE')

        # Plot 2: Standard Deviation (assuming sd() returns an array/list)
        sd_value = self.sd(y_pred)
        sd_vals.append(sd_value)
        axs[0, 1].plot(sd_vals, marker='o', color='green')
        axs[0, 1].set_title('Standard Deviation')
        axs[0, 1].set_xlabel('Iteration/Epoch')
        axs[0, 1].set_ylabel('Standard Deviation')

        # Plot 3: Variance (assuming var() returns an array/list)
        var_value = self.var(y_pred)
        var_vals.append(var_value)
        axs[1, 0].plot(var_vals, marker='o', color='orange')
        axs[1, 0].set_title('Variance')
        axs[1, 0].set_xlabel('Iteration/Epoch')
        axs[1, 0].set_ylabel('Variance')

        # Plot 4: Line fit to the data
        axs[1, 1].scatter(x[:,0], y, label='Actual Data', color='blue')
        sorted_idx = np.argsort(x[:, 0])
        plt.plot(x[:,0][sorted_idx], y_pred[sorted_idx], color='red', label='Model Prediction')
        axs[1, 1].set_title('Line Fit to Data')
        axs[1, 1].set_xlabel('X')
        axs[1, 1].set_ylabel('Y')
        axs[1, 1].legend()

        # Adjust the layout
        plt.tight_layout()

        # Save the plot without displaying
        plt.savefig(f"./figures/gif_images/image_{i}.jpg")

        # Clear the plot to avoid overlapping in subsequent plots
        plt.clf()
