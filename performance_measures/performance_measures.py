import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
from sklearn.neighbors import KNeighborsClassifier
from models.knn.knn import KNN



class PerfMeeasures : 

    def __init__(self , y , y_pred , total_features = 114):
        self.y = y
        self.y_pred = y_pred
        self.features = np.arange(total_features)

    def accuracy(self):
        return (np.sum((self.y == self.y_pred))) / len(self.y)
    

    def class_accuracy(self):
        accuracy = []

        for feature in self.features:
            tp = np.sum((self.y == feature) & (self.y_pred == feature))
            total_actual = np.sum(self.y == feature)

            if total_actual ==0:
                acc =0

            else:
                acc = tp / total_actual 
            accuracy.append(acc)

        return np.array(accuracy)  
    

    def class_precision(self):
        precision= []
        for feature in self.features:
            fp = np.sum((self.y!= feature) & (self.y_pred == feature))
            tp = np.sum((self.y == feature) & (self.y_pred==feature))

            if((tp+fp)==0):
                prec_single_class = 0

            else:
                prec_single_class = (tp)/(tp+fp)
            
            precision.append(prec_single_class)
        
        return np.array(precision)


    def class_recall(self):
        recall= []
        for feature in self.features:
            fn = np.sum((self.y == feature) & (self.y_pred != feature))
            tp = np.sum((self.y == feature) & (self.y_pred==feature))

            if((tp+fn)==0):
                recall_single_class = 0

            else:
                recall_single_class = (tp)/(tp+fn)
            
            recall.append(recall_single_class)
        
        return recall
    
    def f1_score_class(self):
        precision = self.class_precision()
        recall = self.class_recall()

        f1_scores = []

        for prec , rec in zip(precision , recall):
            
            if((prec + rec) ==0 ):
                f1_single_class = 0

            else :
                f1_single_class = 2*prec*rec/(prec+rec)

            f1_scores.append(f1_single_class)
        
        return f1_scores

    def f1_score_all(self , mode = "macro"):
        if(mode=="macro"):
            f1_scores = self.class_accuracy()
            return np.mean(f1_scores)

        elif (mode=="micro"):
            
            # code for combined tp,fp and fn gotten from LLM
            combined_fp = np.sum([np.sum((self.y_pred == feature) and (self.y != feature)) for feature in self.features])
            combined_fn = np.sum([np.sum((self.y == feature) and (self.y_pred != feature)) for feature in self.features])
            combined_tp = np.sum([np.sum((self.y == feature) and (self.y_pred == feature)) for feature in self.features])

            if((combined_fp + combined_tp) == 0):
                precision_micro = 0 
            else :
                precision_micro = combined_tp / (combined_tp + combined_fp)
            
            if((combined_tp + combined_fn) == 0 ):
                recall_micro = 0 
            else: 
                recall_micro = combined_tp / (combined_tp + combined_fn) 
            

            if((precision_micro + recall_micro)==0):
                f1_micro =  0
            else :
                f1_micro = 2 * (precision_micro * recall_micro) / (precision_micro + recall_micro)


            return f1_micro

    def building_confusion_matrix (self):
        total_features = len(self.features)    
        confusion_matrix = np.zeros((total_features, total_features) , dtype= int)

        for true , pred in zip(self.y , self.y_pred):
            confusion_matrix[true , pred]+=1 

        return confusion_matrix

    def plot_conf_matrix(self, save_path=None, top_classes=10):

        # Generate the confusion matrix
        confusion_matrix = self.building_confusion_matrix()

        # Sum the rows of the confusion matrix to get the total predictions per class
        sum_per_class = np.sum(confusion_matrix, axis=1)

        # Identify the indices of the top N classes with the most predictions
        top_class_indices = np.argsort(sum_per_class)[-top_classes:]

        # Create a reduced confusion matrix containing only the top N classes
        reduced_matrix = confusion_matrix[np.ix_(top_class_indices, top_class_indices)]

        # Retrieve the names of the top N classes based on the indices
        selected_classes = [self.features[idx] for idx in top_class_indices]

        # Plot the confusion matrix using a heatmap
        plt.figure(figsize=(14, 12))
        sns.heatmap(reduced_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=selected_classes, yticklabels=selected_classes,
                    annot_kws={"size": 8}, cbar_kws={"shrink": 0.8})

        plt.xlabel('Predicted Class', fontsize=12)
        plt.ylabel('Actual Class', fontsize=12)
        plt.title(f'Top {top_classes} Confusion Matrix', fontsize=14)

        # Save the plot if a save path is provided, otherwise show the plot
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        else:
            plt.show()

        # Close the plot to free up memory
        plt.close()



    def measure_the_inference_time_for_plotting(self , model , X_test):
        
        start_time = time.time()
        model.predict(X_test)
        end_time = time.time()
        print(f"Time taken = {end_time-start_time}\n")
        return end_time - start_time

    def compare_inference_durations(self, init_k,  init_metric, 
                                optimal_k, optimal_metric, 
                                tuned_k,  tuned_metric, 
                                X_train, y_train, X_test, 
                                save_plot_path=None):

        # Initialize different KNN models with varying parameters
        initial_knn_model = KNN(k=init_k,distance_metric=init_metric)
        tuned_knn_model = KNN(k=tuned_k, distance_metric=tuned_metric)

        optimized_knn = KNN(k = optimal_k , distance_metric= optimal_metric)

        # Initialize the default KNN model from sklearn
        sklearn_knn_model = KNeighborsClassifier()

        # Train the models using the training data
        initial_knn_model.fit(X_train, y_train)
        optimized_knn.fit(X_train, y_train)
        tuned_knn_model.fit(X_train, y_train)
        sklearn_knn_model.fit(X_train, y_train)

        # Measure inference time for each model
        inference_time_initial = self.measure_the_inference_time_for_plotting(initial_knn_model, X_test)
        inference_time_optimal = self.measure_the_inference_time_for_plotting(optimized_knn, X_test)
        inference_time_tuned = self.measure_the_inference_time_for_plotting(tuned_knn_model, X_test)
        inference_time_sklearn = self.measure_the_inference_time_for_plotting(sklearn_knn_model, X_test)

        # Prepare data for plotting
        model_names = ['Initial KNN', 'Optimal KNN', 'Tuned KNN', 'sklearn KNN']
        inference_times = [inference_time_initial, inference_time_optimal, inference_time_tuned, inference_time_sklearn]

        # Plot the inference times
        plt.figure(figsize=(10, 6))
        bar_colors = ['navy', 'green', 'orange', 'purple']
        bars = plt.bar(model_names, inference_times, color=bar_colors)
        plt.xlabel('KNN Model')
        plt.ylabel('Inference Time (seconds)')
        plt.title('Comparison of Inference Times for Different KNN Configurations')

        # Annotate the bars with the inference times
        for bar, name, time in zip(bars, model_names, inference_times):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{time:.4f} s', 
                    ha='center', va='bottom', fontsize=10)

        # Save the plot if a path is provided
        if save_plot_path:
            plt.savefig(save_plot_path)

        # Display the plot
        plt.show()



    def measure_inference_time_vs_size(self, model, sizes ,X_train , y_train,  X_test):
        times = []
        for size in sizes:
            X_train_subset = X_train[:size]
            y_train_subset = y_train[:size]
            model.fit(X_train_subset, y_train_subset)

            time_this_case = self.measure_the_inference_time_for_plotting(model=model , X_test=X_test)
            times.append(time_this_case)

        return times
    
    def plot_inference_time_vs_size(self, initial_k, initial_p, initial_metric, 
                                     best_k, best_p, best_metric, 
                                     optimized_k, optimized_p, optimized_metric, 
                                     X_train, y_train, X_test,
                                     image_save_path=None):

        initial_knn = KNN(k=initial_k, p=initial_p, distance_metric=initial_metric)
        best_knn = KNN(k=best_k, p=best_p, distance_metric=best_metric)
        optimized_knn = KNN(k=optimized_k, p=optimized_p, distance_metric=optimized_metric)
        default_sklearn_knn = KNeighborsClassifier()

        sizes = np.linspace(1000, len(X_train), num=10, dtype=int)

        times_initial_knn = self.measure_inference_time_vs_size(initial_knn, sizes, X_train , y_train , X_test)
        times_best_knn = self.measure_inference_time_vs_size(best_knn, sizes,X_train , y_train , X_test)
        times_optimized_knn = self.measure_inference_time_vs_size(optimized_knn, sizes,X_train , y_train , X_test)
        times_default_sklearn_knn = self.measure_inference_time_vs_size(default_sklearn_knn, sizes,X_train , y_train , X_test)


        plt.figure(figsize=(12, 8))
        plt.plot(sizes, times_initial_knn, label=f'Initial KNN (k={initial_k}, p={initial_p}, metric={initial_metric})', linestyle='-', color='blue')
        plt.plot(sizes, times_best_knn, label=f'Best KNN (k={best_k}, p={best_p}, metric={best_metric})', linestyle='-', color='green')
        plt.plot(sizes, times_optimized_knn, label=f'Optimized KNN (k={optimized_k}, p={optimized_p}, metric={optimized_metric})', linestyle='-', color='red')
        plt.plot(sizes, times_default_sklearn_knn, label='Default sklearn KNN', linestyle='-', color='purple')

        plt.xlabel('Training Dataset Size')
        plt.ylabel('Inference Time (seconds)')
        plt.title('Inference Time vs Training Dataset Size')
        plt.legend()
        
        if image_save_path:
            plt.savefig(image_save_path)
        
        plt.show()

