# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 21:02:00 2024

@author: seven
"""
import time
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the data for X
X = pd.read_csv(r'C:\\Users\\seven\\Documents\\Machine Learning\\hwk 3\\Long_Row_Daily_Returns_1747_to_1762.csv')

# Load the data for y
y = pd.read_csv(r'C:\\Users\\seven\\Documents\\Machine Learning\\hwk 3\\Z_column_vector.csv')
 
# Remove the first 15 entries from y
y = y.iloc[16:]

# Remove the last entry from y
y = y.iloc[:-1]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the RandomForestClassifier with specified parameters
rf_classifier = RandomForestClassifier(max_samples=1164, random_state=42)

# Start time
start_time = time.time()

# Fit the classifier to the training data
rf_classifier.fit(X_train, y_train.values.ravel())

# End time
end_time = time.time()

# Computing time
computing_time = end_time - start_time
print("Computing Time:", computing_time, "seconds")

# OOB accuracy
oob_accuracy = rf_classifier.oob_score
print("OOB Accuracy:", oob_accuracy)

# Max features used
max_features_used = rf_classifier.max_features
print("Max Features Used:", max_features_used)

# Predict the class for the unseen data
predicted_class = rf_classifier.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, predicted_class)
print("Accuracy:", accuracy)

# Map integer labels to class names
class_names = {0: 'Stable', 1: 'Gains', 2: 'Loss'}

# Map integer labels to class names in y_test
y_test_mapped = y_test.iloc[:, 0].map(class_names).values

# Map integer labels to class names in predicted_class
predicted_class_mapped = [class_names[i] for i in predicted_class]
# Get the list of class names
class_names = ['Stable', 'Gains', 'Loss']

# Compute the confusion matrix
conf_matrix = confusion_matrix(y_test_mapped, predicted_class_mapped, labels=class_names)

# Confusion matrix
conf_matrix = confusion_matrix(y_test_mapped, predicted_class_mapped, labels=class_names)

# Print confusion matrix with percentages
total_samples = len(y_test)
conf_matrix_percentage = conf_matrix / total_samples * 100
print("Confusion Matrix (Percentage of Total Samples):")
print(conf_matrix_percentage)


"""

# Initialize the RandomForestClassifier with default parameters
# Parameters:
# n_estimators: The number of trees in the forest. (default=100)
# criterion: The function to measure the quality of a split. (default='gini')
# max_depth: The maximum depth of the tree. (default=None)
# min_samples_split: The minimum number of samples required to split an internal node. (default=2)
# min_samples_leaf: The minimum number of samples required to be at a leaf node. (default=1)
# min_weight_fraction_leaf: The minimum weighted fraction of the sum total of weights required to be at a leaf node. (default=0.0)
#      max_features: The number of features to consider when looking for the best split. (default='auto')
# max_leaf_nodes: Grow trees with max_leaf_nodes in best-first fashion. (default=None)
# min_impurity_decrease: A node will be split if this split induces a decrease of the impurity greater than or equal to this value. (default=0.0)
# min_impurity_split: Threshold for early stopping in tree growth. (default=1e-7)
# bootstrap: Whether bootstrap samples are used when building trees. (default=True)
# oob_score: Whether to use out-of-bag samples to estimate the generalization accuracy. (default=False)
# n_jobs: The number of jobs to run in parallel for both fit and predict. (default=None)
# random_state: Controls both the randomness of the bootstrapping of the samples used when building trees (if bootstrap=True) and the sampling of the features to consider when looking for the best split at each node. (default=None)
# verbose: Controls the verbosity when fitting and predicting. (default=0)
# warm_start: When set to True, reuse the solution of the previous call to fit and add more estimators to the ensemble, otherwise, just fit a whole new forest. (default=False)
# class_weight: Weights associated with classes in the form {class_label: weight}. (default=None)
# ccp_alpha: Complexity parameter used for Minimal Cost-Complexity Pruning. (default=0.0)

"""
 





