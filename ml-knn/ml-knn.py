#!/usr/bin/python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load the dataset (iris.csv) into the workspace and display its first five rows:
data = pd.read_csv("iris.csv")
print(data.head())

# Store the "variety" column of the dataset in another variable y , and then remove it from the dataset:
y = data.pop("variety")
print(type(y))

# Replace the string labels in y with numeric values: 0 for Setosa, 1 for Versicolor, and 2 for Verginica
y = y.replace(['Setosa', 'Versicolor', 'Virginica'], [0, 1, 2])

# Count the number of missing values for each feature:
print()
print("Number of missing values for each feature:")
print(data.isnull().sum())

# Apply median imputation to fill the null entries of the dataset:
impute = SimpleImputer(missing_values=np.nan, strategy='median')
X = impute.fit_transform(data)

# Construct the training and testing set as well as their corresponding label arrays:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y)

# Count and print the number of samples, features, and classes the training set contains:
N, p = X_train.shape
classes = np.unique(y)
print("Training set contains {} samples each with {} features.".format(N, p))
print("There are {} classes to be classified.\n".format(len(classes)))

# Convert the training set from Numpy array to Pandas DataFrame:
X_train = pd.DataFrame(data=X_train, columns=['sepal.length', 'sepal.width', 'petal.length', 'petal.width'])

# Extract some descriptive statistics from the training set:
print(data.describe())

# Generate two new features, sepal_area and petal_area, like this:
sepal_area = X_train['sepal.length'] * X_train['sepal.width']
petal_area = X_train['petal.length'] * X_train['petal.width']

# Illustrate the scatter plot of the training samples in the feature space built upon the two new features:
y_train_0 = np.where(y_train==0)
y_train_1 = np.where(y_train==1)
y_train_2 = np.where(y_train==2)

fig, ax = plt.subplots()
ax.scatter(sepal_area.loc[y_train_0], petal_area.loc[y_train_0], \
           label='Setosa', color='blue')
ax.scatter(sepal_area.loc[y_train_1], petal_area.loc[y_train_1], \
           label='Versicolor', color='orange')
ax.scatter(sepal_area.loc[y_train_2], petal_area.loc[y_train_2], \
           label='Virginica', color='green')
# matplotlib raw strings with some Mathtext formatting (which is a subset of Tex markup): 
ax.set_xlabel(r'$x_1$: Sepal length $\times$ width (cm$^2$)', fontsize=12)
ax.set_ylabel(r'$x_2$: Petal length $\times$ width (cm$^2$)', fontsize=12)
ax.set_title('Scatter plot of training samples')
ax.legend()

# Model fitting & training
# n_neighbors allows you to determine the number of neighbors, i.e., the value of K.
# Another useful parameter is metric which allows you to choose the distance metric of the algorithm.
# The default metric is the standard euclidean distance, but there are several more metrics which you can choose:
# manhattan :  Manhattan Distance
# chebyshev :  Chebyshev Distance
# minkowski : Minkowski Distance with parameter p (an integer larger than one)
# So, let's initialize a K-NN classification model in a variable knn with K=3 and the Euclidean distance:
knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn.fit(X_train, y_train)

# Lets train several models with different K-values
knn_list = []
for k in range(1, 11):
    knn_list.append(KNeighborsClassifier(n_neighbors=k, metric='euclidean'))
    knn_list[-1].fit(X_train, y_train)
    
############################
# Performance evaluation ####
##############################

# Training data accuracy
training_acc = knn.score(X_train, y_train)
print("Training data accuracy of K-NN classifier (K=3) = {:.4f}".format(training_acc))

# Testing data accuracy
test_acc = knn.score(X_test, y_test)
print("Test data accuracy of K-NN classifier (K=3) = {:.4f}".format(test_acc))

# Confusion Matrices
fig, ax = plt.subplots(1,2, figsize=(13,4.5))
predictions_test = knn.predict(X_test)
cm = confusion_matrix(y_test, predictions_test, labels=knn.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['Setosa', 'Versicolor', 'Verginica'])
disp.plot(ax=ax[0])
ax[0].set_title('Test confusion matrix (K=3)')
predictions_train = knn.predict(X_train)
cm = confusion_matrix(y_train, predictions_train, labels=knn.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['Setosa', 'Versicolor', 'Verginica'])
disp.plot(ax=ax[1])
ax[1].set_title('Train confusion matrix (K=3)')

# Accuracy for models with different K-values.
# Note that results can change between different runs.
# A model with larger K requires more computations to make predictions.
accuracy_k = []
for k in range(1, 11):
    accuracy_k.append(knn_list[k-1].score(X_test, y_test)*100)

plt.figure(figsize=(8, 4))
plt.plot(range(1, 11), accuracy_k, color='blue', linestyle='dashed', marker='o')
plt.xticks(range(1, 11), range(1, 11))
plt.title('Test accuracy of the $K$-NN classifer vs the value of $K$')
plt.xlabel('$K$');  plt.ylabel('Accuracy (%)')

# Plot everything:
plt.show()

