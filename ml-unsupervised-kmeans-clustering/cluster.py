#!/usr/bin/python3
# This example demonstrated unsupervised learning
# by clustering data to groups by using kmeans clustering algorithm 

import pandas as pd
from sklearn.decomposition import PCA

data = pd.read_csv("clustering_data.csv")
print(data.head())

# Task 2, implement the PCA algorithm to reduce the dimensionality of the data to 2
pca = PCA(n_components=2)

# You should use the fit_transform method in order to get PCA to find the pre-specified number
# of principal components from the dataset and use them transform the original higher-dimensional data
# into their lower-dimensional samples:
X_PCA = pca.fit_transform(data)
print("Dimensionality of the original dataset =", data.shape)
print("Dimensionality of the transformed dataset =", X_PCA.shape)

# Task 3, extract the amount of variance explained by each of the selected principal
print('Variance explained by each of the selected pricipal:', end=" ")
print(pca.explained_variance_)

# Task 4, plot
# Plot
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.scatter(X_PCA[:, 0], X_PCA[:, 1], marker='.', linewidth=2)
ax.set_xlabel('PC1', fontsize=12)
ax.legend(fontsize=12)
plt.show()

# Task 5, perform a clustering analysis on this data. give two suggestions for the number of clusters.
# From previously drawn figure it seem to be clear to use 3.
# Testing with also values 2, 4 and 5, shows that value 5 is better than 2 and 4. ...Eventhough, value 3 looks the best.

# Task 6, perform k-means clustering algorithm for selected cluster numbers (2 & 3)
from sklearn.cluster import KMeans
#kmeans2 = KMeans(init='random', n_clusters=2, n_init=10)
#kmeans2.fit(X_PCA)
kmeans3 = KMeans(init='random', n_clusters=3, n_init=10)
kmeans3.fit(X_PCA)
#kmeans4 = KMeans(init='random', n_clusters=4, n_init=10)
#kmeans4.fit(X_PCA)
kmeans5 = KMeans(init='random', n_clusters=5, n_init=10)
kmeans5.fit(X_PCA)

# Task 7, plot clusters for selected cluster numbers 
# For k = 2
#kmeans2.labels_[:2]
#labels2 = kmeans2.labels_
#u = kmeans2.cluster_centers_

#plt.figure()
#plt.scatter(X_PCA[labels2==0,0], X_PCA[labels2==0,1], marker='.', linewidth=2, label='Cluster 0')
#plt.scatter(X_PCA[labels2==1,0], X_PCA[labels2==1,1], marker='.', linewidth=2, label='Cluster 1')
#plt.scatter(u[:,0], u[:,1], marker='*', linewidth=3, color='black', label='Centroids')
#plt.xlabel('Feature1', fontsize=12)
#plt.ylabel('Feature2', fontsize=12)
#plt.legend(fontsize=12)
#plt.show()

# For k = 3
kmeans3.labels_[:3]
labels3 = kmeans3.labels_
u = kmeans3.cluster_centers_

plt.figure()
plt.scatter(X_PCA[labels3==0,0], X_PCA[labels3==0,1], marker='.', linewidth=2, label='Cluster 0')
plt.scatter(X_PCA[labels3==1,0], X_PCA[labels3==1,1], marker='.', linewidth=2, label='Cluster 1')
plt.scatter(X_PCA[labels3==2,0], X_PCA[labels3==2,1], marker='.', linewidth=2, label='Cluster 2')
plt.scatter(u[:,0], u[:,1], marker='*', linewidth=3, color='black', label='Centroids')
plt.xlabel('Feature1', fontsize=12)
plt.ylabel('Feature2', fontsize=12)
plt.legend(fontsize=12)
plt.show()

# For k = 4
#kmeans4.labels_[:4]
#labels4 = kmeans4.labels_
#u = kmeans4.cluster_centers_

#plt.figure()
#plt.scatter(X_PCA[labels4==0,0], X_PCA[labels4==0,1], marker='.', linewidth=2, label='Cluster 0')
#plt.scatter(X_PCA[labels4==1,0], X_PCA[labels4==1,1], marker='.', linewidth=2, label='Cluster 1')
#plt.scatter(X_PCA[labels4==2,0], X_PCA[labels4==2,1], marker='.', linewidth=2, label='Cluster 2')
#plt.scatter(X_PCA[labels4==3,0], X_PCA[labels4==3,1], marker='.', linewidth=2, label='Cluster 3')
#plt.scatter(u[:,0], u[:,1], marker='*', linewidth=3, color='black', label='Centroids')
#plt.xlabel('Feature1', fontsize=12)
#plt.ylabel('Feature2', fontsize=12)
#plt.legend(fontsize=12)
#plt.show()

# For k = 4
kmeans5.labels_[:5]
labels5 = kmeans5.labels_
u = kmeans5.cluster_centers_

plt.figure()
plt.scatter(X_PCA[labels5==0,0], X_PCA[labels5==0,1], marker='.', linewidth=2, label='Cluster 0')
plt.scatter(X_PCA[labels5==1,0], X_PCA[labels5==1,1], marker='.', linewidth=2, label='Cluster 1')
plt.scatter(X_PCA[labels5==2,0], X_PCA[labels5==2,1], marker='.', linewidth=2, label='Cluster 2')
plt.scatter(X_PCA[labels5==3,0], X_PCA[labels5==3,1], marker='.', linewidth=2, label='Cluster 3')
plt.scatter(X_PCA[labels5==4,0], X_PCA[labels5==4,1], marker='.', linewidth=2, label='Cluster 4')
plt.scatter(u[:,0], u[:,1], marker='*', linewidth=3, color='black', label='Centroids')
plt.xlabel('Feature1', fontsize=12)
plt.ylabel('Feature2', fontsize=12)
plt.legend(fontsize=12)
plt.show()
