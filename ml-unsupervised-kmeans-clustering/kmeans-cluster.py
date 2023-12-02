#!/usr/bin/python3
# This example demonstrates unsupervised learning
# by clustering data to groups by using kmeans clustering algorithm 

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


data = pd.read_csv("clustering_data.csv")
print(data.head())

# PCA algorithm is used to reduce the dimensionality of the data features to 2
pca = PCA(n_components=2)

# PCA fit_transform method is used to to find pre-specified number
# of principal components from the dataset and used them to transform the
# original higher-dimensional data to lower-dimension:
X_PCA = pca.fit_transform(data)
print("Dimensionality of the original dataset =", data.shape)
print("Dimensionality of the transformed dataset =", X_PCA.shape)

# Task 3, extract the amount of variance explained by each of the selected principal
print('Variance explained by each of the selected pricipal:', end=" ")
print(pca.explained_variance_)

# Plot reduced features as plotted on 2D
fig, ax = plt.subplots(2, 1, figsize=(10,10))
fig.tight_layout()
ax[0].scatter(X_PCA[:, 0], X_PCA[:, 1], marker='.', linewidth=2)

# Clustering analysis: from previously figure it is obvious to use 3.
kmeans3 = KMeans(init='random', n_clusters=3, n_init=10)
kmeans3.fit(X_PCA)
kmeans3.labels_[:3]
labels3 = kmeans3.labels_
u = kmeans3.cluster_centers_

# Plot clusters
ax[1].scatter(X_PCA[labels3==0,0], X_PCA[labels3==0,1], marker='.', linewidth=2, label='Cluster 0')
ax[1].scatter(X_PCA[labels3==1,0], X_PCA[labels3==1,1], marker='.', linewidth=2, label='Cluster 1')
ax[1].scatter(X_PCA[labels3==2,0], X_PCA[labels3==2,1], marker='.', linewidth=2, label='Cluster 2')
ax[1].scatter(u[:,0], u[:,1], marker='+', linewidth=3, color='black', label='Centroids')
ax[1].set_xlabel('Feature1', fontsize=12)
ax[1].set_ylabel('Feature2', fontsize=12)
ax[1].legend(fontsize=12)
ax[1].legend(loc='upper right')
plt.show()
