### This example demonstrates basic usage of the python3 scikit-learn unsupervised clustering
At the first step PCA (Principal component analysis) fit_transform method is used to to reduce the dimensionality of the data (features).
At the second step k-means clustering method is used to cluster the data. In the documentations it is said that in very high-dimensional
spaces, Euclidean distances tend to become inflated (this is an instance of the so-called “curse of dimensionality”) and so reduction of
dimensionality can alleviate this problem and speed up computation.
source: https://scikit-learn.org/stable/modules/clustering.html#k-means

In this example k-means clustering works nicely as blobs are isotropically distributed with equal(ish) variance and size.

#### USAGE:
_python3 kmeans-cluster.py_
