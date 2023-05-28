#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def kmeans_clustering(dataset):
    # Remove the last column (Species) and store it separately
    species = dataset['Species']
    data = dataset.drop('Species', axis=1)

    # Convert the DataFrame to a NumPy array
    X = data.values

    # Normalize the data
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    # Initialize the centroids randomly
    np.random.seed(0)
    centroids = X[np.random.choice(X.shape[0], size=k, replace=False)]

    # Assign data points to clusters
    clusters = np.zeros(len(X))
    for _ in range(max_iterations):
        # Calculate the Euclidean distances from each point to the centroids
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)

        # Assign each point to the closest centroid
        new_clusters = np.argmin(distances, axis=1)

        # Check if the assignment has changed
        if np.array_equal(clusters, new_clusters):
            break

        # Update the centroids
        for i in range(k):
            centroids[i] = np.mean(X[new_clusters == i], axis=0)

        clusters = new_clusters

    return clusters, species


def principal_component_analysis(dataset):
    # Remove the last column (Species) and store it separately
    species = dataset['Species']
    data = dataset.drop('Species', axis=1)

    # Convert the DataFrame to a NumPy array
    X = data.values

    # Normalize the data
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    # Compute the covariance matrix
    covariance_matrix = np.cov(X.T)

    # Compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # Sort the eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Project the data onto the first three eigenvectors
    projected_data = X.dot(eigenvectors[:, :3])

    # Plot the data points in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(projected_data[:, 0], projected_data[:, 1], projected_data[:, 2])

    # Print the eigenvalues
    print('Eigenvalues:', eigenvalues[:3])

    return species


# Load the dataset
dataset = pd.read_csv('C:/Users/RAMKRISHNA/Downloads/Iris Dataset.csv')

# Set the number of clusters and maximum iterations for K-Means
k = 3
max_iterations = 100

# Perform K-Means clustering
cluster_labels, species = kmeans_clustering(dataset)

# Map species to numeric values for color mapping
species_mapping = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
species_colors = species.map(species_mapping)

# Perform Principal Component Analysis
species_pca = principal_component_analysis(dataset)

# Plot the output of the cluster
plt.subplot(1, 2, 1)
plt.scatter(dataset['SepalLengthCm'], dataset['SepalWidthCm'], c=cluster_labels, cmap='viridis')
plt.title('Cluster Output')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')

# Plot the actual species
plt.subplot(1, 2, 2)
plt.scatter(dataset['SepalLengthCm'], dataset['SepalWidthCm'], c=species_colors, cmap='viridis')
plt.title('Actual Species')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')

# Show the plots
plt.tight_layout()
plt.show()


# In[ ]:




