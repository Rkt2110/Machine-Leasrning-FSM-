#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def linear_regression(x_train, y_train, learning_rate=0.01, num_iterations=1000):
    # Data preparation
    x_train = pd.get_dummies(x_train, drop_first=True)  # Convert categorical variables to one-hot encoding
    
    # Feature normalization
    x_train = (x_train - x_train.mean()) / x_train.std()
    
    x_train = np.array(x_train)  # Convert DataFrame to NumPy array
    y_train = np.array(y_train)  # Convert DataFrame to NumPy array
    m = len(y_train)  # Number of training examples
    x_train = np.concatenate((np.ones((m, 1)), x_train), axis=1)  # Add a column of ones for bias term

    # Parameter initialization
    n_features = x_train.shape[1]  # Number of features
    theta = np.zeros(n_features)  # Initialize weight vector with zeros

    # Hypothesis function
    def hypothesis(X, theta):
        return np.dot(X, theta)

    # Cost function
    def cost(X, y, theta):
        predictions = hypothesis(X, theta)
        error = predictions - y
        cost = np.sum(error ** 2) / (2 * m)
        return cost

    # Gradient descent
    def gradient_descent(X, y, theta, learning_rate, num_iterations):
        costs = []
        for _ in range(num_iterations):
            predictions = hypothesis(X, theta)
            error = predictions - y
            gradient = np.dot(X.T, error) / m
            theta -= learning_rate * gradient
            costs.append(cost(X, y, theta))
        return theta, costs

    # Train the model
    theta, costs = gradient_descent(x_train, y_train, theta, learning_rate, num_iterations)

    # Plot the cost function convergence
    plt.plot(range(num_iterations), costs)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost Function Convergence')
    plt.show()

    return theta


dataset_dir = 'C:/Users/RAMKRISHNA/Downloads/Medical Price Dataset.csv'
df = pd.read_csv(dataset_dir)
x_train = df[['age', 'bmi', 'children', 'smoker', 'region']]
y_train = df['charges']

theta = linear_regression(x_train, y_train)
print(theta)

# Get the column names of the feature variables
feature_columns = x_train.columns

# Create a scatter plot for each feature-target pair
for feature in feature_columns:
    plt.figure()
    plt.scatter(x_train[feature], y_train)
    plt.xlabel(feature)
    plt.ylabel('charges')
    plt.title(f'Scatter plot: {feature} vs charges')
    plt.show()


# In[ ]:




