#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[4]:


import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Dataset input
train_data = pd.read_csv('C:/Users/RAMKRISHNA/Downloads/titanic/train.csv')
test_data = pd.read_csv('C:/Users/RAMKRISHNA/Downloads/titanic/test.csv')

# Removing irrelevant data (columns)
columns_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
train_data.drop(columns_to_drop, axis=1, inplace=True)

# Fill missing values in the train_data and test_data DataFrames
train_data['Age'].fillna(train_data['Age'].mean(), inplace=True)
train_data['Fare'].fillna(train_data['Fare'].median(), inplace=True)
train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)

# Perform one-hot encoding on the 'Sex' and 'Embarked' columns
train_data = pd.get_dummies(train_data, columns=['Sex', 'Embarked'])

# Extract features and target variable
x_train = train_data.drop('Survived', axis=1)
y_train = train_data['Survived']

# Train the Naive Bayes model
nb_model = GaussianNB()
nb_model.fit(x_train, y_train)

# Train the K-Nearest Neighbors model
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(x_train, y_train)

# Make predictions on the train set
nb_train_predictions = nb_model.predict(x_train)
knn_train_predictions = knn_model.predict(x_train)

# Calculate accuracy on the train set
nb_train_accuracy = accuracy_score(y_train, nb_train_predictions)
knn_train_accuracy = accuracy_score(y_train, knn_train_predictions)

print("Naive Bayes Train Accuracy:", nb_train_accuracy)
print("K-Nearest Neighbors Train Accuracy:", knn_train_accuracy)

# Split the data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Make predictions on the validation set
nb_val_predictions = nb_model.predict(x_val)
knn_val_predictions = knn_model.predict(x_val)

# Calculate accuracy on the validation set
nb_val_accuracy = accuracy_score(y_val, nb_val_predictions)
knn_val_accuracy = accuracy_score(y_val, knn_val_predictions)

print("Naive Bayes Validation Accuracy:", nb_val_accuracy)
print("K-Nearest Neighbors Validation Accuracy:", knn_val_accuracy)

models = ['Naive Bayes', 'K-Nearest Neighbors']
accuracy_scores = [nb_val_accuracy, knn_val_accuracy]

plt.bar(models, accuracy_scores)
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Comparison of Model Accuracy')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




