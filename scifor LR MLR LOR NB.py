#!/usr/bin/env python
# coding: utf-8

# # Multiple Linear Regression

# In[15]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class MultipleLinearRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        for epoch in range(self.epochs):
            # Calculate predictions
            predictions = self.predict(X)

            # Calculate errors
            errors = predictions - y

            # Update weights and bias using gradient descent
            self.weights -= (self.learning_rate / num_samples) * np.dot(errors, X)
            self.bias -= (self.learning_rate / num_samples) * np.sum(errors)

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Generate some random data for demonstration purposes
np.random.seed(42)
X = 2 * np.random.rand(100, 2)  # Assuming 2 features for demonstration
y = 4 + 3 * X[:, 0] + 2 * X[:, 1] + np.random.randn(100)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = MultipleLinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model using Mean Squared Error
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error on test set: {mse}')


# # Logistic Regression

# In[16]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        for epoch in range(self.epochs):
            # Calculate predictions
            predictions = self.sigmoid(np.dot(X, self.weights) + self.bias)

            # Calculate errors
            errors = predictions - y

            # Update weights and bias using gradient descent
            self.weights -= (self.learning_rate / num_samples) * np.dot(errors, X)
            self.bias -= (self.learning_rate / num_samples) * np.sum(errors)

    def predict(self, X):
        return np.round(self.sigmoid(np.dot(X, self.weights) + self.bias))

# Generate some random binary classification data for demonstration purposes
np.random.seed(42)
X = 2 * np.random.rand(100, 2)  # Assuming 2 features for demonstration
y = (4 + 3 * X[:, 0] + 2 * X[:, 1] + np.random.randn(100)) > 0

# Convert y to binary (0 or 1)
y = y.astype(int)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model using accuracy
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy on test set: {accuracy}')


# # NaiveBayes

# In[17]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class NaiveBayes:
    def fit(self, X, y):
        self.classes, counts = np.unique(y, return_counts=True)
        self.class_probs = counts / len(y)
        self.mean = []
        self.std = []

        for c in self.classes:
            class_data = X[y == c]
            class_mean = np.mean(class_data, axis=0)
            class_std = np.std(class_data, axis=0)
            self.mean.append(class_mean)
            self.std.append(class_std)

    def predict(self, X):
        predictions = []

        for x in X:
            class_scores = []

            for i, c in enumerate(self.classes):
                # Calculate the likelihood of the features given the class using Gaussian distribution
                likelihood = np.prod(
                    (1 / (np.sqrt(2 * np.pi) * self.std[i])) * np.exp(-(x - self.mean[i]) ** 2 / (2 * self.std[i] ** 2))
                )

                # Calculate the posterior probability
                posterior = self.class_probs[i] * likelihood
                class_scores.append(posterior)

            # Predict the class with the highest posterior probability
            predicted_class = self.classes[np.argmax(class_scores)]
            predictions.append(predicted_class)

        return predictions

# Generate some random binary classification data for demonstration purposes
np.random.seed(42)
X = np.random.randn(100, 3)  # Assuming 3 features for demonstration
y = (2 * X[:, 0] + 3 * X[:, 1] + 0.5 * X[:, 2] + np.random.randn(100)) > 0

# Convert y to binary (0 or 1)
y = y.astype(int)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Naive Bayes model
model = NaiveBayes()
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy on test set: {accuracy}')


# # LinearRegression

# In[18]:


import numpy as np
class LinearRegression:
    
    def fit(self, X, y, lr = 0.001, epochs=10000, verbose=True, batch_size=1):
        X = self.add_bias(X)
        self.weights = np.zeros(len(X[0]))
        for i in range(epochs):
            idx = np.random.choice(len(X), batch_size) 
            X_batch, y_batch =  X[idx], y[idx]
            self.weights -= lr * self.get_gradient(X_batch, y_batch)
            if i % 1000 == 0 and verbose: 
                print('Iterations: %d - Error : %.4f' %(i, self.get_loss(X,y)))
                
    def predict(self, X):
        return self.predict_(self.add_bias(X))
    
    def get_loss(self, X, y):
        return np.mean((y - self.predict_(X)) ** 2)
    
    def predict_(self, X):
        return np.dot(X,self.weights)
    
    def add_bias(self,X):
        return np.insert(X, 0, np.ones(len(X)), axis=1)
        
    def get_gradient(self, X, y):
        return -1.0 * np.dot(y - self.predict_(X), X) / len(X)
    
    def evaluate(self, X, y):
        return self.get_loss(self.add_bias(X), y)


# In[19]:


from sklearn import datasets
from sklearn.metrics import mean_squared_error, r2_score

# Load the diabetes dataset
diabetes = datasets.load_diabetes()


# Use only one feature
diabetes_X = diabetes.data[:, np.newaxis, 2]

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

# Create linear regression object
regr = LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train, lr=0.1, batch_size=len(diabetes_X_train))

# Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X_test)


# In[20]:


print('Coefficients: \n', regr.weights)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))


# In[ ]:




