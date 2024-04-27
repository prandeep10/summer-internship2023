import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# load the breast cancer dataset
X, y = load_breast_cancer(return_X_y=True)

# split the train and test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=23)

# LogisticRegression
clf = LogisticRegression(random_state=0)
clf.fit(X_train, y_train)

# Prediction for test data
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("Logistic Regression model accuracy (in %):", acc * 100)

# Plotting the accuracy graph
train_accuracy = clf.score(X_train, y_train)
test_accuracy = clf.score(X_test, y_test)

plt.figure(figsize=(8, 6))
plt.bar(['Train Accuracy', 'Test Accuracy'], [train_accuracy, test_accuracy], color=['blue', 'green'])
plt.ylim(0, 1.0)
plt.ylabel('Accuracy')
plt.title('Train and Test Accuracy of Logistic Regression')
plt.show()

# Random input prediction
random_input = np.random.rand(1, X.shape[1])  # Creating a random input of the same dimension as the features
predicted_output = clf.predict(random_input)

print("Random Input:", random_input)
print("Predicted Output:", predicted_output)
