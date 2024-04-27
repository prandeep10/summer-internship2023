import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate a random seed for reproducibility
np.random.seed(42)

# Generate a realistic dataset
X, y = make_classification(n_samples=200, n_features=2, n_informative=2, n_redundant=0, n_classes=2,
                           n_clusters_per_class=1, class_sep=1.0, random_state=42)

# Create a DataFrame from the dataset
data = pd.DataFrame({'X1': X[:, 0], 'X2': X[:, 1], 'y': y})

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Plot the data points
plt.scatter(data[data['y'] == 0]['X1'], data[data['y'] == 0]['X2'], label='Class 0', marker='o')
plt.scatter(data[data['y'] == 1]['X1'], data[data['y'] == 1]['X2'], label='Class 1', marker='x')

plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.title('Realistic Dataset')
plt.grid(True)
plt.show()
