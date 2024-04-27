import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Create a sample dataset
data = {
    'Annual_Income (k$)': [15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90],
    'Spending_Score (1-100)': [39, 81, 6, 77, 40, 6, 76, 3, 77, 13, 1, 10]
}

df = pd.DataFrame(data)

# Define the number of clusters (you can adjust this)
num_clusters = 3

# Select the features for clustering
X = df[['Annual_Income (k$)', 'Spending_Score (1-100)']]

# Initialize the K-means model
kmeans = KMeans(n_clusters=num_clusters, random_state=0)

# Fit the model to the data
kmeans.fit(X)

# Add cluster labels to the original dataset
df['Cluster'] = kmeans.labels_

# Visualize the results using a scatter plot
plt.figure(figsize=(10, 6))
for cluster in range(num_clusters):
    cluster_data = df[df['Cluster'] == cluster]
    plt.scatter(cluster_data['Annual_Income (k$)'], cluster_data['Spending_Score (1-100)'],
                label=f'Cluster {cluster + 1}')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='black', marker='X',
            label='Cluster Centers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Customer Segmentation')
plt.legend()
plt.grid(True)
plt.show()

# Display the segmented data
print("Segmented Customer Data:")
print(df)
