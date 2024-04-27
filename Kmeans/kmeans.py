import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the retail dataset (Hypothetical data)
data = {
    'CustomerID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'AnnualIncome': [30000, 32000, 50000, 80000, 90000, 100000, 110000, 130000, 150000, 200000],
    'SpendingScore': [30, 28, 60, 85, 90, 78, 75, 90, 95, 40]
}

df = pd.DataFrame(data)

# Select features for clustering (AnnualIncome and SpendingScore)
X = df[['AnnualIncome', 'SpendingScore']]
print(X)

# Feature scaling (Standardization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-means clustering
num_clusters = 4
kmeans = KMeans(n_clusters=num_clusters, random_state=42)

# cluster_labels = kmeans.fit_predict(X_scaled)
cluster_labels = kmeans.fit_predict(X.values)  # Convert DataFrame to NumPy array

# Get cluster centroids
centroids = kmeans.cluster_centers_
print(centroids)

# Add cluster labels to the DataFrame
df['Cluster'] = cluster_labels
print(df['Cluster'])

# Visualize the clusters and centroids
plt.figure(figsize=(8, 6))
colors = ['r', 'g', 'b', 'y']  # Added one more color for 4 clusters
for i in range(num_clusters):
    cluster_data = df[df['Cluster'] == i]
    plt.scatter(cluster_data['AnnualIncome'], cluster_data['SpendingScore'], c=colors[i], label=f'Cluster {i + 1}')

plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200, c='red', label='Cluster Centers')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('Customer Segmentation using K-means')
plt.legend()
plt.grid()
plt.show()
