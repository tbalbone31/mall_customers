# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import numpy as np

# %%

# Load the dataset
data = pd.read_csv('data/Mall_Customers.csv')

# %%

# Preprocess the data
cleaned_data = data.drop(columns=['CustomerID'])

# Determine missing values
missing_values = cleaned_data.isnull().sum()

# Encode categorical variables
cleaned_data = pd.get_dummies(cleaned_data, columns=['Genre'],drop_first=True)

# standardise the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(cleaned_data)

# %%

# Determine the optimal number of clusters using the Elbow Method
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Method
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()
# %%

# %% 

# Determine the optimal number of clusters using the Silhouette Score
silhouette_scores = []
for k in range(2, 11):  # Silhouette Score requires at least 2 clusters
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(scaled_data)
    score = silhouette_score(scaled_data, cluster_labels)
    silhouette_scores.append(score)

# Plot the Silhouette Scores
plt.figure(figsize=(8, 5))
plt.plot(range(2, 11), silhouette_scores, marker='o', color='orange')
plt.title('Silhouette Score Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.show()
# %%

# Perform K-Means clustering with the optimal number of clusters
optimal_clusters = 6
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
clusters = kmeans.fit_predict(scaled_data)

# Add cluster labels to the original data
cleaned_data['Cluster'] = clusters

# Evaluate clustering quality
silhouette_avg = silhouette_score(scaled_data, clusters)
print(f"Silhouette Score: {silhouette_avg}")

davies_bouldin = davies_bouldin_score(scaled_data, clusters)
print(f"Davies-Bouldin Index: {davies_bouldin}")

# %%

# Visualize the clusters using two original features
plt.figure(figsize=(8, 5))
sns.scatterplot(
    x=cleaned_data['Annual Income (k$)'],  # Replace with a relevant feature
    y=cleaned_data['Spending Score (1-100)'],  # Replace with another relevant feature
    hue=clusters,
    palette='tab10'  # Use a categorical palette like 'tab10'
)
plt.title('Clusters Visualization (Original Features)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Cluster')
plt.show()

# %%

# Visualize the clusters with centroids using original features
plt.figure(figsize=(8, 5))
sns.scatterplot(
    x=cleaned_data['Annual Income (k$)'],
    y=cleaned_data['Spending Score (1-100)'],
    hue=clusters,
    palette='tab10'
)

# Add centroids (convert from scaled space to original space)
centroids_original = scaler.inverse_transform(kmeans.cluster_centers_)

# Plot centroids
plt.scatter(
    centroids_original[:, cleaned_data.columns.get_loc('Annual Income (k$)')],  # X-coordinate
    centroids_original[:, cleaned_data.columns.get_loc('Spending Score (1-100)')],  # Y-coordinate
    s=200, c='red', label='Centroids', marker='X'
)

# Add labels to centroids with an offset
dx, dy = 5, 5  # Offset values for x and y
for i, (x, y) in enumerate(zip(
    centroids_original[:, cleaned_data.columns.get_loc('Annual Income (k$)')],
    centroids_original[:, cleaned_data.columns.get_loc('Spending Score (1-100)')]
)):
    plt.text(x + dx, y + dy, f'Cluster {i}', color='black', fontsize=10, ha='center', va='center')

plt.title('Clusters Visualization with Centroids (Original Features)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Cluster')
plt.show()

# %%

# Visualize the clusters using PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)
print("Explained Variance Ratio:", pca.explained_variance_ratio_)
print("Total Explained Variance:", sum(pca.explained_variance_ratio_))

plt.figure(figsize=(8, 5))
sns.scatterplot(x=pca_data[:, 0], y=pca_data[:, 1], hue=clusters, palette='viridis')
plt.title('Clusters Visualization (PCA)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title='Cluster')
plt.show()
# %%

# Perform DBSCAN clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)  # Adjust `eps` and `min_samples` as needed
dbscan_clusters = dbscan.fit_predict(scaled_data)

# Add DBSCAN cluster labels to the original data
cleaned_data['DBSCAN_Cluster'] = dbscan_clusters

# Evaluate DBSCAN clustering quality
if len(set(dbscan_clusters)) > 1:  # Ensure there are at least 2 clusters
    dbscan_silhouette = silhouette_score(scaled_data, dbscan_clusters)
    print(f"DBSCAN Silhouette Score: {dbscan_silhouette}")
else:
    print("DBSCAN did not form enough clusters to calculate a Silhouette Score.")

# Visualize DBSCAN clusters
plt.figure(figsize=(8, 5))
sns.scatterplot(
    x=cleaned_data['Annual Income (k$)'],
    y=cleaned_data['Spending Score (1-100)'],
    hue=dbscan_clusters,
    palette='tab10'
)
plt.title('DBSCAN Clusters (Original Features)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Cluster')
plt.show()
# %%

# Perform hierarchical clustering
linkage_matrix = linkage(scaled_data, method='ward')  # Use Ward's method for minimizing variance

# Plot dendrogram
plt.figure(figsize=(10, 7))
dendrogram(linkage_matrix)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()

# Cut the dendrogram to form clusters
hierarchical_clusters = fcluster(linkage_matrix, t=6, criterion='maxclust')  # Adjust `t` for the number of clusters

# Add hierarchical cluster labels to the original data
cleaned_data['Hierarchical_Cluster'] = hierarchical_clusters

# Evaluate hierarchical clustering quality
hierarchical_silhouette = silhouette_score(scaled_data, hierarchical_clusters)
print(f"Hierarchical Clustering Silhouette Score: {hierarchical_silhouette}")

# Visualize hierarchical clusters
plt.figure(figsize=(8, 5))
sns.scatterplot(
    x=cleaned_data['Annual Income (k$)'],
    y=cleaned_data['Spending Score (1-100)'],
    hue=hierarchical_clusters,
    palette='tab10'
)
plt.title('Hierarchical Clusters (Original Features)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Cluster')
plt.show()
# %%
