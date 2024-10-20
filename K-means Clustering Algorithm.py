#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Creating a sample dataset of customers' purchase history
data = {
    'Customer_ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Total_Purchases': [5, 15, 25, 35, 45, 10, 20, 30, 40, 50],
    'Total_Spending': [500, 1500, 2500, 3500, 4500, 1000, 2000, 3000, 4000, 5000],
    'Average_Purchase_Value': [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
}

df = pd.DataFrame(data)

# Step 2: Dropping Customer_ID and preparing the feature matrix
X = df[['Total_Purchases', 'Total_Spending']]

# Step 3: Using the elbow method to find the optimal number of clusters
wcss = []  # Within-cluster sum of squares

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plotting the elbow method to determine the optimal number of clusters
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method to Determine Optimal Number of Clusters')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Step 4: Assuming the optimal number of clusters is 3
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)

# Step 5: Adding the cluster labels to the original dataframe
df['Cluster'] = y_kmeans
print(df)

# Step 6: Visualizing the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Total_Purchases', y='Total_Spending', hue='Cluster', data=df, palette='viridis', s=100)
plt.title('Customer Clusters Based on Purchase History')
plt.xlabel('Total Purchases')
plt.ylabel('Total Spending')
plt.show()

# Optional: Display the first 10 customers with their cluster labels
print(df[['Customer_ID', 'Total_Purchases', 'Total_Spending', 'Cluster']])


# In[ ]:




