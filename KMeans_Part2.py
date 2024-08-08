#!/usr/bin/env python
# coding: utf-8

# In[135]:


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
import seaborn as sns


# In[136]:


#Importing the dataset 
dataset = pd.read_csv('',header= None)


# In[137]:


#Handling Null data
dataset=dataset.replace(np.NaN,0)
dataset.head()


# In[138]:


#Checking if there are any null values in the dataset
null_count=dataset.isnull().sum()
null_count.describe()


# In[140]:


#Feature Scaling
data = (dataset.to_numpy(copy= True))/255


# In[141]:


print(data[0])


# In[142]:


#Dimensionality reduction using PCA
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(data)
print(f"reduced_dataset shape: {reduced_data.shape}, first few entries: {reduced_data[:5]}")


# In[143]:


sns.scatterplot(x=reduced_data[:,0], y=reduced_data[:,1]);


# In[145]:


#K-Means implementation

k=10
rng = np.random.default_rng()

#create random centroids
centroids = np.zeros((k, data.shape[1]))
for i in range(k):
    random_index = int(150 * rng.random())
    centroids[i] = data[random_index]
print(f"Centroids: {centroids}")

#Reducing the dimensions of centroid for visualizing
centroids_reduced = pca.transform(centroids)
print(f"Centroids reduced by PCA: {centroids_reduced}")

counter = 0

#Clustering of data points based on closest centroid
old_centroids = np.zeros((k, data.shape[1]))
while np.linalg.norm(old_centroids - centroids)>0:
    clusters = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        distances = [np.linalg.norm(data[i] - c) for c in centroids]
        nearest_centroid = np.argmin(distances)
        clusters[i] = nearest_centroid
    
    old_centroids = centroids.copy()
    centroids = np.zeros((k, data.shape[1]))
    for i in range(k):
        centroids[i] = np.average(data[clusters == i], axis=0)

    counter += 1
    
print(f"Finished {counter} iterations")
    


# In[146]:


#Data Visualization
centroids_reduced = pca.transform(centroids)
sns.scatterplot(x=reduced_data[:,0], y=reduced_data[:,1], hue=clusters, palette='Spectral')
sns.scatterplot(x=centroids_reduced[:,0],y=centroids_reduced[:,1], color='black',marker='X')


