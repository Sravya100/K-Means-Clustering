#!/usr/bin/env python
# coding: utf-8

# In[67]:


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
import seaborn as sns


# In[68]:


#Importing the dataset 
dataset = pd.read_csv('', sep= ' ', names=["sl", "sw", "pl", "pw"],header= None)


# In[69]:


dataset.head()


# In[71]:


#Checking if there are any null values in the dataset
null_count=dataset.isnull().sum()
null_count.describe()


# In[72]:


#Feature scaling
def normalize(arg):
    arg = (arg- np.min(arg))/ (np.max(arg) - np.min(arg))
    return arg


data = normalize(dataset).to_numpy(copy= True)


# In[73]:


print(data[0])


# In[74]:


#Dimensionality reduction using PCA
pca = PCA(n_components=2)
reduced_dataset = pca.fit_transform(data)
print(f"reduced_dataset shape: {reduced_dataset.shape}, first few entries: {reduced_dataset[:5]}")


# In[75]:


sns.scatterplot(x=reduced_dataset[:,0], y=reduced_dataset[:,1]);


# In[76]:


#K-Means implementation

k=3
rng = np.random.default_rng()

#create random centroids
centroids = np.zeros((k, data.shape[1]))
for i in range(k):
    random_index = int(150 * rng.random())
    centroids[i] = data[random_index]
print(f"Centroids: {centroids}")

centroids_reduced = pca.transform(centroids)
print(f"Centroids reduced by PCA: {centroids_reduced}")

counter = 0

old_centroids = np.zeros((k, data.shape[1]))
while np.linalg.norm(old_centroids - centroids)>0:
    clusters = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        distances = [np.linalg.norm(data[i] - c) for c in centroids]
        nearest_centroid = np.argmin(distances)
        clusters[i] = nearest_centroid
        
    clusters
    
    old_centroids = centroids.copy()
    centroids = np.zeros((k, data.shape[1]))
    for i in range(k):
        centroids[i] = np.average(data[clusters == i], axis=0)

    counter += 1
    
print(f"Finished {counter} iterations")
    
    


# In[77]:


#Data Visualization
centroids_reduced = pca.transform(centroids)
sns.scatterplot(x=reduced_dataset[:,0], y=reduced_dataset[:,1], hue=clusters, palette='Set1')
sns.scatterplot(x=centroids_reduced[:,0],y=centroids_reduced[:,1], color='black',marker='X')


# In[78]:


print (clusters)


