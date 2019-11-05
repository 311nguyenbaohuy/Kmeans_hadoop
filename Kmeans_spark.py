#!/usr/bin/env python
# coding: utf-8

# In[294]:


import numpy as np
import datetime
import random

from pyspark.mllib.random import RandomRDDs
from pyspark import SparkContext
import matplotlib.pyplot as plt


# In[295]:


def customSplit(row):
    values = row[0]
    index = row[1]
    dataItems = values.split(',')
    for i in range(len(dataItems) - 1):
        dataItems[i] = float(dataItems[i])
    return (index, dataItems)


# In[296]:


def loadData(path):
    dataFromText = sc.textFile(path)
    data = dataFromText.filter(lambda x: x is not None).filter(lambda x: x != "")
    dataZipped = data.zipWithIndex()
    return dataZipped.map(lambda x: customSplit(x))


# In[297]:


def initCentroids(data, numClusters):
    sample = sc.parallelize(data.takeSample(False, numClusters))
    centroids = sample.map(lambda point: point[1][:-1])
    return centroids.zipWithIndex().map(lambda point: (point[1], point[0]))


# In[298]:


def assginToCluster(data, centroids):
    # Calculate Cartesian product of centroids and data
    cartesianData = centroids.cartesian(data)
    cartesianDataDistances = cartesianData.map(lambda dataPoint: calculateDistance(dataPoint[0], dataPoint[1]))
    dataMinDistance = cartesianDataDistances.groupByKey().map(lambda x: (x[0], list(x[1]))).map(lambda row: minDist(row))
    return dataMinDistance


# In[299]:


def calculateDistance(centroid, dataPoint):
    list1 = centroid[1]
    list2 = dataPoint[1][:-1]
    array1 = np.array(list1)
    array2 = np.array(list2)
    dist = np.linalg.norm(array1-array2)
    return (dataPoint[0], (centroid[0], dist))


# In[300]:


def minDist(row):
    index = row[0]
    lst = row[1]
    minDist = float('inf')
    minPoint = None
    for elem in lst:
        centroidIndex = elem[0]
        distance = elem[1]
        if (distance < minDist):
            minDist = distance
            minPoint = (centroidIndex, distance)
    return (index, minPoint)


# In[301]:


def reCalculating(iCentroid, clusterItems):
    allLists = []
    for element in clusterItems:
        # element = ([5.4, 3.7, 1.5, 0.2, 'Iris-setosa'], 3.7349698793966195)
        allLists.append(element[0][:-1])
    averageArray = list(np.average(allLists, axis=0))
    newCenteroid = (iCentroid, averageArray)
    return newCenteroid


# In[326]:


def computeCentroids(dataMinDistance):
    databyCluster = dataMinDistance.join(data).map(lambda x: (x[1][0][0], (x[1][1], x[1][0][1])))
    databyCluster = databyCluster.groupByKey().map(lambda data: (data[0], list(data[1])))
    return databyCluster.map(lambda x: reCalculating(x[0], x[1]))


# In[360]:


def hasConverged(centroids, newCentroids):
    centroidsData = centroids.join(newCentroids)
    centroidsDataBool = centroidsData.map(lambda cluster: cluster[1][0] == cluster[1][1])
    return all(item == True for item in centroidsDataBool.collect())


# In[303]:


sc = SparkContext("local", "generator")


# In[367]:


data = loadData('data/iris_clustering.dat')


# In[368]:


data.collect()


# In[369]:


centroids = initCentroids(data, 3)


# In[370]:


centroids.collect()


# In[371]:


iterations = 0
startTime = datetime.datetime.now()

while True:
    iterations = iterations + 1
    dataMinDistance = assginToCluster(data, centroids)
    newCenteroids = computeCentroids(dataMinDistance)
    
    if hasConverged(centroids, newCenteroids):
        break;
    centroids = sc.parallelize(newCenteroids.collect())

endTime = datetime.datetime.now()

print("Elapsed time: " + str(endTime - startTime))
print("Number of iterations: " + str(iterations))


# In[372]:


centroids.collect()


# In[373]:


sc.stop()


# In[ ]:




