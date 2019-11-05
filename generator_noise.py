#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import numpy

from pyspark import SparkContext
from pyspark.mllib.random import RandomRDDs
from matplotlib import pyplot as plt
# constants
MIN_MEAN_VALUE = 0
MAX_MEAN_VALUE = 100
STEPS = 0.1


# In[2]:


# methods

def noise_values(values):
    result = ""
    cluster = random.randint(0, count_cluster - 1)
    for v in values:
        if not result:
            result = str(v)
        else:
            result = result + "," + str(v)
    return (result + "," + str(cluster))


# In[3]:



def point_values(means_value, normal_value, std, cluster, dimension):
    values = ""
    for d in range(dimension):
        value = means_value[d] + normal_value[d] * std
        if not values:
            values = str(value)
        else:
            values = values + "," + str(value)
    return (values + "," + str(cluster))


# In[4]:


def write_into_csv(file_name, rdd): 
    with open(file_name,'wb') as file:
        for row in rdd.collect():
            file.write(row.encode())
            file.write('\n'.encode())


# In[5]:


# inputs
file_name = "out" + '.csv'  # file name to be generated
points = 100 # number of points to be generated
count_cluster = 3 # number of clusters
dimension = 2 # dimension of the data
std = 1 # standard deviation
noise_points = points * 2 # number of noise points to be generated / double the number of points
file_name_noise ='out-noise.csv' # file name for noise points to be generated

sc = SparkContext("local", "generator_noise") # spark context


# In[6]:


# array of the clusters : clusters = [0, 1, 2]
clusters = sc.parallelize(range(0, count_cluster))


# In[7]:


# random means of each cluster : means_cluster = [ (0, [0.6, 80.9]), (1, [57.8, 20.2]), (2, [15.6, 49.9]) ]
lst = list(numpy.arange(MIN_MEAN_VALUE, MAX_MEAN_VALUE, STEPS))
means_cluster = clusters.map(lambda cluster : (cluster, random.sample(lst, dimension)))


# In[8]:


# creating random vector using normalVectorRDD 
random_values_vector = RandomRDDs.normalVectorRDD(sc, numRows = points, numCols = dimension, numPartitions = count_cluster, seed = 1)


# In[9]:


# assiging a random cluster for each point
cluster_normal_values_vector = random_values_vector.map(lambda point : (random.randint(0, count_cluster - 1), point.tolist()))


# In[10]:


# generate a value depending of the mean of the cluster, standard deviation and the normal value 
points_value_vector = cluster_normal_values_vector.join(means_cluster).map(lambda x: point_values(x[1][0], x[1][1], std, x[0], dimension))

points_value_vector.collect()


# In[11]:


# generate random points that represent noise points
lst = list(numpy.arange(MIN_MEAN_VALUE, MAX_MEAN_VALUE, STEPS))
noise_points_vector = sc.parallelize(range(0, noise_points)).map(lambda x : random.sample(lst, dimension)).map(lambda v: noise_values(v))
        
# noise_points_vector = noise_points_vector.map(lambda row : str(row).replace("[", "").replace("]",""))
noise_points_vector.collect()


# In[12]:


data = points_value_vector.union(noise_points_vector)

data.collect()


# In[13]:


# writing points value in a 1 csv file
write_into_csv(file_name, points_value_vector);

# saving noise points generated into a file
write_into_csv(file_name_noise, noise_points_vector);


# In[14]:


X = points_value_vector.collect()
type(X)


# In[15]:


colors = ['ro', 'b^', 'g*']
for x in X:
    lst = x[1:].split(',')
    plt.plot(float(lst[0]), float(lst[1]), colors[int(lst[2])], markersize=4, alpha=0.8)


# In[16]:


sc.stop()

