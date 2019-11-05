# import numpy as np
# from scipy.spatial.distance import cdist

# file = 'data/iris_clustering.dat'
# x = open(file)
# a = x.readlines()
# data = []
# for line in a:
#     line = line[:-1]
#     data.append(line.split(','))

# new_data = []
# for a in data:
#     a = a[:-1]
#     new_data.append([float(x) for x in a])

# def init_centers(X, K):
#     # Radomly pick k rows of X as a initial centers
#     return X[np.random.choice(X.shape[0], K, replace=False)]

# def assign_lable(X, centers):
#     # We can parallel in here 
#     D = cdist(X, centers)
#     return np.argmin(D, axis=1).T

# def update_centers(X, labels, K):
#     centers = np.zeros((K, X.shape[1]))
#     # We can parallel in here 
#     for k in range(K):
#         Xk = X[labels == k, :]
#         centers[k,:] = np.mean(Xk, axis=0)
    
#     return centers

# def has_converged(centers, new_centers):
#     return set([tuple(a) for a in centers]) == set([tuple(a) for a in new_centers]) 

# def kmeans(X, K):
#     centers = [init_centers(X, K)]
#     labels = []
#     it = 0
#     while True:
#         labels.append(assign_lable(X, centers[-1]))
#         new_centers = update_centers(X, labels[-1], K)
#         if has_converged(centers[-1], new_centers):
#             break
#         else:
#             centers.append(new_centers)
#         it += 1
#     return (centers, labels, it)


# data = np.asfarray(new_data)

# (centers, labels, it) = kmeans(data, 3)

# print(centers[-1])
# print(it)

lst = ['a', ['b']]
print('b' in lst)