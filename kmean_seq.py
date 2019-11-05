import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


means = [[2, 2], [4, 8], [8, 3]]
cov = [[1, 0], [0, 1]]

N = 500
k = 3

X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)

X0 = np.array(X0)
X1 = np.array(X1)
X2 = np.array(X2)

# Data
X = np.concatenate((X0, X1, X2))
# Label
label = np.array([0]*N + [1]*N + [2]*N)

def display(X, label):
    K = np.amax(label) + 1
    X_plot = []
    color = ['ro','b^', 'g*', 'cs']
    for k in range(K):
        X_plot.append(X[label == k, :])
        plt.plot(X_plot[k][:, 0], X_plot[k][:, 1], color[k], markersize = 4, alpha = 0.8)

    plt.axis('equal')
    plt.plot()
    plt.show()

def init_centers(X, K):
    # Radomly pick k rows of X as a initial centers
    return X[np.random.choice(X.shape[0], K, replace=False)]

def assign_lable(X, centers):
    # We can parallel in here 
    D = cdist(X, centers)
    return np.argmin(D, axis=1).T

def update_centers(X, labels, K):
    centers = np.zeros((K, X.shape[1]))
    # We can parallel in here 
    for k in range(K):
        Xk = X[labels == k, :]
        centers[k,:] = np.mean(Xk, axis=0)
    
    return centers

def has_converged(centers, new_centers):
    return set([tuple(a) for a in centers]) == set([tuple(a) for a in new_centers]) 

def kmeans(X, K):
    centers = [init_centers(X, K)]
    labels = []
    it = 0
    while True:
        labels.append(assign_lable(X, centers[-1]))
        new_centers = update_centers(X, labels[-1], K)
        if has_converged(centers[-1], new_centers):
            break
        else:
            centers.append(new_centers)
        it += 1
    return (centers, labels, it)

(centers, labels, it) = kmeans(X, k)
