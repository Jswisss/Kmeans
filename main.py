from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style

X = np.array([
    [2, 4],
    [1.7, 2.8],
    [7, 8],
    [8.6, 8],
    [3.4, 1.5],
    [9, 11]
])

K = 2


def euclidean_distance(a, b, Aaxis=1):
    # Euclidean distance (l2 norm)
    # 2-d scenario:
    return np.linalg.norm(a - b, axis=Aaxis)

def closestCentroid(x, centroids, new_clusters):
    for i in range(len(X)):
        distance = euclidean_distance(centroids, X[i])
        assignment = np.argmin(distance)
        new_clusters[i] = assignment
    return new_clusters

def updateCentroid(x, clusters, K, new_centroids):
    for i in range(K):
        xArrayPos = [X[j] for j in range(len(X)) if (clusters[j] == i)]
        new_centroids[i] = np.mean(xArrayPos, axis=0)
    return new_centroids

def Kmeans(x,K):
    centroids_y = 12 * np.random.rand(K)
    centroids_x = 12 * np.random.rand(K)
    centroids = np.array(list(zip(centroids_x, centroids_y)))
    zeroed_Out_Centroids = np.zeros(centroids.shape)
    clusters = np.zeros((len(X)))
    UpdatedCentroids = euclidean_distance(centroids, zeroed_Out_Centroids, None)
    while UpdatedCentroids != 0:
        clusters = closestCentroid(x, centroids, clusters)
        New_Centroids = deepcopy(centroids)
        centroids = updateCentroid(x, clusters, K, centroids)
        UpdatedCentroids = euclidean_distance(centroids, New_Centroids, None)
    plt.scatter(X[:, 0], X[:, 1], cmap='rainbow', s=50)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=250)
    print('Initialized centroids: {}'.format(centroids))
    plt.show()

Kmeans(X,K)