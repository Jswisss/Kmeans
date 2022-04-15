import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from matplotlib import style
X = np.array([
 [2, 4],
 [1.7, 2.8],
 [7, 8],
 [8.6, 8],
 [3.4, 1.5],
 [9,11]
])
kmeans = KMeans(n_clusters= 2)
kmeans.fit(X)

print(kmeans.cluster_centers_)

print(kmeans.labels_)
plt.scatter(X[:,0], X[:,1], c=kmeans.labels_, cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], color= 'black')
plt.show()