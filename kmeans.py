import numpy as np

def kmeans(X, k, t):
    m, d = X.shape
    # initialize centroids randomly
    centroids = X[np.random.choice(m, k, replace=False), :]
    for i in range(t):
        # assign each point to closest centroid
        distances = np.array([np.linalg.norm(X - centroid, axis=1) for centroid in centroids])
        clusters = np.argmin(distances, axis=0)
        # update centroids
        new_centroids = np.array([X[clusters == j].mean(axis=0) for j in range(k)])
        if np.array_equal(centroids, new_centroids):
            break
        centroids = new_centroids
    return clusters