import numpy as np

def singlelinkage(X, k):
    m, d = X.shape
    C = np.zeros((m,1))
    clusters = [np.array([i]) for i in range(m)]
    while len(clusters) > k:
        distances = np.min(np.linalg.norm(X[np.newaxis,clusters[0]] - X[np.newaxis,clusters[1:]], axis=2), axis=1)
        min_i, min_j = np.argmin(distances) , min_i + 1 + np.argmin(distances[min_i])
        new_cluster = np.concatenate((clusters[min_i], clusters[min_j]))
        clusters.pop(min_j)
        clusters.pop(min_i)
        clusters.append(new_cluster)
    for i in range(k):
        for j in clusters[i]:
            C[j] = i+1
    return C