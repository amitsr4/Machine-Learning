import numpy as np

def singlelinkage(X, k):
    m, d = X.shape
    C = np.zeros((m,1))
    clusters = [np.array([i]) for i in range(m)]
    while len(clusters) > k:
        min_dist = float('inf')
        min_i, min_j = -1, -1
        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                dist = float('inf')
                for x in clusters[i]:
                    for y in clusters[j]:
                        dist = min(dist, np.linalg.norm(X[x]-X[y]))
                if dist < min_dist:
                    min_dist = dist
                    min_i, min_j = i, j
        new_cluster = np.concatenate((clusters[min_i], clusters[min_j]))
        clusters.pop(min_j)
        clusters.pop(min_i)
        clusters.append(new_cluster)
    for i in range(k):
        for j in clusters[i]:
            C[j] = i+1
    return C