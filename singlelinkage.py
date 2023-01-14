import math

import numpy as np
from scipy.spatial import distance


def singlelinkage(X, k):
    """
    :param X: numpy array of size (m, d) containing the test samples
    :param k: the number of clusters
    :return: a column vector of length m, where C(i) ∈ {1, . . . , k} is the identity of the cluster in which x_i has been assigned.
    """

    # Initialize each example as its own cluster
    clusters = [X[i:i+1] for i in range(X.shape[0])]
    while len(clusters) > k:
        closest_distance = np.inf
        clust_1 = clust_2 = None
        for i, cluster1 in enumerate(clusters[:len(clusters) - 1]):
            for j, cluster2 in enumerate(clusters[i + 1:]):
                # Calculate pairwise distances between all points in the current and prospective clusters
                dist = np.min(np.min(np.linalg.norm(cluster1[:, None] - cluster2, axis=2)))
                if dist < closest_distance:
                    closest_distance = dist
                    clust_1 = i
                    clust_2 = j + i + 1
        # Merge the closest pair of clusters using np.concatenate
        clusters[clust_2] = np.concatenate((clusters[clust_2], clusters[clust_1]))
        # Remove the merged cluster
        clusters.pop(clust_1)
        if len(clusters) == k:
            break
    labels = np.zeros(X.shape[0])
    for i, cluster in enumerate(clusters):
        indices = np.concatenate(cluster)
        for j in range(indices.shape[0]):
            labels[indices[j]] = i + 1
    return labels.reshape(-1, 1)




def simple_test():
    # load sample data (this is just an example code, don't forget the other part)
    data = np.load('mnist_all.npz')
    train0 = data['train0']
    train1 = data['train1']
    train2 = data['train2']
    train3 = data['train3']
    train4 = data['train4']
    train5 = data['train5']
    train6 = data['train6']
    train7 = data['train7']
    train8 = data['train8']
    train9 = data['train9']
    indices0 = np.random.permutation(train0.shape[0])
    train0 = train0[indices0[:30], :]
    indices1 = np.random.permutation(train1.shape[0])
    train1 = train1[indices1[:30], :]
    indices2 = np.random.permutation(train2.shape[0])
    train2 = train2[indices2[:30], :]
    indices3 = np.random.permutation(train3.shape[0])
    train3 = train3[indices3[:30], :]
    indices4 = np.random.permutation(train4.shape[0])
    train4 = train4[indices4[:30], :]
    indices5 = np.random.permutation(train5.shape[0])
    train5 = train5[indices5[:30], :]
    indices6 = np.random.permutation(train6.shape[0])
    train6 = train6[indices6[:30], :]
    indices7 = np.random.permutation(train7.shape[0])
    train7 = train7[indices7[:30], :]
    indices8 = np.random.permutation(train8.shape[0])
    train8 = train8[indices8[:30], :]
    indices9 = np.random.permutation(train9.shape[0])
    train9 = train9[indices9[:30], :]

    X = np.concatenate((train0, train1, train2, train3, train4, train5, train6, train7, train8, train9))

    # X = np.concatenate((data['train0'], data['train1']))
    # X = np.concatenate((samples1, samples2))

    m, d = X.shape

    # run single-linkage

    c = singlelinkage(X, k=10)

    assert isinstance(c, np.ndarray), "The output of the function softsvm should be a numpy array"
    assert c.shape[0] == m and c.shape[1] == 1, f"The shape of the output should be ({m}, 1)"


if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    simple_test()

    # here you may add any code that uses the above functions to solve question 2
