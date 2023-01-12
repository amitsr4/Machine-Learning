import numpy as np


def singlelinkage(X, k):
    """
    :param X: numpy array of size (m, d) containing the test samples
    :param k: the number of clusters
    :return: a column vector of length m, where C(i) ∈ {1, . . . , k} is the identity of the cluster in which x_i has been assigned.
    """
    # m, d = X.shape
    # distances = np.zeros((m, m))
    # for i in range(m):
    #     for j in range(i, m):
    #         distances[i, j] = np.linalg.norm(X[i] - X[j])
    #         distances[j, i] = distances[i, j]
    # clusters = np.arange(m)
    # for _ in range(m - k):
    #     i, j = np.unravel_index(np.argmin(distances), distances.shape)
    #     clusters[clusters == j] = i
    #     distances[i, :] = np.minimum(distances[i, :], distances[j, :])
    #     distances[:, i] = distances[i, :]
    #     distances[j, :] = np.inf
    #     distances[:, j] = np.inf
    # return clusters.reshape(-1,1)

    # get the number of examples and coordinates
    m, d = X.shape
    # calculate pairwise Euclidean distances
    distances = np.linalg.norm(X[:, np.newaxis] - X, axis=-1)
    # set the diagonal elements to infinity
    np.fill_diagonal(distances, np.inf)
    # initialize each point as its own cluster
    clusters = np.arange(m)
    for _ in range(m - k):
        # find the closest pair of clusters
        i, j = np.unravel_index(np.argmin(distances), distances.shape)
        # merge the closest pair of clusters
        clusters[clusters == j] = clusters[i]
        # update the distance matrix
        distances = np.minimum(distances, np.minimum(distances[j, :], distances[:, j]))
        distances[j, :] = np.inf
        distances[:, j] = np.inf
    # assign a label to each cluster
    _, labels = np.unique(clusters, return_inverse=True)
    return labels.reshape(-1, 1) + 1
#TODO FIX !
    # m, d = X.shape
    # C = np.arange(m).reshape(-1, 1) # Create initial cluster assignments
    # distances = np.sqrt(np.sum((X[:, None] - X)**2, axis=-1)) # calculate pairwise distances
    # while len(np.unique(C)) > k:
    #     C_repeated = np.repeat(C, m).reshape(m, m)
    #     C_tiled = np.tile(C, (m, 1))
    #     C_mask = (C_repeated.astype(int) != C_tiled.astype(int)) & (C_repeated.astype(int) != C_tiled.T.astype(int))
    #     distances_mask = np.ma.array(distances, mask=C_mask)
    #     min_distance = np.min(distances_mask)
    #     i, j = np.where(distances == min_distance)
    #     if i.size and j.size:
    #         i, j = i[0], j[0]
    #         C[C == C[j]] = C[i] # merge the two closest clusters
    #     else:
    #         break
    # return C

def simple_test():

    # load sample data (this is just an example code, don't forget the other part)
    data = np.load('mnist_all.npz')
    train0 = data['train0']
    train1 = data['train1']
    train3 = data['train3']
    train2 = data['train2']
    train5 = data['train5']
    train4 = data['train4']
    train6 = data['train6']
    train7 = data['train7']
    train8 = data['train8']
    train9 = data['train9']
    indices0 = np.random.permutation(train0.shape[0])
    train0= train0[indices0[:30], :]
    indices1 = np.random.permutation(train1.shape[0])
    train1 = train1[indices1[:30], :]
    indices2 = np.random.permutation(train2.shape[0])
    train2 = train2[indices2[:30], :]
    indices3 = np.random.permutation(train3.shape[0])
    train3 = train3[indices3[:30], :]
    indices4 = np.random.permutation(train4.shape[0])
    train4 = train4[indices4[:30], :]
    X = np.concatenate((train0, train1, train2, train3, train4))



    # indices1 = np.random.permutation(train0.shape[0])
    # samples1 = train0[indices1[:30], :]
    # indices2 = np.random.permutation(train1.shape[0])
    # samples2 = train1[indices2[:30], :]
    # sample_train0 = np.random.choice(data['train0'][0], size=(30, 784))
    # sample_train1 = np.random.choice(data['train1'][0], size=(30, 784))

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
