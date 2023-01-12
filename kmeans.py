import numpy as np


def kmeans(X, k, t):
    """
    :param X: numpy array of size (m, d) containing the test samples
    :param k: the number of clusters
    :param t: the number of iterations to run
    :return: a column vector of length m, where C(i) ∈ {1, . . . , k} is the identity of the cluster in which x_i has been assigned.
    """
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

#amit

def simple_test():
    # load sample data (this is just an example code, don't forget the other part)
    data = np.load('mnist_all.npz')
    X = np.concatenate((data['train0'], data['train1']))
    m, d = X.shape

    # run K-means
    c = kmeans(X, k=10, t=10)

    assert isinstance(c, np.ndarray), "The output of the function softsvm should be a numpy array"
    assert c.shape[0] == m and c.shape[1] == 1, f"The shape of the output should be ({m}, 1)"

if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    simple_test()

    # here you may add any code that uses the above functions to solve question 2
