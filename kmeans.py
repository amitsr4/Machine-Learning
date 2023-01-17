import numpy as np
import pandas as pd


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
    return clusters.reshape(-1, 1)


def task1ce(k):
    # Load the MNIST data filee
    data = np.load('mnist_all.npz')
    X = np.concatenate((data['train0'], data['train1'], data['train2'], data['train3'], data['train4'], data['train5'],
                        data['train6'], data['train7'], data['train8'], data['train9']))

    # Flatten the images
    X = X.reshape(X.shape[0], -1)

    # Generate a random sample of size 1000
    random_sample = np.random.choice(X.shape[0], 1000, replace=False)
    sample_data = X[random_sample]

    # Create an array to store the true labels for each point in the sample data
    true_labels = []
    for i in range(10):
        true_labels += [i] * len(data['train' + str(i)])
    true_labels = np.array(true_labels)[random_sample]

    # Run the k-means algorithm
    clusters = kmeans(sample_data, k=k, t=10)

    # Create a list to store the majority label for each cluster
    majority_labels = []

    # Create a list to store the size of each cluster
    cluster_sizes = []

    # Create a list to store the percentage of each cluster
    cluster_percentage = []

    # Iterate through the clusters
    for i in range(10):
        # Extract the points in the current cluster
        cluster_points = sample_data[np.where(clusters == i)]

        # Extract the true labels for the points in the current cluster
        cluster_labels = true_labels[np.where(clusters == i)[0]]

        # Count the number of instances of each true label within the cluster
        label_counts = np.array([np.sum(cluster_labels == j) for j in range(10)])

        # Determine the majority label for the cluster
        majority_label = np.argmax(label_counts)
        majority_labels.append(majority_label)

        # Determine the size of the cluster
        cluster_size = len(cluster_points)
        cluster_sizes.append(cluster_size)

        # Determine the percentage of the points in the cluster that have the majority label
        percentage = label_counts[majority_label] / cluster_size
        cluster_percentage.append(percentage)

    # Create a table showing the results
    table = [["Cluster", "Size", "Majority Label", "Percentage"]]
    for i in range(10):
        table.append([i + 1, cluster_sizes[i], majority_labels[i], cluster_percentage[i]])




    df = pd.DataFrame(table[1:], columns=table[0])
    file_name = 'C:\\Users\\97252\Desktop\\Computer Sience\\G\\Introduction to Machine Learning\\Assignment 3\\k-means_results_for_k='+'{}.xlsx'.format(k)
    df.to_excel(file_name, index=False)
    print(f'The table has been exported to {file_name}')

    # print(table)

    incorrect_classifications = 0

    # Iterate through the clusters
    for i in range(10):
        # Extract the majority label for the current cluster
        majority_label = majority_labels[i]

        # Extract the true labels for the points in the current cluster
        cluster_labels = true_labels[np.where(clusters == i)[0]]

        # Count the number of points in the cluster that have been assigned the wrong label
        incorrect_classifications += np.sum(cluster_labels != majority_label)

    # Calculate the classification error
    classification_error = incorrect_classifications / len(sample_data)

    # Print the classification error
    print("Classification error: {:.2f}%".format(classification_error * 100))


def simple_test():
    # load sample data (this is just an example code, don't forget the other part)
    data = np.load('mnist_all.npz')
    X = np.concatenate((data['train0'], data['train1']))
    m, d = X.shape

    # run K-means
    c = kmeans(X, k=10, t=10)
    print(c);
    assert isinstance(c, np.ndarray), "The output of the function softsvm should be a numpy array"
    assert c.shape[0] == m and c.shape[1] == 1, f"The shape of the output should be ({m}, 1)"


if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    # simple_test()
    task1ce(10) # for task1 c
    task1ce(6)  # for task1 e

    # here you may add any code that uses the above functions to solve question 2
