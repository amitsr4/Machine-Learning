import numpy as np
import scipy.io as sio
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge


class RidgeRegression:
    def __init__(self, lambda_=0.1):
        self.lambda_ = lambda_
        self.w = None

    def fit(self, X, Y):
        n_samples, n_features = X.shape
        # closed form solution
        identity = np.eye(n_features)
        identity[0, 0] = 0
        X = np.hstack((np.ones((X.shape[0], 1)), X))

        self.w = np.linalg.inv(X.T @ X + self.lambda_ * identity) @ X.T @ Y

    def predict(self, X):
        n_samples, n_features = X.shape
        X = np.hstack((np.ones((n_samples, 1)), X))
        return X @ self.w



def task2e(best_lambda, training_sizes):
    plt.plot(training_sizes, best_lambda, '-o')
    plt.xlabel("Training set size")
    plt.ylabel("Best lambda")
    plt.title("Best lambda vs Training set size")
    plt.grid()
    plt.xticks(training_sizes)
    plt.show()

def task2():
    # Load data from file
    data = sio.loadmat('regdata.mat')
    print(data.keys())
    X_train = data['X']
    Y_train = data['Y']
    Xtest = data['Xtest']
    Ytest = data['Ytest']
    print(X_train.shape)
    print(Y_train.shape)
    print(Xtest.shape)
    print(Ytest.shape)
    best_lambda = []
    best_mse = []
    training_sizes = range(10, 101)

    # Loop over different training set sizes
    for n in training_sizes:
        # Select a subset of the training data
        X, y = X_train[:n], Y_train[:n]
        min_mse = float("inf")
        best_l = 0
        # Loop over different values of lambda
        for l in range(31):
            # Fit the model
            ridge = Ridge(alpha=l)
            ridge.fit(X, y)
            # Predict on the test set
            y_pred = ridge.predict(Xtest)
            # Calculate the mean squared error
            mse = mean_squared_error(Ytest, y_pred)
            # Update the best lambda and mse if a new minimum is found
            if mse < min_mse:
                min_mse = mse
                best_l = l
        best_lambda.append(best_l)
        best_mse.append(min_mse)

        task2e(best_lambda, training_sizes)

# def task2():  # without skilearn
#     # Load data from file
#     data = sio.loadmat('regdata.mat')
#     print(data.keys())
#     X_train = data['X']
#     Y_train = data['Y']
#     Xtest = data['Xtest']
#     Ytest = data['Ytest']
#     # Initialize lists to store results
#     best_lambda = []
#     best_mse = []
#     training_sizes = range(10, 101)
#
#     # Loop over different training set sizes
#     for n in training_sizes:
#         # Select a subset of the training data
#         X, y = X_train[:n], Y_train[:n]
#         min_mse = float("inf")
#         best_l = 0
#         # Loop over different values of lambda
#         for l in range(31):
#             # Fit the model
#             ridge = RidgeRegression(lambda_=l)
#             ridge.fit(X_train, Y_train)
#             # Predict on the test set
#             y_pred = ridge.predict(Xtest)
#             # Calculate the mean squared error
#             mse = mean_squared_error(Ytest, y_pred)
#             # Update the best lambda and mse if a new minimum is found
#             if mse < min_mse:
#                 min_mse = mse
#                 best_l = l
#         best_lambda.append(best_l)
#         best_mse.append(min_mse)
#
#
#     task2e(best_lambda, training_sizes)
#
#     print("Best lambda for different training set sizes:", best_lambda)
#     print("Corresponding MSE for different training set sizes:", best_mse)


task2()