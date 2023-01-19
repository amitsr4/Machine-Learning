import numpy as np
import scipy.io as sio
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge


def task2e(best_lambdas):
    x, y = zip(*best_lambdas)
    plt.plot(x, y)
    plt.xlabel('Training sample size')
    plt.ylabel('Best lambda value')
    plt.title('Best lambda value as a function of the training sample size')
    plt.show()


def task2():
    # Load data from file
    data = sio.loadmat('regdata.mat')
    X_train = data['X'].T
    X_test = data['Xtest'].T
    Y_train = data['Y']
    Y_test = data['Ytest']
    train_errors = []
    test_errors = []
    best_lambdas = []

    for n in range(10, 101):
        # Take a subset of the training data
        X_train_subset = X_train[:n, :]
        Y_train_subset = Y_train[:n]

        # Initialize the best lambda value and error
        best_lambda = None
        best_error = float('inf')

        # Loop over lambda values
        for lambd in range(31):
            # Create and fit the Ridge model
            model = Ridge(alpha=lambd)
            model.fit(X_train_subset, Y_train_subset)

            # Compute the train and test errors
            train_error = mean_squared_error(Y_train_subset, model.predict(X_train_subset))
            test_error = mean_squared_error(Y_test, model.predict(X_test))

            # Update the best lambda value and error if necessary
            if test_error < best_error:
                best_error = test_error
                best_lambda = lambd

        best_lambdas.append((n, best_lambda))

        # Append the results to the lists
        train_errors.append(train_error)
        test_errors.append(best_error)

    task2e(best_lambdas)



task2()