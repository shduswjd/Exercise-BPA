import matplotlib.pyplot as plt
import numpy as np

# %matplotlib inline

np.random.seed(42)

train = np.genfromtxt("train.txt")
train_X, train_y = train[:, 0], train[:, 1]
test = np.genfromtxt("test.txt")
test_X, test_y = test[:, 0], test[:, 1]

# you don't have to modify this cell
plt.plot(test_X, test_y, "bo")
plt.plot(train_X, train_y, "rx")
plt.legend(["test", "train"])

def fourier_features(x, n):
    """
    Produces fourier features given a basis function index.

    Parameters
    ----------
    x : array_like
        input array of shape (N,).
    n : int
        basis function index.

    Returns
    -------
    x_tilda : array_like
        resulting fourier features of shape (N,).

    """
    if n == 0:
        return np.ones_like(x) #원본과 동일한 크기로 행렬 만들어짐
    elif n % 2 == 0:
        l = n/2
        return (1/l) * np.sin(2 * np.pi * l * x)
    else:
        l = (n + 1) / 2
        return (1/l) * np.cos(2 * np.pi * l * x)
    
assert 1 == fourier_features(np.array([0]), 1)
assert 0 == fourier_features(np.array([0]), 2)
assert np.allclose(np.array([1, 1]), fourier_features(np.array([0, 0]), 0))

# Plot of 9 resuting fourier functions.
# you don't have to modify this cell
train_tmp = np.arange(0, 2 * np.pi, 0.01)
question_check = []
labels = []
for n in range(9):
    ffeatures = fourier_features(train_tmp, n)
    plt.plot(train_tmp, ffeatures)
    question_check.append(ffeatures[11])
    labels.append(f"{n}")
plt.legend(labels)

def get_feature_matrix(train, max_m):
    """
    Get a feature matrix.

    Parameters
    ----------
    train : array_like
        input array of shape (N,).
    max_m : int
        maximum number of fourier features M.

    Returns
    -------
    features : array_like
        resulting features matrix (N,K).

    """
    # YOUR CODE HERE
    features = []
    for i in range(max_m):
        features.append(fourier_features(train, i)[:, np.newaxis])
    features = np.hstack(features)
    return features


assert 1 == get_feature_matrix(np.array([0]), 1)
assert np.allclose(np.array([[1, 1]]), get_feature_matrix(np.array([0]), 2))

def calculate_weights(train_features, train_labels):
    """
    Weights calculation for regression.

    Parameters
    ----------
    features : array_like
        input array of shape (N,K).
    labels : array_like
        labels of shape (N).

    Returns
    -------
    w : array_like
        resulting weights matrix (N,).

    """
    # hint: you can use np.linalg.pinv for this one
    # YOUR CODE HERE
    w = np.linalg.pinv(train_features) @ (train_labels)
    return w


assert 1 == calculate_weights(np.array([[1]]), np.array([1]))
assert np.allclose(
    [0.5, 0.5], calculate_weights(np.array([[0, 0], [1, 1]]), np.array([0, 1]))
)

def get_rmse(labels, prediction):
    """
    Calculate root mean square metrics.

    Parameters
    ----------
    labels : array_like
        labels of shape (N,).
    prediction : array_like
        prediction array of shape (N,).

    Returns
    -------
    rmse : float
        resulting rmse.

    """
    # YOUR CODE HERE
    # error = np.square(labels - prediction)
    # mse = np.mean(error)
    # return mse ** 0.5
    rmse = np.sqrt(np.mean(np.square(labels - prediction))) # 위에 있는 내 코드로 해도 됨
    return rmse


assert 0 == get_rmse(np.array([1, 1]), np.array([1, 1]))
assert 1 == get_rmse(np.array([0, 0]), np.array([1, 1]))

# you don't have to modify this cell
x = np.linspace(0, 1, 1000)
train_errors = []
test_errors = []

for max_m in range(1, 18, 2):
    # calculate features and weights
    phi = get_feature_matrix(train_X, max_m)
    w = calculate_weights(phi, train_y)

    # get predictions
    train_prediction = get_feature_matrix(train_X, max_m) @ w
    test_prediction = get_feature_matrix(test_X, max_m) @ w
    prediction_on_x = get_feature_matrix(x, max_m) @ w

    # calculate rmse metrics
    train_rmse = get_rmse(train_y, train_prediction)
    test_rmse = get_rmse(test_y, test_prediction)
    train_errors.append(train_rmse)
    test_errors.append(test_rmse)

    # plotting
    plt.title(f"Train RMSE = {train_rmse:.2f}, Test RMSE = {test_rmse:.2f}")

    legend = []
    legend.append("test")
    plt.plot(test_X, test_y, "bo", markersize=0.5)
    legend.append("train")
    plt.plot(train_X, train_y, "rx")
    legend.append(f"Model with M = {max_m}")
    plt.plot(x, prediction_on_x, "g-")

    plt.xlim([0, 1])
    plt.ylim([-2, 2])
    plt.legend(legend)
    plt.show()

    # you don't have to modify this cell
plt.plot(train_errors, "r")
plt.plot(test_errors, "b")
plt.legend(["train error", "test_error"])
plt.ylabel("RMSE")
plt.xlabel("M components")
plt.xticks(range(9), list(range(1, 18, 2)))
plt.show()

# you don't have to modify this cell
print(f"Minimum RMSE = {np.min(test_errors):.2f}")

def calculate_weights_ridge_regression(train_features, train_labels, alpha=0.01):
    """
    Simple weights calculation from features and labels, with ridge regression.

    Parameters
    ----------
    features : array_like
        input array of shape (N,K).
    labels : array_like
        labels of shape (N).

    Returns
    -------
    w : array_like
        resulting weights matrix (N,).

    """
    # YOUR CODE HERE
    pinv = train_features.T @ train_features
    N, D = pinv.shape[0], pinv.shape[1]
    inv = np.linalg.inv(pinv + alpha * np.eye(N, D))
    w = inv @ train_features.T @ train_labels
    return w

assert 0 == calculate_weights_ridge_regression(np.array([[0]]), np.array([0]))
assert 1 == calculate_weights_ridge_regression(np.array([[1]]), np.array([1]), alpha=0)
assert 0.5 == calculate_weights_ridge_regression(
    np.array([[1]]), np.array([1]), alpha=1
)

# you don't have to modify this cell
alpha = 0.01
x = np.linspace(0, 1, 1000)
train_errors_lambda = []
test_errors_lambda = []

for max_m in range(1, 18, 2):
    # calculate features and weights
    phi = get_feature_matrix(train_X, max_m)
    w = calculate_weights_ridge_regression(phi, train_y, alpha)

    # get predictions
    train_prediction = get_feature_matrix(train_X, max_m) @ w
    test_prediction = get_feature_matrix(test_X, max_m) @ w
    prediction_on_x = get_feature_matrix(x, max_m) @ w

    # calculate rmse metrics
    train_rmse = get_rmse(train_y, train_prediction)
    test_rmse = get_rmse(test_y, test_prediction)
    train_errors_lambda.append(train_rmse)
    test_errors_lambda.append(test_rmse)

    # plotting
    plt.title(f"Train RMSE = {train_rmse:.2f}, Test RMSE = {test_rmse:.2f}")

    legend = []
    legend.append("test")
    plt.plot(test_X, test_y, "bo", markersize=0.5)
    legend.append("train")
    plt.plot(train_X, train_y, "rx")
    legend.append(f"Model with M = {max_m}")
    plt.plot(x, prediction_on_x, "g-")

    plt.xlim([0, 1])
    plt.ylim([-2, 2])
    plt.legend(legend)
    plt.show()

    # you don't have to modify this cell
print(f"Minimum RMSE for ridge regression = {np.min(test_errors_lambda):.2f}") # 0.17

# you don't have to modify this cell
plt.plot(train_errors, "r")
plt.plot(test_errors, "b")
plt.plot(train_errors_lambda, "--r")
plt.plot(test_errors_lambda, "--b")
plt.legend(
    [
        "train error",
        "test error",
        "train error with regularization",
        "test error with regularization",
    ]
)
plt.ylabel("RMSE")
plt.xlabel("M components")
plt.xticks(range(9), list(range(1, 18, 2)))
plt.ylim([0, 1])
plt.show()

def exponential_kernel(x, y, length_scale=0.4):
    """
    Calculation of exponential kernel based on inputs x and y.

    Parameters
    ----------
    x : array_like
        input array of shape (N).
    y : array_like
        labels of shape (N).
    length_scale : float
        parameter of exponential kernel.

    Returns
    -------
    kernel : array_like
        resulting kernel matrix (N,N).

    """
    # YOUR CODE HERE
    x, y = x[:, None], y[:, None]
    D = (
        ((-2 * x @ y.T) / (length_scale**2))
        + np.sum(y**2 / (length_scale**2), axis=1)
        + np.sum(x**2 / (length_scale**2), axis=1)[:, None]
    )
    kernel = np.exp(-0.5 * D)
    return kernel

def get_matrix_a(kernel, train_labels, alpha=0.001):
    """
    Calculation of representation matrix.

    Parameters
    ----------
    kernel : array_like
        input array of shape (N,K).
    labels : array_like
        labels of shape (N).
    alpha : float
        constant to prevent matrix inverse from failing.

    Returns
    -------
    a : array_like
        resulting solution to dual representation matrix (N,).

    """
    # YOUR CODE HERE
    return np.linalg.inv(kernel + alpha * np.eye(kernel.shape[0])) @ train_labels
    
def estimate_at_location(test, train, a, kernel_function, kernel_parameters):
    """
    Simple weights calculation from features and labels, with ridge regression.

    Parameters
    ----------
    test : array_like
        input test data of shape (M,);
    train : array_like
        input train data of shape (N,);
    a : array_like
        dual representation matrix (N,);
    kernel_function : Callable
        desired kernel funciton. Default: polynomial_kernel;
    kernel_parameters : float
        kernel paramters for `kernel_function`.

    Returns
    -------
    prediction : array_like
        resulting estimation (M,).

    """
    # YOUR CODE HERE
    kernel = kernel_function(test, train, kernel_parameters)
    prediction = kernel @ a
    return prediction

# you don't have to modify this cell
x = np.linspace(0, 1, 1000)
train_errors = []
test_errors = []

for kernel_parameters in np.linspace(0.001, 1, 10):
    # compute kernel
    kernel_matrix = exponential_kernel(train_X, train_X, length_scale=kernel_parameters)
    # compute solution for dual representation
    a = get_matrix_a(kernel_matrix, train_y, 0.001)

    # compute estimate at sampled locations
    train_prediction = estimate_at_location(
        train_X,
        train_X,
        a,
        kernel_function=exponential_kernel,
        kernel_parameters=kernel_parameters,
    )
    test_prediction = estimate_at_location(
        test_X,
        train_X,
        a,
        kernel_function=exponential_kernel,
        kernel_parameters=kernel_parameters,
    )
    prediction = estimate_at_location(
        x,
        train_X,
        a,
        kernel_function=exponential_kernel,
        kernel_parameters=kernel_parameters,
    )

    # calculate rmse metrics
    train_rmse = get_rmse(train_y, train_prediction.ravel())
    test_rmse = get_rmse(test_y, test_prediction.ravel())
    train_errors.append(train_rmse)
    test_errors.append(test_rmse)

    # plotting
    plt.title(f"Train RMSE = {train_rmse:.2f}, Test RMSE = {test_rmse:.2f}")

    legend = []
    legend.append("test")
    plt.plot(test_X, test_y, "bo", markersize=0.5)
    legend.append("train")
    plt.plot(train_X, train_y, "rx")
    legend.append(f"Model with kernel parameter = {kernel_parameters}")
    plt.plot(x, prediction, "g-")

    plt.xlim([0, 1])
    plt.ylim([-2, 2])
    plt.legend(legend)
    plt.show()

# you don't have to modify this cell
print(f"Minimum RMSE for ridge regression = {np.min(test_errors):.2f}")

def polynomial_kernel(x, y, d=5):
    """
    Calculation of polynomial kernel based on inputs x and y.

    Parameters
    ----------
    x : array_like
        input array of shape (N).
    y : array_like
        second input of shape (N).
    d : float
        parameter of polynomial kernel.

    Returns
    -------
    kernel : array_like
        resulting kernel matrix (N,N).

    """
    # YOUR CODE HERE
    x, y = x[:, np.newaxis], y[:, np.newaxis]
    kernel = (x @ y.T + 1) ** d
    return kernel


x = np.linspace(0, 1, 1000)
train_errors = []
test_errors = []

for kernel_parameters in np.linspace(2, 20, 6):
    # compute kernel
    kernel_matrix = polynomial_kernel(train_X, train_X, d=kernel_parameters)
    # compute solution for dual representation
    a = get_matrix_a(kernel_matrix, train_y, 0.001)

    # compute estimate at sampled locations
    train_prediction = estimate_at_location(
        train_X,
        train_X,
        a,
        kernel_function=polynomial_kernel,
        kernel_parameters=kernel_parameters,
    )
    test_prediction = estimate_at_location(
        test_X,
        train_X,
        a,
        kernel_function=polynomial_kernel,
        kernel_parameters=kernel_parameters,
    )
    prediction = estimate_at_location(
        x,
        train_X,
        a,
        kernel_function=polynomial_kernel,
        kernel_parameters=kernel_parameters,
    )

    # calculate rmse metrics
    train_rmse = get_rmse(train_y, train_prediction.ravel())
    test_rmse = get_rmse(test_y, test_prediction.ravel())
    train_errors.append(train_rmse)
    test_errors.append(test_rmse)

    # plotting
    plt.title(f"Train RMSE = {train_rmse:.2f}, Test RMSE = {test_rmse:.2f}")

    legend = []
    legend.append("test")
    plt.plot(test_X, test_y, "bo", markersize=0.5)
    legend.append("train")
    plt.plot(train_X, train_y, "rx")
    legend.append(f"Model with k = {kernel_parameters}")
    plt.plot(x, prediction, "g-")

    plt.xlim([0, 1])
    plt.ylim([-2, 2])
    plt.legend(legend)
    plt.show()

# you don't have to modify this cell
print(f"Minimum RMSE for ridge regression = {np.min(test_errors):.2f}")