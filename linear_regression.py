import numpy as np


def univariate_loss(x: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
    """
    :param x: 1D array that represents the feature vector
    :param y: 1D array that represents the target vector
    :param theta: 1D array that represents the parameter vector theta = (b, w)
    :return: a scalar that represents the loss \mathcal{L}_U(theta)
    """
    # TODO: Implement the univariate loss \mathcal{L}_U(theta) (as specified in Equation 1)
    
    b, w = theta
    
    predictions = b + w * x
    
    mse_loss = np.mean((predictions - y) ** 2)

    return mse_loss


def fit_univariate_lin_model(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    :param x: 1D array that contains the feature of each subject
    :param y: 1D array that contains the target of each subject
    :return: the parameter vector theta^* that minimizes the loss \mathcal{L}_U(theta)
    """

    N = x.size
    assert N > 1, "There must be at least 2 points given!"
    # TODO: Implement the expressions you have derived in the pen & paper exercise (Task 1.1.1)

    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_x_sq = np.sum(x ** 2)
    sum_xy = np.sum(x * y)

    numerator = sum_xy - (1/N) * sum_x * sum_y
    denominator = sum_x_sq - (1/N) * (sum_x ** 2)
    w = numerator / denominator

    b = (1/N) * sum_y - w * (1/N) * sum_x
    
    return np.array([b, w])


def calculate_pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """
    :param x: 1D array that contains the feature of each subject
    :param y: 1D array that contains the target of each subject
    :return: a scalar that represents the Pearson correlation coefficient between x and y
    """
    # TODO: Implement Pearson correlation coefficient, as shown in Equation 3 (Task 1.1.2).

    x_mean = np.mean(x)
    y_mean = np.mean(y)

    numerator = np.sum((x - x_mean) * (y - y_mean))

    denominator = np.sqrt(np.sum((x - x_mean) ** 2)) * np.sqrt(np.sum((y - y_mean) ** 2))

    pearson_r = numerator / denominator

    return pearson_r


def compute_design_matrix(data: np.ndarray) -> np.ndarray:
    """
    :param data: 2D array of shape (N, D) that represents the data matrix
    :return: 2D array that represents the design matrix. Think about the shape of the output.
    """

    # TODO: Implement the design matrix for multiple linear regression (Task 1.2.2)

    N = data.shape[0]
    
    ones_column = np.ones((N, 1))
        
    design_matrix = np.hstack((ones_column, data))

    return design_matrix


def multiple_loss(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
    """
    :param X: 2D array that represents the design matrix
    :param y: 1D array that represents the target vector
    :param theta: 1D array that represents the parameter vector
    :return: a scalar that represents the loss \mathcal{L}_M(theta)
    """
    # TODO: Implement the multiple regression loss \mathcal{L}_M(theta) (as specified in Equation 5)

    predictions = X @ theta
    mse_loss = np.mean((predictions - y) ** 2)
    return float(mse_loss)


def fit_multiple_lin_model(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    :param X: 2D array that represents the design matrix
    :param y: 1D array that represents the target vector
    :return: the parameter vector theta^* that minimizes the loss \mathcal{L}_M(theta)
    """
    from numpy.linalg import pinv

    # TODO: Implement the expressions you have derived in the pen & paper exercise (Task 1.2.1). 
    # Note: Use the pinv function.

    theta = pinv(X) @ y
    return theta


def compute_polynomial_design_matrix(x: np.ndarray, K: int) -> np.ndarray:
    """
    :param x: 1D array that represents the feature vector
    :param K: the degree of the polynomial
    :return: 2D array that represents the design matrix. Think about the shape of the output.
    """

    # TODO: Implement the polynomial design matrix (Task 1.3.2)

    polynomial_design_matrix = np.column_stack([x**k for k in range(K + 1)])
    return polynomial_design_matrix