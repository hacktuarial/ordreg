import numpy as np
from scipy.special import expit


def jacobian(X, y, C, params):
    eta = params[:(y.shape[1]-1)]
    beta = params[(y.shape[1]-1):]
    alpha = np.append(eta[0], np.exp(eta[1:])).cumsum()
    IL = _l2_clogistic_gradient_IL(X, alpha, beta)
    grad_alpha = _l2_clogistic_gradient_intercept(IL=IL, Y=y, alpha=alpha)
    grad_alpha_eta = _l2_clogistic_gradient_alpha_eta(eta)
    grad_eta = np.dot(grad_alpha, grad_alpha_eta)
    grad_beta = l2_clogistic_gradient_slope(X=X, Y=y, IL=IL, beta=beta, C=C)
    return np.concatenate([grad_eta, grad_beta])


def _l2_clogistic_gradient_alpha_eta(eta):
    J = len(eta)
    X = np.zeros(shape=(J, J))
    X[:, 1] = 1
    for j in range(1, J):
        X[j] = eta[j] * np.exp(eta[j])
    return X


def l2_clogistic_gradient_slope(X, Y, IL, beta, C):
    """ Gradient of penalized loglikelihood with respect to the slope parameters
    Args:
        X : array_like. design matrix
        Y : array_like. response matrix
        IL : array_like. See _l2_clogistic_gradient_IL
        beta : array_like. parameters. must have shape == number of columns of X
    Returns:
        array_like : same length as `beta`.
    """
    grad_beta = np.zeros(len(beta))
    J = Y.shape[1]
    XT = X.transpose()  # CSC format
    for j in range(J):  # coefficients
        if j == 0:
            grad_beta = XT.dot(Y[:, j] * (1.0 - IL[:, j]))
        elif j < J - 1:
            grad_beta += XT.dot(Y[:, j] * (1.0 - IL[:, j] - IL[:, j - 1]))
        else:  # j == J-1. this is the highest level of response
            grad_beta -= XT.dot(Y[:, j] * IL[:, j - 1])
    grad_beta -= C * beta
    return grad_beta


def _l2_clogistic_gradient_IL(X, alpha, beta):
    """ Helper function for calculating the cumulative logistic gradient. \
        The inverse logit of alpha[j + X*beta] is \
        ubiquitous in gradient and Hessian calculations \
        so it's more efficient to calculate it once and \
        pass it around as a parameter than to recompute it every time
    Args:
        X : array_like. design matrix
        alpha : array_like. intercepts. must have shape == one less than the number of columns of `Y`
        beta : array_like. parameters. must have shape == number of columns of X
        n : int, optional.\
        You must specify the number of rows if there are no main effects
    Returns:
        array_like. n x J-1 matrix where entry i,j is the inverse logit of (alpha[j] + X[i, :] * beta)
    """
    J = len(alpha) + 1
    n = X.shape[0]
    Xb = X.dot(beta)
    IL = np.zeros((n, J - 1))
    for j in range(J - 1):
        IL[:, j] = expit(alpha[j] + Xb)
    return IL


def _l2_clogistic_gradient_intercept(IL, Y, alpha):
    """ Gradient of penalized loglikelihood with respect to the intercept parameters
    Args:
        IL : array_like. See _l2_clogistic_gradient_IL
        Y : array_like. response matrix
        alpha : array_like. intercepts. must have shape == one less than the number of columns of `Y`
    Returns:
        array_like : length J-1
    """
    exp_int = np.exp(alpha)
    grad_alpha = np.zeros(len(alpha))
    J = len(alpha) + 1
    for j in range(J - 1):  # intercepts
        # there are J levels, and J-1 intercepts
        # indexed from 0 to J-2
        if j == 0:
            grad_alpha[j] = np.dot(Y[:, j], 1 - IL[:, j]) -\
                            np.dot(Y[:, j + 1], exp_int[j] / (exp_int[j + 1] - exp_int[j]) + IL[:, j])
        elif j < J - 2:
            grad_alpha[j] = np.dot(Y[:, j], exp_int[j] / (exp_int[j] - exp_int[j - 1]) - IL[:, j]) - \
                            np.dot(Y[:, j + 1], exp_int[j] / (exp_int[j + 1] - exp_int[j]) + IL[:, j])
        else:  # j == J-2. the last intercept
            grad_alpha[j] = np.dot(Y[:, j], exp_int[j] / (exp_int[j] - exp_int[j - 1]) - IL[:, j]) - \
                            np.dot(Y[:, j + 1], IL[:, j])
    return grad_alpha
