import numpy as np
from ordreg.jacobian import jacobian

def test_finite_differences():
    N = 1000
    J = 4
    P = 5
    X = np.random.normal(size=(N, P))
    y = np.zeros(shape=(N, J))
    for i in range(N):
        j = np.random.randint(J)
        y[i, j] = 1
    params = np.random.normal(size=P + J - 1)
    C = np.random.uniform()
    actual = jacobian(X=X, y=y, C=C, params=params)
    fte_diff = _finite_differences(X=X, y=y, C=C, params=params)
    assert np.allclose(actual, fte_diff)

def objective(X, y, C, params):
    """
    Return the penalized negative log likelihood
    for params under the proportional odds model.
    First elements of params are eta terms, remaining
    are beta coefficients.
    """
    eta = params[:(y.shape[1]-1)]
    beta = params[(y.shape[1]-1):]
    alpha = np.append(eta[0], np.exp(eta[1:])).cumsum()
    s = X.dot(beta)
    dfunc = 1 / (np.exp(-(np.repeat(s[:, np.newaxis],
                                    len(alpha),
                                    axis=1) + alpha)) + 1)
    dfunc = np.append(np.zeros(dfunc.shape[0])[:, np.newaxis],
                      np.append(dfunc,
                                np.ones(dfunc.shape[0])[:, np.newaxis],
                                axis=1),
                      axis=1)
    mass = np.diff(dfunc, axis=1)
    v = (- np.sum(np.log(np.sum(np.multiply(mass, y), axis=1))) +
         C * np.sum(beta**2))
    return v

def _finite_differences(X, y, C, params, eps=1e-5):
    fdiff = np.zeros(shape=(len(params), ))
    for i in range(len(params)):
        new_params = params[:]
        delta = np.zeros(len(params))
        delta[i] = eps
        forward = objective(X=X, y=y, C=C, params=params + delta)
        backward = objective(X=X, y=y, C=C, params=params - delta)
        fdiff[i] = (forward - backward) / (2 * eps)
    return fdiff
