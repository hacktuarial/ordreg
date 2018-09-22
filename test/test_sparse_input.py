import pytest
import numpy as np
from scipy import sparse
from ordreg.ordinal import OrdinalRegression


N = 100
P = 4
J = 3


@pytest.fixture
def X_dense():
    return np.random.normal(size=(N, P))


@pytest.fixture
def X_sparse(X_dense):
    return sparse.csr_matrix(X_dense)


@pytest.fixture
def y_dense():
    ones = np.random.choice(list(range(J)), size=N).reshape(-1, 1)
    y = np.zeros((N, J))
    for i, j in enumerate(ones):
        y[i, j] = 1
    return y


@pytest.fixture
def y_sparse(y_dense):
    return sparse.csr_matrix(y_dense)


def test_sparse_input_X(X_dense, X_sparse, y_dense):
    """Make sure results are the same for sparse and dense input"""
    model_d = OrdinalRegression().fit(X_dense, y_dense)
    model_s = OrdinalRegression().fit(X_sparse, y_dense)
    assert np.allclose(model_d.intercept_, model_s.intercept_)
    assert np.allclose(model_d.coef_, model_s.coef_)


def test_sparse_input_y(X_dense, y_dense, y_sparse):
    """Make sure results are the same for sparse and dense input"""
    model_d = OrdinalRegression().fit(X_dense, y_dense)
    model_s = OrdinalRegression().fit(X_dense, y_sparse)
    assert np.allclose(model_d.intercept_, model_s.intercept_)
    assert np.allclose(model_d.coef_, model_s.coef_)
