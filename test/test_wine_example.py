"""
This test case matches an example from the R ordinal package
"""
import pytest
import numpy as np
from ordreg.ordinal import OrdinalRegression
from sklearn.preprocessing import OneHotEncoder


@pytest.fixture
def wine():
    rating = np.array([
       2, 3, 3, 4, 4, 4, 5, 5, 1, 2, 1, 3, 2, 3, 5, 4, 2, 3, 3, 2, 5, 5,
       4, 4, 3, 2, 3, 2, 3, 2, 5, 3, 2, 3, 4, 3, 3, 3, 3, 3, 3, 2, 3, 2,
       2, 4, 5, 4, 1, 1, 2, 2, 2, 3, 2, 3, 2, 2, 2, 3, 3, 3, 3, 4, 1, 2,
       3, 2, 3, 2, 4, 4]).reshape(-1, 1)
    temp = np.array([
       0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1,
       1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0,
       1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0,
       0, 0, 1, 1, 1, 1])
    contact = np.array([
       0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0,
       1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1,
       0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0,
       1, 1, 0, 0, 1, 1])
    X = np.vstack([temp, contact, temp * contact]).T
    y_encoder = OneHotEncoder(sparse=False, categories=[range(1, 6)])
    y = y_encoder.fit_transform(rating)
    return X, y


def test_wine(wine):
    X, y = wine
    model = OrdinalRegression(C=0.)
    model.fit(X, y)
    R_intercept = np.array([-1.4112620, 1.1435537, 3.3770825, 4.9419823])
    R_coef = np.array([2.3211843, 1.3474604, 0.3595489])
    assert np.allclose(model.intercept_, R_intercept, atol=1e-4)
    # NB: the R model is slightly different in that
    # Pr(y <= j) = \alpha_j - \beta x
    assert np.allclose(model.coef_, -1 * R_coef, atol=1e-4)
