import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin
# from sklearn.utils import check_X_y


class OrdinalRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, penalty='l2', C=1.0, random_state=None):
        self.penalty = penalty
        self.C = C
        self.random_state = random_state

    def fit(self, X, y):
        def objective(params):
            """
            Return the penalized negative log likelihood
            for params under the proportional odds model.
            First elements of params are eta terms, remaining
            are beta coefficients.
            """
            eta = params[:(y.shape[1]-1)]
            beta = params[(y.shape[1]-1):]
            alpha = np.append(eta[0], np.exp(eta[1:])).cumsum()

            s = np.dot(X, beta)
            dfunc = 1 / (np.exp(-(np.repeat(s[:, np.newaxis],
                                            len(alpha),
                                            axis=1) + alpha)) + 1)
            dfunc = np.append(np.zeros(dfunc.shape[0])[:, np.newaxis],
                              np.append(dfunc,
                                        np.ones(dfunc.shape[0])[:, np.newaxis],
                                        axis=1),
                              axis=1)

            mass = np.diff(dfunc, axis=1)

            if self.penalty == "l2":
                norm = np.sum(beta**2)
            elif self.penalty == "l1":
                norm = np.sum(abs(beta))
            else:
                raise ValueError(("penalty must be 'l2' or 'l1',"
                                  " {penalty} was provided".
                                  format(penalty=self.penalty)))

            return - np.sum(np.log(np.sum(mass * y, axis=1))) + self.C * norm

        np.random.seed(self.random_state)
        result = minimize(fun=objective,
                          x0=np.random.randn(X.shape[1] + y.shape[1] - 1))

        eta_hat = result["x"][:(y.shape[1]-1)]
        self.intercept_ = np.append(eta_hat[0],
                                    np.exp(eta_hat[1:])).cumsum()
        self.coef_ = result["x"][(y.shape[1]-1):]

        return self

    def predict_proba(self, X):
        if not hasattr(self, "coef_"):
            raise AttributeError("Model has not been fit")

        s = np.dot(X, self.coef_)

        dfunc = 1 / (np.exp(-(np.repeat(s[:, np.newaxis],
                                        len(self.intercept_),
                                        axis=1) + self.intercept_)) + 1)
        dfunc = np.append(np.zeros(dfunc.shape[0])[:, np.newaxis],
                          np.append(dfunc,
                                    np.ones(dfunc.shape[0])[:, np.newaxis],
                                    axis=1),
                          axis=1)

        return np.diff(dfunc, axis=1)

    def predict_log_proba(self, X):
        return np.log(self.predict_proba(X))

    def predict(self, X):
        return np.apply_along_axis(arr=self.predict_proba(X),
                                   func1d=np.argmax,
                                   axis=1)


class OrdinalRegressionCV(OrdinalRegression, BaseEstimator,
                          ClassifierMixin):
    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def predict_proba(self, X):
        pass

    def predict_log_proba(self, X):
        pass
