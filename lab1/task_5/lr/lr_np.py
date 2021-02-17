from typing import List
import numpy as np

from lr import base


class LinearRegressionNumpy(base.LinearRegression):
    def fit(self, X: List[float], y: List[float]) -> base.LinearRegression:
        """Fits model using numpy."""
        self._coef = [None, None]

        X = np.array(X)
        X_mean = np.mean(X)
        y = np.array(y)
        y_mean = np.mean(y)

        b_nominator = np.sum(((X - X_mean) * (y - y_mean)))
        b_denominator = np.sum((X - X_mean)**2)

        self._coef[1] = b_nominator / b_denominator
        self._coef[0] = y_mean - self._coef[1] * X_mean
