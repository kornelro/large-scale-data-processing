from typing import List

from lr import base


class LinearRegressionSequential(base.LinearRegression):
    def fit(self, X: List[float], y: List[float]) -> base.LinearRegression:
        """Fits model using sequential approach"""
        self._coef = [None, None]

        X_mean = sum(X) / len(X)
        y_mean = sum(y) / len(y)

        b_nominator_elements = []
        b_denominator_elements = []

        for i in range(len(X)):
            b_nominator_elements.append((X[i] - X_mean) * (y[i] - y_mean))
            b_denominator_elements.append((X[i] - X_mean)**2)

        b_nominator = sum(b_nominator_elements)
        b_denominator = sum(b_denominator_elements)

        self._coef[1] = b_nominator / b_denominator
        self._coef[0] = y_mean - self._coef[1] * X_mean
