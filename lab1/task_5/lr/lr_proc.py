import multiprocessing
from typing import List

from lr import base


class LinearRegressionProcess(base.LinearRegression):
    def fit(self, X: List[float], y: List[float]) -> base.LinearRegression:
        """Fits model using multiprocessing."""
        self._coef = [None, None]

        self._X_mean = sum(X) / len(X)
        self._y_mean = sum(y) / len(y)

        data = list(zip(X, y))
        self._b_nominator_elements = []
        self._b_denominator_elements = []

        with multiprocessing.Pool(processes=2) as pool:
            result = pool.map(
                self._get_b_element,
                data
            )

        result = list(zip(*result))

        b_nominator = sum(result[0])
        b_denominator = sum(result[1])

        self._coef[1] = b_nominator / b_denominator
        self._coef[0] = self._y_mean - self._coef[1] * self._X_mean

    def _get_b_element(self, d: List[float]) -> None:
        nom_el = (d[0] - self._X_mean) * (d[1] - self._y_mean)
        denom_el = (d[0] - self._X_mean)**2

        return (nom_el, denom_el)
