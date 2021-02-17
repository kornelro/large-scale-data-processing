import concurrent.futures
from typing import List

from lr import base


class LinearRegressionThreads(base.LinearRegression):
    def fit(self, X: List[float], y: List[float]) -> base.LinearRegression:
        """Fits model using multithreading."""
        self._coef = [None, None]

        self._X_mean = sum(X) / len(X)
        self._y_mean = sum(y) / len(y)

        data = list(zip(X, y))
        self._b_nominator_elements = []
        self._b_denominator_elements = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            executor.map(
                self._get_b_element,
                data
            )

        b_nominator = sum(self._b_nominator_elements)
        b_denominator = sum(self._b_denominator_elements)

        self._coef[1] = b_nominator / b_denominator
        self._coef[0] = self._y_mean - self._coef[1] * self._X_mean

    def _get_b_element(self, d: List[float]) -> None:
        self._b_nominator_elements.append(
            (d[0] - self._X_mean) * (d[1] - self._y_mean)
        )
        self._b_denominator_elements.append((d[0] - self._X_mean)**2)
