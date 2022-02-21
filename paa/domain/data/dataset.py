from typing import Dict, Tuple

from numpy import ndarray
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


class IrisDataset:
    def __init__(self, test_size: float = 0.33, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state

    @staticmethod
    def _get_iris() -> Dict:
        return load_iris(return_X_y=True)

    @staticmethod
    def _split_data(iris_dictionary: Dict) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
        return train_test_split(*iris_dictionary)

    def get_data(self) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
        return self._split_data(self._get_iris())
