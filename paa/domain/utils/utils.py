import numpy as np

from paa.infra.application.data_model.data_model import Iris


def convert_iris_data_structure_to_numpy_array(iris: Iris) -> np.ndarray:
    return np.array(list(iris.dict().values())).reshape(1, -1)
