import pickle

import numpy as np

from paa.domain.data.data_model import Iris
from paa.domain.modelling.classifier import IrisClassifier


def load_trained_iris_classifier(path_to_model: str) -> IrisClassifier:
    with open(path_to_model, "rb") as bytes:
        model = pickle.load(bytes)
    return model


def convert_iris_data_structure_to_numpy_array(iris: Iris) -> np.ndarray:
    return np.array(list(iris.dict().values())).reshape(1, -1)
