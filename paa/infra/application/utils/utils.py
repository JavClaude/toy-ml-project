import pickle

from paa.domain.modeling.classifier import IrisClassifier
from paa.domain.utils import READ_BYTES


def load_trained_iris_classifier(path_to_model: str) -> IrisClassifier:
    with open(path_to_model, READ_BYTES) as file:
        model = pickle.load(file)
    return model
