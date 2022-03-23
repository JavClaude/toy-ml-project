import pickle

from numpy import ndarray
from sklearn.ensemble import RandomForestClassifier

from paa.domain.modeling import WRITE_BYTES, TRAINING_MESSAGE


class IrisClassifier:
    def __init__(self, **kwargs) -> None:
        self._model = RandomForestClassifier(**kwargs)

    def predict(self, X: ndarray) -> ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: ndarray) -> ndarray:
        return self._model.predict_proba(X)

    def train_classifier_on_cv(self) -> None:
        raise NotImplementedError()

    def train_classifier(self, X: ndarray, y: ndarray) -> None:
        print(TRAINING_MESSAGE.format(self._model.get_params()))
        self._model.fit(X, y)

    def save_classifier(self, filepath: str = None) -> None:
        with open(filepath, WRITE_BYTES) as output_file:
            pickle.dump(self, output_file)
