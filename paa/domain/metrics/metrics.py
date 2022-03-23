from typing import Dict

from numpy import ndarray
from sklearn.metrics import accuracy_score, precision_score, recall_score

from paa.domain.metrics import (
    ACCURACY_SCORE,
    AVERAGE_MACRO,
    PRECISION_SCORE,
    RECALL_SCORE,
)


class MetricsScorer:
    @staticmethod
    def _compute_accuracy_score(y_true: ndarray, y_prediction: ndarray) -> float:
        return accuracy_score(y_true, y_prediction)

    @staticmethod
    def _compute_precision_score(y_true: ndarray, y_prediction: ndarray) -> float:
        return precision_score(y_true, y_prediction, average=AVERAGE_MACRO)

    @staticmethod
    def _compute_recall_score(y_true: ndarray, y_prediction: ndarray) -> float:
        return recall_score(y_true, y_prediction, average=AVERAGE_MACRO)

    def compute_metrics(self, y_true: ndarray, y_prediction: ndarray) -> Dict:
        return {
            ACCURACY_SCORE: self._compute_accuracy_score(y_true, y_prediction),
            PRECISION_SCORE: self._compute_precision_score(y_true, y_prediction),
            RECALL_SCORE: self._compute_recall_score(y_true, y_prediction),
        }
