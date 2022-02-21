from typing import Dict

from numpy import ndarray
from sklearn.metrics import accuracy_score, precision_score, recall_score


class MetricsScorer:
    @staticmethod
    def _compute_accuracy_score(y_true: ndarray, y_pred: ndarray) -> float:
        return accuracy_score(y_true, y_pred)

    @staticmethod
    def _compute_precision_score(y_true: ndarray, y_pred: ndarray) -> float:
        return precision_score(y_true, y_pred, average="macro")

    @staticmethod
    def _compute_recall_score(y_true: ndarray, y_pred: ndarray) -> float:
        return recall_score(y_true, y_pred, average="macro")

    def compute_metrics(self, y_true: ndarray, y_pred: ndarray) -> Dict:
        return {
            "accuracy_score": self._compute_accuracy_score(y_true, y_pred),
            "precision_score": self._compute_precision_score(y_true, y_pred),
            "recall_score": self._compute_recall_score(y_true, y_pred),
        }
