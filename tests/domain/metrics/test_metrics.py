from unittest.mock import patch

from paa.domain.metrics.metrics import MetricsScorer


@patch("paa.domain.metrics.metrics.accuracy_score")
def test_metric_scorer_compute_accuracy_score_should_call_accuracy_score_with_correct_parameters(
        mock_accuracy_score
):
    # Given
    y_true = "a"
    y_pred = "b"
    metrics_scorer = MetricsScorer()

    # When
    metrics_scorer._compute_accuracy_score(y_true, y_pred)

    # Then
    mock_accuracy_score.assert_called_with(y_true, y_pred)


@patch("paa.domain.metrics.metrics.precision_score")
def test_metric_scorer_compute_precision_score_should_call_precision_score_with_correct_parameters(
        mock_precision_score
):
    # Given
    y_true = "a"
    y_pred = "b"
    metrics_scorer = MetricsScorer()

    # When
    metrics_scorer._compute_precision_score(y_true, y_pred)

    # Then
    mock_precision_score.assert_called_with(y_true, y_pred, average="macro")


@patch("paa.domain.metrics.metrics.recall_score")
def test_metric_scorer_compute_recall_score_should_call_precision_score_with_correct_parameters(
        mock_recall_score
):
    # Given
    y_true = "a"
    y_pred = "b"
    metrics_scorer = MetricsScorer()

    # When
    metrics_scorer._compute_recall_score(y_true, y_pred)

    # Then
    mock_recall_score.assert_called_with(y_true, y_pred, average="macro")


@patch("paa.domain.metrics.metrics.accuracy_score", return_value=1)
@patch("paa.domain.metrics.metrics.precision_score", return_value=2)
@patch("paa.domain.metrics.metrics.recall_score", return_value=3)
def test_metric_scorer_compute_metrics_should_call_other_compute_method_with_correct_parameters(
        mock_recall_score,
        mock_precision_score,
        mock_accuracy_score
):
    # Given
    y_true = "a"
    y_pred = "b"
    expected = {
        "accuracy_score": 1,
        "precision_score": 2,
        "recall_score": 3
    }
    metrics_scorer = MetricsScorer()

    # When
    output = metrics_scorer.compute_metrics(y_true, y_pred)

    # Then
    assert output == expected
    mock_recall_score.assert_called_with(y_true, y_pred, average="macro")
    mock_precision_score.assert_called_with(y_true, y_pred, average="macro")
    mock_accuracy_score.assert_called_with(y_true, y_pred)
