from unittest.mock import patch, mock_open

import pytest

from paa.domain.modelling.classifier import IrisClassifier


@patch("paa.domain.modelling.classifier.RandomForestClassifier.predict")
def test_iris_classifier_predict_method_should_call_predict_method_from_underlying_classifier(mock_predict):
    # Given
    iris_classifier = IrisClassifier()

    # When
    iris_classifier.predict("a")

    # Then
    mock_predict.assert_called_with("a")


@patch("paa.domain.modelling.classifier.RandomForestClassifier.predict_proba")
def test_iris_classifier_predict_proba_method_should_call_predict_proba_method_from_underlying_classifier(
        mock_predict_proba
):
    # Given
    iris_classifier = IrisClassifier()

    # When
    iris_classifier.predict_proba("a")

    # Then
    mock_predict_proba.assert_called_with("a")


@patch("paa.domain.modelling.classifier.RandomForestClassifier.fit")
@patch("paa.domain.modelling.classifier.RandomForestClassifier.get_params")
def test_iris_classifier_train_method_should_call_get_params_and_fit_method_from_underlying_classifier(
        mock_get_params, mock_fit
):
    # Given
    iris_classifier = IrisClassifier()

    # When
    iris_classifier.train_classifier("a", "b")

    # Then
    mock_get_params.assert_called()
    mock_fit.assert_called_with("a", "b")


def test_iris_classifier_train_classifier_on_cv_should_raise_not_implemented_error():
    # Given
    iris_classifier = IrisClassifier()

    # When
    with pytest.raises(NotImplementedError) as exc_info:
        iris_classifier.train_classifier_on_cv()

    # Then
    assert exc_info.type == NotImplementedError


@patch("paa.domain.modelling.classifier.pickle.dump", return_value="a")
def test_iris_classifier_save_classifier_method_should_call_dump_with_correct_parameter(mock_dump):
    # Given
    iris_classifier = IrisClassifier()

    # When
    with patch("builtins.open", new_callable=mock_open()) as mock_writer:
        mock_writer.return_value.__enter__.return_value = "b"
        iris_classifier.save_classifier("")

    mock_dump.assert_called_with(
        iris_classifier, "b"
    )
