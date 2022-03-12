from unittest.mock import patch, MagicMock

import numpy as np
from fastapi.testclient import TestClient

from paa.infra.application.data_model.data_model import IrisPrediction
from paa.infra.application.app import create_app


@patch("paa.infra.application.app.load_trained_iris_classifier")
def test_application_get_model_parameters_should_return_correct_parameters(
        mock_load_trained_iris_classifier
):
    # Given
    mock_classifier = MagicMock()
    mock_classifier._model.get_params.return_value = {"n_estimators": 100}
    mock_load_trained_iris_classifier.return_value = mock_classifier

    app = create_app("")
    client = TestClient(app)

    # Then
    response = client.get("/v1/get_model_parameters")

    # When
    assert response.status_code == 200
    assert response.json() == {"n_estimators": 100}


@patch("paa.infra.application.app.convert_iris_data_structure_to_numpy_array")
@patch("paa.infra.application.app.load_trained_iris_classifier")
def test_application_predict_should_return_correct_parameters(
        mock_load_trained_iris_classifier,
        mock_convert_iris_data_structure_to_numpy_array
):
    # Given
    mock_classifier = MagicMock()
    mock_classifier.predict.return_value = np.array([1])

    mock_load_trained_iris_classifier.return_value = mock_classifier
    mock_convert_iris_data_structure_to_numpy_array.return_value = "a"

    app = create_app("")
    client = TestClient(app)
    data = {
            "sepal_length_cm": 0.1,
            "sepal_width_cm": 0.2,
            "petal_length_cm": 0.3,
            "petal_width_cm": 0.4
        }

    # Then
    response = client.post(
        "/v1/predict/",
        json=data
    )

    # When
    assert response.status_code == 200
    assert response.json() == {
        **IrisPrediction(
            **data,
            prediction=1
        ).dict()
    }
    mock_classifier.predict.assert_called_with("a")
