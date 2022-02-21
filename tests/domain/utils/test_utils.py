from unittest.mock import patch, mock_open

from paa.domain.data.data_model import Iris
from paa.domain.utils.utils import (
    load_trained_iris_classifier,
    convert_iris_data_structure_to_numpy_array
)

import numpy as np


@patch("paa.domain.utils.utils.pickle.load", return_value="b")
def test_load_trained_iris_classifier_should_call_pickle_load_with_correct_parameter(mock_load):
    # Given

    # When
    with patch("builtins.open", new_callable=mock_open()) as mock_reader:
        mock_reader.return_value.__enter__.return_value = "a"

        output = load_trained_iris_classifier("a")

    # Then
    mock_load.assert_called_with("a")
    assert output == "b"


def test_convert_iris_data_structure_to_numpy_array_should_return_correct_numpy_array():
    # Given
    iris = Iris(
        sepal_length_cm=0.2,
        sepal_width_cm=1.5,
        petal_length_cm=2.4,
        petal_width_cm=0.7
    )
    expected = np.array(
        [
            [0.2, 1.5, 2.4, 0.7]
        ]
    )

    # When
    output = convert_iris_data_structure_to_numpy_array(iris)

    # Then
    np.testing.assert_array_equal(output, expected)
