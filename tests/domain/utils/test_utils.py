import numpy as np

from paa.infra.application.data_model.data_model import Iris
from paa.domain.utils.utils import (
    convert_iris_data_structure_to_numpy_array
)


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
