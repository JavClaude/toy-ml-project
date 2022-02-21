from unittest.mock import patch

from paa.domain.data.dataset import IrisDataset


@patch("paa.domain.data.dataset.load_iris")
def test_iris_dataset_get_iris_method_should_call_load_iris_with_correct_parameters(
        mock_load_iris
):
    # Given
    iris_dataset = IrisDataset()

    # When
    _ = iris_dataset._get_iris()

    # Then
    mock_load_iris.assert_called_with(return_X_y=True)


@patch("paa.domain.data.dataset.train_test_split")
def test_iris_dataset_split_data_should_call_train_test_split_with_correct_parameters(
        mock_train_test_split
):
    # Given
    value_used_for_test = "a"
    iris_dataset = IrisDataset()

    # When
    _ = iris_dataset._split_data({value_used_for_test: 2})

    # Then
    mock_train_test_split.assert_called_with(value_used_for_test)


@patch("paa.domain.data.dataset.IrisDataset._get_iris", return_value="a")
@patch("paa.domain.data.dataset.IrisDataset._split_data")
def test_iris_dataset_get_data_should_orchestrate_other_methods(
        mock_split_data,
        mock_get_iris
):
    # Given
    iris_dataset = IrisDataset()

    # When
    iris_dataset.get_data()

    # Then
    mock_get_iris.assert_called()
    mock_split_data.assert_called_with("a")
