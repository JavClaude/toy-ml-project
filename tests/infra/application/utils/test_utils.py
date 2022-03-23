from unittest.mock import patch, mock_open

from paa.infra.application.utils.utils import load_trained_iris_classifier


@patch("paa.infra.application.utils.utils.pickle.load", return_value="b")
def test_load_trained_iris_classifier_should_call_pickle_load_with_correct_parameter(mock_load):
    # Given

    # When
    with patch("builtins.open", new_callable=mock_open()) as mock_reader:
        mock_reader.return_value.__enter__.return_value = "a"

        output = load_trained_iris_classifier("a")

    # Then
    mock_load.assert_called_with("a")
    assert output == "b"