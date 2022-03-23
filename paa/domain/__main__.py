import argparse
import logging

from paa.domain.data.dataset import IrisDataset
from paa.domain.metrics.metrics import MetricsScorer
from paa.domain.modeling.classifier import IrisClassifier

logger = logging.getLogger(__name__)
logger.setLevel(logging.NOTSET)


def train_model() -> None:
    argument_parser = argparse.ArgumentParser(
        description="Little package for training a random forest on iris"
    )
    argument_parser.add_argument(
        "--path_to_save_model", type=str, required=False, default="model.bin"
    )
    argument_parser.add_argument("--test_size", type=float, default=0.33)
    argument_parser.add_argument(
        "--n_estimators", type=int, default=100, required=False
    )
    argument_parser.add_argument("--max_depth", type=int, default=4, required=False)
    argument_parser.add_argument(
        "--random_state_splitter", type=int, default=42, required=False
    )
    argument_parser.add_argument(
        "--random_state_classifier", type=int, default=42, required=False
    )

    arguments = argument_parser.parse_args()

    iris_dataset = IrisDataset(
        test_size=arguments.test_size, random_state=arguments.random_state_splitter
    )
    metrics_scorer = MetricsScorer()
    iris_classifier = IrisClassifier(
        n_estimators=arguments.n_estimators,
        max_depth=arguments.max_depth,
        random_state=arguments.random_state_classifier,
    )

    x_train, x_test, y_train, y_test = iris_dataset.get_data()

    iris_classifier.train_classifier(x_train, y_train)

    print(
        "Metric on training set: {}".format(
            metrics_scorer.compute_metrics(y_train, iris_classifier.predict(x_train))
        )
    )
    print(
        "Metric on test set: {}".format(
            metrics_scorer.compute_metrics(y_test, iris_classifier.predict(x_test))
        )
    )

    iris_classifier.save_classifier(arguments.path_to_save_model)
