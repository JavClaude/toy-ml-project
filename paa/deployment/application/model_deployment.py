from typing import Any, Dict

from fastapi import FastAPI

from paa.deployment.data_model.data_model import IrisPrediction
from paa.domain.data.data_model import Iris
from paa.domain.utils.utils import (
    convert_iris_data_structure_to_numpy_array,
    load_trained_iris_classifier,
)


def create_app(path_to_model: str) -> FastAPI:
    app = FastAPI()
    iris_classifier = load_trained_iris_classifier(path_to_model)

    @app.get("/v1/get_model_parameters/")
    def get_model_parameters() -> Dict:
        return {**iris_classifier._model.get_params()}

    @app.post("/v1/predict/", response_model=IrisPrediction)
    def get_model_prediction(iris: Iris) -> Dict[str, Any]:
        prediction = iris_classifier.predict(
            convert_iris_data_structure_to_numpy_array(iris)
        )[0]
        iris_prediction = IrisPrediction(**iris.dict(), prediction=prediction)
        return iris_prediction

    return app
