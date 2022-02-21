from pydantic import BaseModel


class IrisPrediction(BaseModel):
    sepal_length_cm: float
    sepal_width_cm: float
    petal_length_cm: float
    petal_width_cm: float
    prediction: int
