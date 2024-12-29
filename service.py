import numpy as np
import bentoml
from bentoml.io import NumpyNdarray

iris_classifier_runner = bentoml.sklearn.get("iris_classifier:jzlk4yofootwnff3").to_runner()

svc = bentoml.Service("iris_classifier", runners=[iris_classifier_runner])

@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def classify(input_series: np.ndarray) -> np.ndarray:
    result = iris_classifier_runner.predict.run(input_series)
    return result