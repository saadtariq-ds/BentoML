import bentoml

iris_classifier_runner = bentoml.sklearn.get("iris_classifier:jzlk4yofootwnff3").to_runner()
iris_classifier_runner.init_local()

print(iris_classifier_runner.predict.run([[5.9, 3., 5.1, 1.8]]))