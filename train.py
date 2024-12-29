import bentoml
from sklearn.svm import SVC
from sklearn.datasets import load_iris

# Load training dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train the model
model = SVC(gamma='scale')
model.fit(X, y)

# Save the model to the BentoML local model store
saved_model = bentoml.sklearn.save_model("iris_classifier", model)
print(f"Model Saved: {saved_model}")


# Inferencing: jzlk4yofootwnff3