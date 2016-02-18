from sklearn import datasets, metrics
from skflow import TensorFlowDNNClassifier, TensorFlowEstimator
from os import path

model_dir = '../models/'
model1 = path.join(model_dir, 'model1')

iris = datasets.load_iris()
classifier = TensorFlowDNNClassifier(hidden_units=[10, 20, 10], n_classes=3)
classifier.fit(iris.data, iris.target, model1)
score = metrics.accuracy_score(iris.target, classifier.predict(iris.data))
print("original accuracy = ", score)
classifier.save(model1)

classifier = TensorFlowEstimator.restore(model1)
score = metrics.accuracy_score(iris.target, classifier.predict(iris.data))
print("restored accuracy = ", score)
