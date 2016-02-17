from sklearn import datasets, metrics
from skflow import TensorFlowDNNClassifier
iris = datasets.load_iris()
classifier = TensorFlowDNNClassifier(hidden_units=[10, 20, 10], n_classes=3)
classifier.fit(iris.data, iris.target)
score = metrics.accuracy_score(iris.target, classifier.predict(iris.data))
print("Accuracy: %f" % score)
