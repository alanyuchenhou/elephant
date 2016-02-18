from sklearn import cross_validation, datasets, metrics
from skflow import TensorFlowDNNClassifier, TensorFlowEstimator
from os import path


model_dir = '../models/'
model1 = path.join(model_dir, 'model1')

iris = datasets.load_iris()
train_data, test_data, train_target, test_target = cross_validation.train_test_split(
    iris.data, iris.target, test_size=.1, random_state=4)

n_classes = 3
hidden_layer_sizes = [10, 20, 10]
classifier = TensorFlowDNNClassifier(hidden_layer_sizes, n_classes)
classifier.fit(train_data, train_target, model1)
score = metrics.accuracy_score(test_target, classifier.predict(test_data))
print("original accuracy = ", score)
classifier.save(model1)

classifier = TensorFlowEstimator.restore(model1)
score = metrics.accuracy_score(test_target, classifier.predict(test_data))
print("restored accuracy = ", score)
