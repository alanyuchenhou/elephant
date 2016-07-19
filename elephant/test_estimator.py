from unittest import TestCase

from pandas import DataFrame
from sklearn import cross_validation

from elephant.estimator import Estimator


class TestEstimator(TestCase):
    def test_estimate(self):
        config = {
            "n_classes": 0,
            "n_attributes": 2,
            "n_hidden_layers": 2,
            "layer_size": 2,
            "learning_rate": 0.1,
            "dropout": 0.1,
            "batch_size": 1
        }
        data_set = DataFrame({'id1': [6, 2, 8, 2, 1, ], 'id2': [1, 8, 3, 6, 0, ], 'rating': [6, 5, 7, 8, 4, ]})
        print(data_set)
        x = data_set.ix[:, :2].values
        estimator = Estimator(config, x)
        y = data_set.ix[:, 2].values
        y_train, y_test = cross_validation.train_test_split(y, test_size=0.2)
        y_predicted = estimator.estimate(0.1, 33, 0.2, y_train)
        print(y_test)
        print(y_predicted)
