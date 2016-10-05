from unittest import TestCase

from pandas import DataFrame

from elephant.estimator import Estimator


class TestEstimator(TestCase):
    def test_estimate(self):
        config = {
            "n_attributes": 2,
            "n_hidden_layers": 2,
            "layer_size": 20,
            "learning_rate": 0.1,
            "batch_size": 8,
        }
        data_set = DataFrame({'id1': [2, 2, 1, 1, 1, ], 'id2': [1, 1, 2, 2, 2, ], 'rating': [2, 2, 3, 3, 3, ]})
        print(data_set)
        estimator = Estimator(config, data_set.ix[:, :2].values)
        print('testing_error =', estimator.estimate(0.1, config['batch_size'], data_set.ix[:, 2].values, 8))
