from unittest import TestCase

from pandas import DataFrame

from elephant.estimator import Estimator


class TestEstimator(TestCase):
    def test_estimate(self):
        config = {
            "n_attributes": 2,
            "learning_rate": 0.1,
            "batch_size": 1,
        }
        data_set = DataFrame({'id1': [2, 2, 1, 1, 1, ], 'id2': [1, 1, 2, 2, 2, ], 'rating': [2, 2, 3, 3, 3, ]})
        estimator = Estimator(data_set.ix[:, :2].values, config, 2, 2)
        y = data_set.ix[:, 2].values
        for metric in ['MSE', 'MAE']:
            error = estimator.estimate(y, config['batch_size'], 0.1, metric, 8)
            assert error > 0
            assert error < 1
