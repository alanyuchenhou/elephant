import json
import os
import unittest

import numpy
import pandas
import skflow
from hamcrest import assert_that
from hamcrest.core.core.is_ import is_
from hamcrest.library.number.ordering_comparison import less_than
from skflow import monitors
from sklearn import metrics, cross_validation

import factories


class TestEstimatorFactory(unittest.TestCase):
    def test_build_estimator(self):
        with open(os.path.join(os.path.dirname(__file__), 'config.json')) as config_file:
            config = json.load(config_file)
        with open(config['specs_file']) as specs_file:
            specs = json.load(specs_file)
        target_category = specs['target_category']
        data_categories = specs['data_categories']
        log_dir = specs['log_dir']
        embedding_size = specs['embedding_size']
        hidden_units_formation = specs['hidden_units_formation']
        expected_error = specs['expected_error']
        steps = specs['steps']
        train = pandas.read_table(specs['train_file'])
        test = pandas.read_table(specs['test_file'])

        y_train = train[target_category]
        y_test = test[target_category]
        categorical_processor = skflow.preprocessing.CategoricalProcessor()
        x_train = numpy.array(list(categorical_processor.fit_transform(train[data_categories])))
        x_test = numpy.array(list(categorical_processor.transform(test[data_categories])))
        x_train, x_validate, y_train, y_validate = cross_validation.train_test_split(x_train, y_train,
                                                                                     test_size=0.01, random_state=42)
        vocabulary_sizes = [len(categorical_processor.vocabularies_[i]) for i in range(len(data_categories))]

        monitor = monitors.ValidationMonitor(x_validate, y_validate, 0, steps / 10, steps / 10)
        edge_estimator_factory = factories.EstimatorFactory(data_categories, vocabulary_sizes, embedding_size,
                                                            hidden_units_formation, 0, steps)
        estimator = edge_estimator_factory.build_estimator()
        estimator.fit(x_train, y_train, monitor, log_dir)
        error = metrics.mean_squared_error(y_train, estimator.predict(x_train))
        print('train_error =', error)

        estimator.save(log_dir)
        estimator2 = skflow.TensorFlowEstimator.restore(log_dir)
        error = metrics.mean_squared_error(y_test, estimator2.predict(x_test))
        print('test_error =', error)
        assert_that(error, is_(less_than(expected_error)))
