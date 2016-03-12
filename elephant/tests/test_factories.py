import json
import os
import sys
import unittest

import numpy
import pandas
import skflow
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
        log_path = specs['log_path']
        embedding_size = specs['embedding_size']
        hidden_units_formation = specs['hidden_units_formation']
        batch_size = specs['batch_size']
        data_set = pandas.read_table(specs['data_set'])
        optimizer = specs['optimizer']
        learning_rate = specs['learning_rate']
        keep_probability = specs['keep_probability']

        categorical_processor = skflow.preprocessing.CategoricalProcessor()
        x_train = numpy.array(list(categorical_processor.fit_transform(data_set[data_categories])))
        y_train = data_set[target_category]
        x_train, x_test, y_train, y_test = cross_validation.train_test_split(x_train, y_train, test_size=0.2)
        x_train, x_validate, y_train, y_validate = cross_validation.train_test_split(x_train, y_train, test_size=0.1)
        vocabulary_sizes = [len(categorical_processor.vocabularies_[i]) for i in range(len(data_categories))]
        # monitor = monitors.ValidationMonitor(x_validate, y_validate, n_classes=0, print_steps=steps/100,
        #                                      early_stopping_rounds=steps/5)
        edge_estimator_factory = factories.EstimatorFactory(data_categories, vocabulary_sizes, embedding_size,
                                                            hidden_units_formation, keep_probability)
        estimator = edge_estimator_factory.build_estimator(batch_size, len(x_train) // batch_size, optimizer,
                                                           learning_rate)
        with open(log_path, mode='w') as log:
            print('training_error\tvalidation_error', file=log)
            validation_errors = [sys.float_info.max, sys.float_info.max]
            while validation_errors[-2] >= validation_errors[-1]:
                estimator.fit(x_train, y_train)
                training_error = metrics.mean_absolute_error(y_train, estimator.predict(x_train).round())
                validation_error = metrics.mean_absolute_error(y_validate, estimator.predict(x_validate).round())
                print('\t'.join(map(str, [training_error, validation_error])), file=log)
                validation_errors.append(validation_error)
        print('testing_error =', metrics.mean_absolute_error(y_test, estimator.predict(x_test).round()))

        # assert_that(metrics.mean_squared_error(y_test, estimator.predict(x_test)), is_(less_than(expected_error)))
        # estimator.save(log_path)
        # estimator2 = skflow.TensorFlowEstimator.restore(log_path)
