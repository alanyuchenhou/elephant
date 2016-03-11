import json
import os
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
        batch_size = 32
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
        # monitor = monitors.ValidationMonitor(x_validate, y_validate, n_classes=0, print_steps=steps/100,
        #                                      early_stopping_rounds=steps/5)
        edge_estimator_factory = factories.EstimatorFactory(data_categories, vocabulary_sizes, embedding_size,
                                                            hidden_units_formation, 0)
        estimator = edge_estimator_factory.build_estimator(batch_size, len(x_train) // batch_size)
        with open(log_path, mode='w') as log:
            print('training_error\tvalidation_error', file=log)
            for epoch in range(8):
                estimator.fit(x_train, y_train)
                errors = [metrics.mean_squared_error(y_train, estimator.predict(x_train)),
                          metrics.mean_squared_error(y_validate, estimator.predict(x_validate))]
                print('\t'.join(map(str, errors)), file=log)
        print('test_error =', metrics.mean_squared_error(y_test, estimator.predict(x_test)))

        # assert_that(metrics.mean_squared_error(y_test, estimator.predict(x_test)), is_(less_than(expected_error)))
        # estimator.save(log_path)
        # estimator2 = skflow.TensorFlowEstimator.restore(log_path)
