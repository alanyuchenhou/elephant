import json
import os
import unittest

import numpy
import pandas
import skflow
from hamcrest import assert_that
from hamcrest.core.core.is_ import is_
from hamcrest.library.number.ordering_comparison import greater_than
from sklearn import metrics

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
        embedding_sizes = specs['embedding_sizes']
        hidden_units_formation = specs['hidden_units_formation']
        expected_score = specs['expected_score']
        train = pandas.read_table(specs['train_file'])
        test = pandas.read_table(specs['test_file'])

        train_target = train[target_category]
        test_target = test[target_category]
        categorical_processor = skflow.preprocessing.CategoricalProcessor()
        train_data = numpy.array(list(categorical_processor.fit_transform(train[data_categories])))
        test_data = numpy.array(list(categorical_processor.transform(test[data_categories])))
        vocabulary_sizes = [len(categorical_processor.vocabularies_[i]) for i in range(len(data_categories))]
        edge_estimator_factory = factories.EstimatorFactory(data_categories, vocabulary_sizes, embedding_sizes,
                                                            hidden_units_formation, train_target.nunique())
        estimator = edge_estimator_factory.build_estimator()
        estimator.fit(train_data, train_target, log_dir)
        score = metrics.accuracy_score(test_target, estimator.predict(test_data))
        print("score = ", score)
        assert_that(score, is_(greater_than(expected_score)))

        # estimator.save(log_dir)
        # estimator2 = skflow.TensorFlowEstimator.restore(log_dir)
        # score2 = metrics.accuracy_score(test_target, estimator2.predict(test_data))
        # assert_that(score2, is_(equal_to(score)))
