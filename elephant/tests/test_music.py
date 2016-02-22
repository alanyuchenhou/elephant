import unittest

import numpy
import pandas
import skflow
from hamcrest import assert_that
from hamcrest.core.core.is_ import is_
from hamcrest.library.number.ordering_comparison import less_than
from sklearn import metrics

import factories


class TestEdgeEstimator(unittest.TestCase):
    def test_build_estimator(self):
        log_dir = '../.log/music'
        train = pandas.read_table('../resources/test.tsv')
        test = pandas.read_table('../resources/test.tsv')
        target_category = "rating"
        data_categories = ['user_id', 'item_id']
        embedding_sizes = [16, 8]
        hidden_units_formation = [8, 4]
        error_max = 600

        train_target = train[target_category]
        test_target = test[target_category]
        categorical_processor = skflow.preprocessing.CategoricalProcessor()
        train_data = numpy.array(list(categorical_processor.fit_transform(train[data_categories])))
        test_data = numpy.array(list(categorical_processor.transform(test[data_categories])))
        vocabulary_sizes = [len(categorical_processor.vocabularies_[i]) for i in range(len(data_categories))]

        edge_estimator_factory = factories.EdgeEstimatorFactory(data_categories, vocabulary_sizes, embedding_sizes,
                                                                hidden_units_formation)
        estimator = edge_estimator_factory.build_edge_estimator()
        estimator.fit(train_data, train_target, log_dir)
        error = metrics.mean_squared_error(test_target, estimator.predict(test_data))
        print("error = ", error)
        assert_that(error, is_(less_than(error_max)))
