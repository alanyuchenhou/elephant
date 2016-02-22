import unittest

import numpy
import pandas
import skflow
from hamcrest import assert_that
from hamcrest.core.core.is_ import is_
from hamcrest.library.number.ordering_comparison import greater_than
from sklearn import metrics

import factories


class TestEdgeEstimator(unittest.TestCase):
    def test_build_estimator(self):
        log_dir = '../.log/model2'
        train = pandas.read_csv('../resources/titanic_train.csv')
        test = pandas.read_csv('../resources/titanic_test.csv')
        target_category = "Survived"
        data_categories = ['Embarked', 'Sex']
        embedding_sizes = [3, 2]
        hidden_units_formation = [4, 4]
        score_min = 0.8

        train_target = train[target_category]
        test_target = test[target_category]
        categorical_processor = skflow.preprocessing.CategoricalProcessor()
        train_data = numpy.array(list(categorical_processor.fit_transform(train[data_categories])))
        test_data = numpy.array(list(categorical_processor.transform(test[data_categories])))
        vocabulary_sizes = [len(categorical_processor.vocabularies_[i]) for i in range(len(data_categories))]

        edge_estimator_factory = factories.EdgeEstimatorFactory(data_categories, vocabulary_sizes, embedding_sizes,
                                                                hidden_units_formation, train_target.nunique())
        estimator = edge_estimator_factory.build_edge_estimator()
        estimator.fit(train_data, train_target, log_dir)
        score = metrics.accuracy_score(test_target, estimator.predict(test_data))
        print("score = ", score)
        assert_that(score, is_(greater_than(score_min)))

        # estimator.save(log_dir)
        # estimator2 = skflow.TensorFlowEstimator.restore(log_dir)
        # score2 = metrics.accuracy_score(test_target, estimator2.predict(test_data))
        # assert_that(score2, is_(equal_to(score)))
