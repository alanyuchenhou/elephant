import pandas
import skflow
import numpy
from sklearn import metrics
import unittest
from hamcrest import assert_that
from hamcrest.core.core.is_ import is_
from hamcrest.core.core.isequal import equal_to
from hamcrest.library.number.ordering_comparison import greater_than
from estimator_factory import EstimatorFactory


class TestEstimatorFactory(unittest.TestCase):
    def test_build_estimator(self):
        model_directory = '../models/model2'
        train = pandas.read_csv('../data/titanic_train.csv')
        test = pandas.read_csv('../data/titanic_test.csv')
        data_categories = ['Embarked', 'Sex']
        embedding_sizes = [3, 2]
        target_category = "Survived"
        hidden_units_formation = [4, 4, 2]
        minimum_expected_score = 0.8

        train_target = train[target_category]
        test_target = test[target_category]
        categorical_processor = skflow.preprocessing.CategoricalProcessor()
        train_data = numpy.array(list(categorical_processor.fit_transform(train[data_categories])))
        test_data = numpy.array(list(categorical_processor.transform(test[data_categories])))
        vocabulary_sizes = [len(categorical_processor.vocabularies_[i]) for i in range(len(data_categories))]

        estimator_factory = EstimatorFactory(data_categories, vocabulary_sizes, embedding_sizes, hidden_units_formation)
        estimator = estimator_factory.build_estimator()
        estimator.fit(train_data, train_target, model_directory)
        score = metrics.accuracy_score(test_target, estimator.predict(test_data))
        assert_that(score, is_(greater_than(minimum_expected_score)))

        estimator.save(model_directory)
        estimator2 = skflow.TensorFlowEstimator.restore(model_directory)
        score2 = metrics.accuracy_score(test_target, estimator2.predict(test_data))
        assert_that(score2, is_(equal_to(score)))
        print("score = ", score)
