import math

import numpy
import tensorflow
from sklearn import cross_validation
from tensorflow.contrib.learn import ops, models, TensorFlowEstimator, preprocessing, monitors


class Estimator(object):
    def __init__(self, config, x):
        self.n_classes = config['n_classes']
        self.n_ids = config['n_attributes']
        self.dropout = config['dropout']
        self.layer_size = config['layer_size']
        self.hidden_units_formation = [self.layer_size] * config['n_hidden_layers']
        categorical_processor = preprocessing.CategoricalProcessor()
        self.x = numpy.array(list(categorical_processor.fit_transform(x)))
        self.vocabulary_sizes = [len(categorical_processor.vocabularies_[i]) for i in range(self.n_ids)]

    def _build_model(self, data, target):
        ids = tensorflow.split(1, self.n_ids, data)
        node_vectors = [ops.categorical_variable(ids[i], self.vocabulary_sizes[i], self.layer_size // 2, str(i)
                                                 ) for i in range(self.n_ids)]
        activation_in = tensorflow.squeeze(tensorflow.concat(2, node_vectors), [1])
        activation_out = ops.dnn(activation_in, self.hidden_units_formation, dropout=self.dropout)
        if self.n_classes > 1:
            return models.logistic_regression(activation_out, target)
        else:
            return models.linear_regression(activation_out, target)

    def estimate(self, test_size, batch_size, learning_rate, y_train):
        x_train, x_test = cross_validation.train_test_split(self.x, test_size=test_size)
        x_train, x_validate, y_train, y_validate = cross_validation.train_test_split(x_train, y_train, test_size=0.1)
        monitor = monitors.ValidationMonitor(x_validate, y_validate, every_n_steps=(len(x_train) // batch_size),
                                             early_stopping_rounds=3)
        estimator = TensorFlowEstimator(self._build_model, self.n_classes, batch_size, learning_rate)
        estimator.fit(x_train, y_train, math.inf, [monitor])
        return estimator.predict(x_test).round()
