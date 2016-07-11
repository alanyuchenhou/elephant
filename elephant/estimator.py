import math

import numpy
import tensorflow
from sklearn import metrics, cross_validation
from tensorflow.contrib.learn import ops, models, TensorFlowEstimator, preprocessing, monitors


class Estimator(object):
    def __init__(self, config, data_set):
        self.data_categories = config['data_categories']
        self.batch_size = config['batch_size']
        self.optimizer = config['optimizer']
        self.learning_rate = config['learning_rate']
        self.dropout = config['dropout']
        layer_size = config['layer_size']
        n_hidden_layers = config['n_hidden_layers']
        self.hidden_units_formation = [layer_size] * n_hidden_layers
        self.embedding_size = layer_size
        categorical_processor = preprocessing.CategoricalProcessor()
        self.x = numpy.array(list(categorical_processor.fit_transform(data_set[self.data_categories].values)))
        self.y = data_set[config['target_category']].values
        self.n_classes = 0
        self.vocabulary_sizes = [len(categorical_processor.vocabularies_[i]) for i in range(len(self.data_categories))]

    def _build_model(self, data, target):
        ids = tensorflow.split(1, len(self.data_categories), data)
        node_vectors = [ops.categorical_variable(
            ids[i], self.vocabulary_sizes[i], self.embedding_size, self.data_categories[i]
        ) for i in range(len(self.data_categories))]
        activation_in = tensorflow.squeeze(tensorflow.concat(2, node_vectors), [1])
        activation_out = ops.dnn(activation_in, self.hidden_units_formation, dropout=self.dropout)
        if self.n_classes > 1:
            return models.logistic_regression(activation_out, target)
        else:
            return models.linear_regression(activation_out, target)

    def estimate(self):
        x, x_test, y, y_test = cross_validation.train_test_split(self.x, self.y, test_size=0.2)
        x_train, x_validate, y_train, y_validate = cross_validation.train_test_split(x, y, test_size=0.1)
        monitor = monitors.ValidationMonitor(x_validate, y_validate, every_n_steps=(len(x_train) // self.batch_size),
                                             early_stopping_rounds=4)
        estimator = TensorFlowEstimator(self._build_model, self.n_classes, self.batch_size, 200, self.optimizer,
                                        self.learning_rate)
        estimator.fit(x_train, y_train, math.inf, [monitor])
        print('testing_error =', metrics.mean_absolute_error(y_test, estimator.predict(x_test).round()))
