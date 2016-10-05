import math

import numpy
import tensorflow
from sklearn import cross_validation, metrics
from tensorflow.contrib import learn, layers, framework


class Estimator(object):
    def __init__(self, config, x):
        self.learning_rate = config['learning_rate']
        self.n_ids = config['n_attributes']
        self.layer_size = config['layer_size']
        self.hidden_units_formation = [self.layer_size] * config['n_hidden_layers']
        categorical_processor = learn.preprocessing.CategoricalProcessor()
        self.x = numpy.array(list(categorical_processor.fit_transform(x)))
        self.vocabulary_sizes = [len(categorical_processor.vocabularies_[i]) for i in range(self.n_ids)]

    def _build_model(self, data, target):
        ids = tensorflow.split(1, self.n_ids, data)
        node_vectors = [learn.ops.categorical_variable(ids[i], self.vocabulary_sizes[i], self.layer_size, str(i)
                                                       ) for i in range(self.n_ids)]
        activation_in = tensorflow.squeeze(tensorflow.concat(2, node_vectors), [1])
        activation_out = layers.stack(activation_in, layers.fully_connected, self.hidden_units_formation)
        prediction, loss = learn.models.linear_regression(activation_out, target)
        train_op = layers.optimize_loss(loss, framework.get_global_step(), self.learning_rate, 'SGD')
        return prediction, loss, train_op

    def estimate(self, y, batch_size, test_size, metric, steps=math.inf):
        x, x_test, y, y_test = cross_validation.train_test_split(self.x, y, test_size=test_size)
        x_train, x_validate, y_train, y_validate = cross_validation.train_test_split(x, y, test_size=0.1)
        monitor = learn.monitors.ValidationMonitor(x_validate, y_validate, every_n_steps=(len(x_train) // batch_size),
                                                   early_stopping_rounds=1)
        estimator = learn.Estimator(self._build_model)
        estimator.fit(x_train, y_train, steps=steps, batch_size=batch_size, monitors=[monitor])
        y_predicted = estimator.predict(x_test)
        if metric == 'MAE':
            return metrics.mean_absolute_error(y_test, y_predicted)
        elif metric == 'MSE':
            return metrics.mean_squared_error(y_test, y_predicted)
        else:
            assert False
