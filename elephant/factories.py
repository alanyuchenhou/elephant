import skflow
import tensorflow


class EstimatorFactory(object):
    def __init__(self, data_categories, vocabulary_sizes, layer_size, n_hidden_layers, keep_probability, n_classes=0):
        self.keep_probability = keep_probability
        self.n_classes = n_classes
        self.hidden_units_formation = [layer_size] * n_hidden_layers
        self.embedding_size = layer_size
        self.vocabulary_sizes = vocabulary_sizes
        self.data_categories = data_categories

    def _build_model(self, data, target):
        ids = tensorflow.split(1, len(self.data_categories), data)
        node_vectors = [skflow.ops.categorical_variable(
            ids[i], self.vocabulary_sizes[i], self.embedding_size, self.data_categories[i]
        ) for i in range(len(self.data_categories))]
        activation_in = tensorflow.squeeze(tensorflow.concat(2, node_vectors), [1])
        activation_out = skflow.ops.dnn(activation_in, self.hidden_units_formation, keep_prob=self.keep_probability)
        if self.n_classes > 1:
            return skflow.models.logistic_regression(activation_out, target)
        else:
            return skflow.models.linear_regression(activation_out, target)

    def build_estimator(self, batch_size, n_steps, optimizer, learning_rate):
        return skflow.TensorFlowEstimator(self._build_model, self.n_classes, batch_size=batch_size, steps=n_steps,
                                          optimizer=optimizer, learning_rate=learning_rate, continue_training=True)
