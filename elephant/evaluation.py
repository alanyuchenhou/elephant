import os
import shutil

import numpy
import pandas
import tensorflow
from sklearn import metrics

ATTRIBUTES = ['source_node', 'destination_node', 'link_weight', ]
ATTRIBUTE_TYPES = [str, str, float, ]
FEATURE_ATTRIBUTES = ATTRIBUTES[:2]
TARGET_ATTRIBUTE = ATTRIBUTES[2]


def read_file(data_file):
    return pandas.read_csv(
        data_file,
        names=ATTRIBUTES,
        dtype={attribute_name: attribute_type for attribute_name, attribute_type in zip(ATTRIBUTES, ATTRIBUTE_TYPES)},
    )


def input_fn(links, num_epochs, shuffle):
    targets = links[TARGET_ATTRIBUTE]
    links = links.dropna(how='any', axis=0, )
    return tensorflow.estimator.inputs.pandas_input_fn(links, targets, num_epochs=num_epochs, shuffle=shuffle, )


def evaluate(data_set_name, training_file, testing_file, dimension, num_epochs, trial):
    model_dir = os.path.join('../log', data_set_name, str(num_epochs), str(dimension), str(trial))
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    training_set = read_file(training_file)
    testing_set = read_file(testing_file)
    categorical_columns = [
        tensorflow.feature_column.categorical_column_with_vocabulary_list(
            attribute, training_set[attribute].apply(str).unique(),
        ) for attribute in FEATURE_ATTRIBUTES
    ]
    feature_columns = [
        tensorflow.feature_column.embedding_column(column, dimension) for column in categorical_columns
    ]
    model = tensorflow.estimator.DNNRegressor([dimension, dimension], feature_columns, model_dir, )
    model.train(input_fn(training_set, num_epochs, True), )
    predictions = list(model.predict(input_fn(testing_set, 1, False)))
    actual_targets = numpy.concatenate([prediction['predictions'] for prediction in predictions])
    return metrics.mean_squared_error(actual_targets, testing_set[TARGET_ATTRIBUTE])


def main():
    for data_set_name in ['airport', 'collaboration', 'forum', ]:
        # for data_set_name in ['congress']:
        errors = pandas.DataFrame(columns=['num_epochs', 'dimension', 'error', ])
        training_file = os.path.join('../data', data_set_name + '_training.csv')
        testing_file = os.path.join('../data', data_set_name + '_testing.csv')
        for num_epochs in range(3, 4, 1):
            # for num_epochs in range(12, 13, 1):
            for dimension in range(10, 100, 10):
                error = numpy.mean([
                    evaluate(
                        data_set_name, training_file, testing_file, dimension, num_epochs, trial
                    ) for trial in range(10)
                ])
                errors.loc[len(errors)] = [num_epochs, dimension, error]
        print(errors)
        errors.to_csv('../log/' + data_set_name + '/errors.csv', index=False)


if __name__ == '__main__':
    main()
