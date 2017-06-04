import os
import shutil
import sys

import pandas
import tensorflow
from sklearn import model_selection
from tensorflow.contrib import learn, layers, metrics, tensorboard

ATTRIBUTES = [
    "age", "work_class", "final_weight", "education", "education_num", "marital_status", "occupation", "relationship",
    "race", "gender", "capital_gain", "capital_loss", "hours_per_week", "native_country", "income_bracket",
]
FEATURE_ATTRIBUTES = ["occupation", "native_country", ]
TARGET_ATTRIBUTE = "label"


def generate_metadata(data_frame, log_dir, ):
    feature_attributes = [{
        'name': attribute, 'metadata_path': os.path.join(log_dir, attribute + '_metadata.tsv')
    } for attribute in FEATURE_ATTRIBUTES
    ]
    projector_configuration = tensorboard.plugins.projector.ProjectorConfig()
    for attribute in feature_attributes:
        embedding = projector_configuration.embeddings.add()
        embedding.tensor_name = 'dnn/input_from_feature_columns/' + attribute['name'] + '_embedding/weights:0'
        embedding.metadata_path = attribute['metadata_path']
        with open(attribute['metadata_path'], 'w+') as metadata_handler:
            for item in data_frame[attribute['name']].unique():
                metadata_handler.write(item + '\n')
    tensorboard.plugins.projector.visualize_embeddings(tensorflow.summary.FileWriter(log_dir), projector_configuration)


def input_fn(data_frame):
    feature_tensor = {
        attribute: tensorflow.SparseTensor(
            indices=[[i, 0] for i in range(data_frame[attribute].size)],
            values=data_frame[attribute].values,
            dense_shape=[data_frame[attribute].size, 1],
        ) for attribute in FEATURE_ATTRIBUTES
    }
    target_tensor = tensorflow.constant(data_frame[TARGET_ATTRIBUTE].values)
    return feature_tensor, target_tensor


def train_and_eval(train_steps, log_dir, training_set, validation_set, testing_set, ):
    sparse_columns = [
        layers.sparse_column_with_keys(attribute, training_set[attribute].unique()) for attribute in FEATURE_ATTRIBUTES
    ]
    embedding_columns = [
        layers.embedding_column(column, dimension=8) for column in sparse_columns
    ]
    m = learn.DNNClassifier(
        hidden_units=[10, 50, ],
        feature_columns=embedding_columns,
        model_dir=log_dir,
        config=learn.RunConfig(save_checkpoints_secs=1, ),
    )
    validation_metrics = {
        "accuracy": learn.MetricSpec(metric_fn=metrics.streaming_accuracy, prediction_key="classes"),
        "precision": learn.MetricSpec(metric_fn=metrics.streaming_precision, prediction_key="classes"),
        "recall": learn.MetricSpec(metric_fn=metrics.streaming_recall, prediction_key="classes"),
    }
    monitors = [
        learn.monitors.ValidationMonitor(
            input_fn=lambda: input_fn(validation_set),
            every_n_steps=1000,
            metrics=validation_metrics,
            early_stopping_rounds=1,
        ),
    ]
    m.fit(
        input_fn=lambda: input_fn(training_set),
        steps=train_steps,
        monitors=monitors,
    )
    results = m.evaluate(input_fn=lambda: input_fn(testing_set), steps=1)
    for key in sorted(results):
        print("%s: %s" % (key, results[key]))


def main(_):
    tensorflow.logging.set_verbosity(tensorflow.logging.INFO)
    log_dir = os.path.join(os.path.dirname(__file__), '../log/model_r')
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    data_set = pandas.read_csv(
        os.path.join(os.path.dirname(__file__), "../resources/adult.training.csv"),
        names=ATTRIBUTES,
        skipinitialspace=True,
        engine="python",
    )
    data_set = data_set.dropna(how='any', axis=0)
    data_set[TARGET_ATTRIBUTE] = (data_set["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)
    data_set, testing_set = model_selection.train_test_split(data_set, test_size=0.2, )
    training_set, validation_set = model_selection.train_test_split(data_set, test_size=0.2, )
    train_and_eval(200, log_dir, training_set, validation_set, testing_set, )
    generate_metadata(data_set, log_dir)


if __name__ == "__main__":
    tensorflow.app.run(main=main, argv=[sys.argv[0]])
