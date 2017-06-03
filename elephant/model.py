import os
import shutil
import sys

import pandas
import tensorflow
from tensorflow.contrib import learn, layers, metrics, tensorboard

LOG_DIR = os.path.join(os.path.dirname(__file__), '../log/model_r')
if os.path.exists(LOG_DIR):
    shutil.rmtree(LOG_DIR)
TRAINING_SET = os.path.join(os.path.dirname(__file__), "../resources/adult.training.csv")
TESTING_SET = os.path.join(os.path.dirname(__file__), "../resources/adult.testing.csv")
ATTRIBUTES = [
    "age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation", "relationship", "race",
    "gender", "capital_gain", "capital_loss", "hours_per_week", "native_country", "income_bracket",
]
FEATURE_ATTRIBUTES = ["occupation", "native_country", ]
TARGET_ATTRIBUTE = "label"


def generate_metadata(data_frame):
    feature_attributes = [{
        'name': attribute, 'metadata_path': os.path.join(LOG_DIR, attribute + '_metadata.tsv')
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
    tensorboard.plugins.projector.visualize_embeddings(tensorflow.summary.FileWriter(LOG_DIR), projector_configuration)


def input_fn(data_frame):
    feature_tensor = {
        attribute: tensorflow.SparseTensor(
            indices=[[i, 0] for i in range(data_frame[attribute].size)],
            values=data_frame[attribute].values,
            dense_shape=[data_frame[attribute].size, 1]
        )
        for attribute in FEATURE_ATTRIBUTES
    }
    target_tensor = tensorflow.constant(data_frame[TARGET_ATTRIBUTE].values)
    return feature_tensor, target_tensor


def train_and_eval(train_steps, ):
    tensorflow.logging.set_verbosity(tensorflow.logging.INFO)
    df_train = pandas.read_csv(TRAINING_SET, names=ATTRIBUTES, skipinitialspace=True, engine="python", )
    df_test = pandas.read_csv(TESTING_SET, names=ATTRIBUTES, skipinitialspace=True, engine="python", )
    df_train = df_train.dropna(how='any', axis=0)
    df_test = df_test.dropna(how='any', axis=0)
    df_train[TARGET_ATTRIBUTE] = (df_train["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)
    df_test[TARGET_ATTRIBUTE] = (df_test["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)

    sparse_columns = [
        layers.sparse_column_with_hash_bucket(
            attribute, hash_bucket_size=len(set(df_train[attribute]))
        ) for attribute in FEATURE_ATTRIBUTES
    ]
    embedding_columns = [
        layers.embedding_column(column, dimension=8) for column in sparse_columns
    ]
    m = learn.DNNClassifier(
        hidden_units=[10, 50, ],
        feature_columns=embedding_columns,
        model_dir=LOG_DIR,
        config=learn.RunConfig(save_checkpoints_secs=1, ),
    )
    validation_metrics = {
        "accuracy": learn.MetricSpec(metric_fn=metrics.streaming_accuracy, prediction_key="classes"),
        "precision": learn.MetricSpec(metric_fn=metrics.streaming_precision, prediction_key="classes"),
        "recall": learn.MetricSpec(metric_fn=metrics.streaming_recall, prediction_key="classes"),
    }
    monitors = [
        learn.monitors.ValidationMonitor(
            input_fn=lambda: input_fn(df_test), every_n_steps=1000, metrics=validation_metrics, early_stopping_rounds=1,
        ),
    ]
    m.fit(
        input_fn=lambda: input_fn(df_train),
        steps=train_steps,
        monitors=monitors,
    )
    results = m.evaluate(input_fn=lambda: input_fn(df_test), steps=1)
    for key in sorted(results):
        print("%s: %s" % (key, results[key]))

    generate_metadata(df_train)


def main(_):
    train_and_eval(200, )


if __name__ == "__main__":
    tensorflow.app.run(main=main, argv=[sys.argv[0]])
