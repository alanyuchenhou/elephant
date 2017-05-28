import os
import shutil

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)
IRIS_TRAINING = os.path.join(os.path.dirname(__file__), "iris_training.csv")
IRIS_TEST = os.path.join(os.path.dirname(__file__), "iris_test.csv")
LOG_DIR = "log/iris_model"


def main(unused_argv):
    training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=IRIS_TRAINING, target_dtype=np.int, features_dtype=np.float)
    test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=IRIS_TEST, target_dtype=np.int, features_dtype=np.float)
    validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
        test_set.data,
        test_set.target,
        every_n_steps=50,
        early_stopping_metric="loss",
        early_stopping_metric_minimize=True,
        early_stopping_rounds=200)
    # Specify that all features have real-value data
    feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]
    if os.path.exists(LOG_DIR):
        shutil.rmtree(LOG_DIR)
    classifier = tf.contrib.learn.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[10, 20, 10],
        n_classes=3,
        model_dir=LOG_DIR,
        config=tf.contrib.learn.RunConfig(save_checkpoints_secs=1))
    classifier.fit(x=training_set.data,
                   y=training_set.target,
                   steps=2000,
                   monitors=[validation_monitor])
    accuracy_score = classifier.evaluate(
        x=test_set.data, y=test_set.target)["accuracy"]
    print("Accuracy: {0:f}".format(accuracy_score))
    new_samples = np.array(
        [[6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7]], dtype=float)
    y = list(classifier.predict(new_samples))
    print("Predictions: {}".format(str(y)))


if __name__ == "__main__":
    tf.app.run()
