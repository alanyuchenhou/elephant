import sys
import tempfile
import urllib.request

import pandas
import tensorflow

COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num",
           "marital_status", "occupation", "relationship", "race", "gender",
           "capital_gain", "capital_loss", "hours_per_week", "native_country",
           "income_bracket"]
LABEL_COLUMN = "label"
CATEGORICAL_COLUMNS = ["workclass", "education", "marital_status", "occupation",
                       "relationship", "race", "gender", "native_country"]


def download_data():
    train_file = tempfile.NamedTemporaryFile(delete=False)
    urllib.request.urlretrieve("http://mlr.cs.umass.edu/ml/machine-learning-databases/adult/adult.data",
                               train_file.name)
    train_file_name = train_file.name
    train_file.close()
    print("Training data is downloaded to %s" % train_file_name)
    test_file = tempfile.NamedTemporaryFile(delete=False)
    urllib.request.urlretrieve("http://mlr.cs.umass.edu/ml/machine-learning-databases/adult/adult.test",
                               test_file.name)
    test_file_name = test_file.name
    test_file.close()
    print("Test data is downloaded to %s" % test_file_name)
    return train_file_name, test_file_name


def build_estimator(model_dir):
    sparse_attributes = [
        tensorflow.contrib.layers.sparse_column_with_hash_bucket(attribute, hash_bucket_size=1000)
        for attribute in CATEGORICAL_COLUMNS
    ]
    embedding_attributes = [
        tensorflow.contrib.layers.embedding_column(attribute, dimension=8)
        for attribute in sparse_attributes
    ]
    m = tensorflow.contrib.learn.DNNClassifier(
        model_dir=model_dir, feature_columns=embedding_attributes, hidden_units=[100, 50]
    )
    return m


def input_fn(df):
    categorical_cols = {
        attribute: tensorflow.SparseTensor(
            indices=[[i, 0] for i in range(df[attribute].size)],
            values=df[attribute].values,
            dense_shape=[df[attribute].size, 1]
        )
        for attribute in CATEGORICAL_COLUMNS
    }
    feature_cols = dict(categorical_cols)
    # Converts the label column into a constant Tensor.
    label = tensorflow.constant(df[LABEL_COLUMN].values)
    return feature_cols, label


def train_and_eval(train_steps, ):
    """Train and evaluate the model."""
    train_file_name, test_file_name = download_data()
    df_train = pandas.read_csv(
        tensorflow.gfile.Open(train_file_name),
        names=COLUMNS,
        skipinitialspace=True,
        engine="python")
    df_test = pandas.read_csv(
        tensorflow.gfile.Open(test_file_name),
        names=COLUMNS,
        skipinitialspace=True,
        skiprows=1,
        engine="python")

    # remove NaN elements
    df_train = df_train.dropna(how='any', axis=0)
    df_test = df_test.dropna(how='any', axis=0)
    df_train[LABEL_COLUMN] = (df_train["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)
    df_test[LABEL_COLUMN] = (df_test["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)
    model_dir = tempfile.mkdtemp()
    print("model directory = %s" % model_dir)

    m = build_estimator(model_dir)
    m.fit(input_fn=lambda: input_fn(df_train), steps=train_steps)
    results = m.evaluate(input_fn=lambda: input_fn(df_test), steps=1)
    for key in sorted(results):
        print("%s: %s" % (key, results[key]))


def main(_):
    train_and_eval(200, )


if __name__ == "__main__":
    tensorflow.app.run(main=main, argv=[sys.argv[0]])
