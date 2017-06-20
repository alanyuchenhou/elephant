import os
import shutil
import sys

import pandas
import tensorflow
from sklearn import model_selection
from tensorflow.contrib import learn, layers, tensorboard

ATTRIBUTES = [
    {
        'name': 'user',
        'type': str,
        'path': os.path.join(os.path.dirname(__file__), '../resources/movie-lens-100k-users.tsv'),
    },
    {
        'name': 'item',
        'type': str,
        'path': os.path.join(os.path.dirname(__file__), '../resources/movie-lens-100k-items.tsv'),
    },
    {
        'name': 'rating',
        'type': float,
        'path': os.path.join(os.path.dirname(__file__), '../resources/movie-lens-100k-ratings.tsv'),
    },
]
FEATURE_ATTRIBUTES = ATTRIBUTES[:2]


def configure_projector(model_dir):
    projector_configuration = tensorboard.plugins.projector.ProjectorConfig()
    for attribute in FEATURE_ATTRIBUTES:
        embedding = projector_configuration.embeddings.add()
        embedding.tensor_name = 'dnn/input_from_feature_columns/' + attribute['name'] + '_embedding/weights:0'
        embedding.metadata_path = attribute['path']
    tensorboard.plugins.projector.visualize_embeddings(
        tensorflow.summary.FileWriter(model_dir), projector_configuration,
    )


def input_fn(data_set):
    feature_tensor = {
        attribute['name']: tensorflow.SparseTensor(
            indices=[[i, 0] for i in range(data_set[attribute['name']].size)],
            values=data_set[attribute['name']].values,
            dense_shape=[data_set[attribute['name']].size, 1],
        ) for attribute in FEATURE_ATTRIBUTES
    }
    target_tensor = tensorflow.constant(data_set[ATTRIBUTES[2]['name']].values)
    return feature_tensor, target_tensor


def train_and_eval(model_dir, training_set, testing_set, ):
    sparse_columns = [
        layers.sparse_column_with_keys(
            attribute['name'], pandas.read_csv(attribute['path'], sep='\t')['id'].apply(str),
        ) for attribute in FEATURE_ATTRIBUTES
    ]
    embedding_columns = [layers.embedding_column(column, dimension=3) for column in sparse_columns]
    model = learn.DNNRegressor(
        hidden_units=[3, ],
        feature_columns=embedding_columns,
        model_dir=model_dir,
        config=learn.RunConfig(save_checkpoints_secs=1, ),
    )
    model.fit(input_fn=lambda: input_fn(training_set), steps=len(training_set), )
    results = model.evaluate(input_fn=lambda: input_fn(testing_set), steps=1)
    for key in sorted(results):
        print('%s: %s' % (key, results[key]))


def main(_):
    model_dir = os.path.join(os.path.dirname(__file__), '../log/model_r')
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    data_set = pandas.read_csv(
        ATTRIBUTES[2]['path'], sep='\t', dtype={attribute['name']: attribute['type'] for attribute in ATTRIBUTES},
    )
    data_set = data_set.dropna(how='any', axis=0, )
    training_set, testing_set = model_selection.train_test_split(data_set, test_size=0.2, )
    train_and_eval(model_dir, training_set, testing_set, )
    configure_projector(model_dir)


if __name__ == "__main__":
    tensorflow.app.run(main=main, argv=[sys.argv[0]])
