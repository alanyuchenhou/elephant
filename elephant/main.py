import json
import os
import numpy
import sklearn
import pandas

from elephant.estimator import Estimator


def main(data_set_name):
    with open(os.path.join('../specs', data_set_name + '.json')) as specs_file:
        specs = json.load(specs_file)
    data_set = pandas.read_csv(os.path.join('../data', specs['file']), sep=specs['separator'], engine=specs['engine'],
                               skiprows=specs['header_rows'])
    print(data_set.head())
    with open(os.path.join(os.path.dirname(__file__), 'neural-net.json')) as config_file:
        config = json.load(config_file)
    x = data_set.ix[:, :2].values
    estimator = Estimator(config, x)
    y = data_set.ix[:, 2].values.reshape(-1, 1)
    if specs['scaling']:
        y = sklearn.preprocessing.MaxAbsScaler().fit_transform(numpy.log(y))
    print('testing_error =', estimator.estimate(y, config['batch_size'], specs['test_size'], specs['metric']))


if __name__ == '__main__':
    recommendation_data = ['movie-lens-100k', 'movie-lens-1m', 'e-pinions', 'movie-tweeting', ]
    graph_data = ['airport', 'collaboration', 'forum', ]
    main('forum')
