import json
import os

import numpy
import pandas
import sklearn

from elephant.estimator import Estimator


def main(data_set_name, layer_size, n_hidden_layers):
    with open(os.path.join('../specs', data_set_name + '.json')) as specs_file:
        specs = json.load(specs_file)
    data_set = pandas.read_csv(os.path.join('../resources', specs['file']), sep=specs['separator'],
                               engine=specs['engine'])
    print(data_set.head())
    with open(os.path.join(os.path.dirname(__file__), 'neural-net.json')) as config_file:
        config = json.load(config_file)
    x = data_set.ix[:, :2].values
    estimator = Estimator(x, config, layer_size, n_hidden_layers)
    y = data_set.ix[:, 2].values.reshape(-1, 1)
    if specs['scaling']:
        y = sklearn.preprocessing.MaxAbsScaler().fit_transform(numpy.log(y))
    return estimator.estimate(y, config['batch_size'], specs['test_size'], specs['metric'])


if __name__ == '__main__':
    recommendation_data = ['movie-lens-100k', 'movie-lens-1m', 'e-pinions', 'movie-tweeting', ]
    graph_data = ['airport', 'collaboration', 'congress', 'forum', ]
    MSEs = []
    for trial in range(25):
        MSEs.append(main('forum', 4, 1))
    MSEs = numpy.array(MSEs)
    print(MSEs.mean(), MSEs.std())
