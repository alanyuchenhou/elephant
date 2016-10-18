import json
import os

import numpy
import pandas
import seaborn
import sklearn
from matplotlib import pyplot

from elephant.estimator import Estimator


def evaluate(data_set_name, layer_size, n_hidden_layers):
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


def experiment(data_set_name, layer_size, hidden_layer_count, ):
    MSEs = []
    for trial in range(8):
        MSEs.append(evaluate(data_set_name, layer_size, hidden_layer_count, ))
    MSEs = numpy.array(MSEs)
    print(MSEs.mean(), MSEs.std())
    return MSEs.mean()


def grid_search():
    MSEs = []
    layer_sizes = [1, 2, 4, 8, ]
    hidden_layer_counts = [1, 2, 4, 8, ]
    for layer_size in layer_sizes:
        MSEs.append(
            [experiment('airport', layer_size, hidden_layer_count) for hidden_layer_count in hidden_layer_counts])
    mses = pandas.DataFrame(numpy.array(MSEs), layer_sizes, hidden_layer_counts)
    print(mses)
    axes = seaborn.heatmap(mses, annot=True, )
    axes.set_ylabel('layer sizes')
    axes.set_xlabel('hidden layer count')
    pyplot.savefig('../resources/heat-map')


def main():
    # grid_search()
    experiment('airport', 2, 1, )


if __name__ == '__main__':
    recommendation_data = ['movie-lens-100k', 'movie-lens-1m', 'e-pinions', 'movie-tweeting', ]
    graph_data = ['airport', 'collaboration', 'congress', 'forum', ]
    main()
